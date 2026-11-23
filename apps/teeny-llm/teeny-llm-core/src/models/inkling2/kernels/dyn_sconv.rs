/*
 * Copyright (c) 2026 teenygrad (https://teenygrad.org).
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

//! Inkling-2's **dynamic** short convolution
//! (`Inkling2LoopDynamicShortConvolution`), used for keys and values on the last
//! layer of each block (5, 11, 17 and 23 in Mini).
//!
//! The taps are a learned per-channel part plus a per-token part projected from the
//! layer input and shared by every channel of a head:
//!
//! ```text
//! w[t, c, j] = static[j, c] + dyn[t, head(c) * K + j]
//! conv[t, c] = sum(j in 0..K) w[t, c, j] * x[t - j, c]
//! y[t, c]    = (conv[t, c] + x[t, c]) / sqrt(2K + 1)
//! ```
//!
//! **The tap index is the backward distance here**, so `j = 0` is the current token.
//! That is the opposite of [`super::sconv`], where the current token is the last tap,
//! and the reference says so in a comment rather than in the shapes: getting it
//! backwards would reverse the filter and still produce plausible numbers.
//!
//! Note `sqrt(2K + 1)`, not the static convolution's `sqrt(K + 1)`.
//!
//! `dyn` arrives already projected ([`crate::kernels::linear::Linear4Op`] over
//! `weight_proj`), so this kernel does no matmul. History is a separate input, as in
//! [`super::sconv`].
//!
//! Grid: `ceil(C / BLOCK)` by `T`.

use teeny_core::dtype::{AddOffsets, Comparison, Float, Tensor};
use teeny_macros::kernel;
use teeny_triton::triton::*;

/// The dynamic convolution above.
#[allow(clippy::multiple_bound_locations)]
#[allow(clippy::too_many_arguments)]
#[kernel]
pub fn dyn_sconv_forward<T: Triton, D: Float, const BLOCK: i32, const K: i32>(
    h_ptr: InPtr<T::Pointer<D>>,
    x_ptr: InPtr<T::Pointer<D>>,
    d_ptr: InPtr<T::Pointer<D>>,
    w_ptr: InPtr<T::Pointer<D>>,
    y_ptr: OutPtr<T::Pointer<D>>,
    C: i32,
    HEAD_DIM: i32,
    DYN_STRIDE: i32,
    NORM: f32,
) where
    T: Triton<I32Tensor = <T as Triton>::Tensor<i32>>,
    T::I32Tensor: Tensor<i32, 1>,
    T::I32Tensor: core::ops::Add<i32, Output = T::I32Tensor>,
    T::I32Tensor: core::ops::Add<T::I32Tensor, Output = T::I32Tensor>,
    T::I32Tensor: core::ops::Mul<i32, Output = T::I32Tensor>,
    T::I32Tensor: core::ops::Div<i32, Output = T::I32Tensor>,
    T::I32Tensor: Comparison<i32, BoolTensor = T::BoolTensor>,
    T::Pointer<D>: AddOffsets<i32, 1, T::I32Tensor, Output = T::Tensor<T::Pointer<D>>>,
{
    let bid = T::program_id(Axis::X);
    let t = T::program_id(Axis::Y);
    let offs = T::arange(0, BLOCK) + bid * BLOCK;
    let m = offs.lt(C);
    let zeros = T::zeros::<D>(&[BLOCK]);

    // The dynamic taps are per head, so every channel of a head reads the same K.
    let head = offs / HEAD_DIM;

    let mut acc = T::zeros::<D>(&[BLOCK]);
    for j in 0..K {
        // j is the backward distance: row t - j of the virtual [hist; x] stack, whose
        // first K - 1 rows are history and whose row t + K - 1 is the current token.
        let row = t + (K - 1) - j;
        let in_hist = row < K - 1;
        let x = if in_hist {
            T::load(
                h_ptr.add_offsets(offs + row * C),
                Some(m),
                Some(zeros),
                &[],
                None,
                None,
                None,
                false,
            )
        } else {
            T::load(
                x_ptr.add_offsets(offs + (row - (K - 1)) * C),
                Some(m),
                Some(zeros),
                &[],
                None,
                None,
                None,
                false,
            )
        };
        let sw = T::load(
            w_ptr.add_offsets(offs + j * C),
            Some(m),
            Some(zeros),
            &[],
            None,
            None,
            None,
            false,
        );
        // dyn is [T, heads * K]: this token's row, then this head's taps.
        let dw = T::load(
            d_ptr.add_offsets(head * K + (t * DYN_STRIDE + j)),
            Some(m),
            Some(zeros),
            &[],
            None,
            None,
            None,
            false,
        );
        acc = acc + x * (sw + dw);
    }

    let cur = T::load(
        x_ptr.add_offsets(offs + t * C),
        Some(m),
        Some(zeros),
        &[],
        None,
        None,
        None,
        false,
    );
    // 1 / sqrt(2K + 1), computed by the host: the kernel environment has no sqrt.
    let norm = T::full(&[BLOCK], D::from_f64(NORM as f64));
    T::store(
        y_ptr.add_offsets(offs + t * C),
        (acc + cur) * norm,
        Some(m),
        &[],
        None,
        None,
    );
}

// ---------------------------------------------------------------------------
// Graph-level op
// ---------------------------------------------------------------------------

use core::marker::PhantomData;
use std::any::Any;
use std::sync::Arc;
use teeny_core::device::program::ArgVisitor;
use teeny_core::dtype::FloatBytes;
use teeny_core::graph::{CustomOp, Shape};
use teeny_core::model::{RawPtr, RuntimeOp};

const BLOCK: i32 = 256;

/// Graph-level dynamic short convolution: inputs `hist` `[K - 1, C]`, `x` `[T, C]`
/// and `dyn` `[T, heads * K]`, parameter `static_w` `[K, C]`, output `[T, C]`.
pub struct DynSconvOp<D: FloatBytes + Send + Sync + 'static> {
    /// Channels.
    pub channels: usize,
    /// Key/value heads the dynamic taps are shared within.
    pub heads: usize,
    /// Filter width. The shipped checkpoints use 4.
    pub width: usize,
    _phantom: PhantomData<D>,
}

impl<D: FloatBytes + Send + Sync + 'static> DynSconvOp<D> {
    /// A convolution of `width` taps over `channels` channels in `heads` groups.
    pub fn new(channels: usize, heads: usize, width: usize) -> Self {
        assert_eq!(width, 4, "dyn sconv: only the shipped width 4 is compiled");
        assert!(
            heads > 0 && channels.is_multiple_of(heads),
            "dyn sconv: {channels} channels do not divide into {heads} heads"
        );
        Self {
            channels,
            heads,
            width,
            _phantom: PhantomData,
        }
    }
}

impl<D: FloatBytes + Send + Sync + 'static> CustomOp for DynSconvOp<D> {
    fn name(&self) -> &str {
        "llm.dyn_sconv"
    }

    fn infer_output_shape(&self, input_shapes: &[&Shape]) -> Shape {
        vec![input_shapes[1][0], Some(self.channels)]
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn lower(&self) -> Option<(String, String, String, Arc<dyn RuntimeOp>)> {
        let kernel = DynSconvForward::<D>::new(BLOCK, 4);
        Some((
            "dyn_sconv_forward".to_string(),
            kernel.source,
            "entry_point".to_string(),
            Arc::new(DynSconvRuntimeOp::<D> {
                channels: self.channels,
                heads: self.heads,
                width: self.width,
                _phantom: PhantomData,
            }),
        ))
    }
}

/// Runtime dispatch for [`DynSconvOp`].
pub struct DynSconvRuntimeOp<D: FloatBytes + Send + Sync + 'static> {
    channels: usize,
    heads: usize,
    width: usize,
    _phantom: PhantomData<D>,
}

impl<D: FloatBytes + Send + Sync + 'static> RuntimeOp for DynSconvRuntimeOp<D> {
    fn n_activation_inputs(&self) -> usize {
        3
    }

    fn param_shapes(&self, _input_shapes: &[&[usize]], _output_shape: &[usize]) -> Vec<Vec<usize>> {
        // The checkpoint stores it as [K, hidden], tap-major.
        vec![vec![self.width, self.channels]]
    }

    fn param_names(&self) -> &'static [&'static str] {
        &["static_w"]
    }

    fn pack_args(
        &self,
        inputs: &[(RawPtr, &[usize])],
        params: &[RawPtr],
        output: RawPtr,
        _output_shape: &[usize],
        _output_row_stride: i32,
        visitor: &mut dyn ArgVisitor,
    ) {
        assert_eq!(inputs[1].1[1], self.channels, "dyn sconv: input channels");
        assert_eq!(
            inputs[0].1[0],
            self.width - 1,
            "dyn sconv: history must be {} rows",
            self.width - 1
        );
        assert_eq!(
            inputs[2].1[1],
            self.heads * self.width,
            "dyn sconv: the projected taps must be heads * width wide"
        );
        for (ptr, _) in &inputs[..3] {
            visitor.visit_ptr(*ptr);
        }
        visitor.visit_ptr(params[0]);
        visitor.visit_ptr(output);
        visitor.visit_i32(self.channels as i32);
        visitor.visit_i32((self.channels / self.heads) as i32);
        visitor.visit_i32((self.heads * self.width) as i32);
        visitor.visit_f32(1.0 / ((2 * self.width + 1) as f32).sqrt());
    }

    fn grid(&self, output_shape: &[usize]) -> [u32; 3] {
        [
            self.channels.div_ceil(BLOCK as usize) as u32,
            output_shape[0] as u32,
            1,
        ]
    }
}
