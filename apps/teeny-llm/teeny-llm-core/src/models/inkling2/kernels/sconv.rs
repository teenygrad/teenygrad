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

//! Inkling-2's short convolution: a causal depthwise `Conv1d` of width `K` over the
//! token axis, one filter per channel.
//!
//! The checkpoint stores each filter as `[C, 1, K]`, PyTorch's depthwise layout, and
//! the port flattens it to `[C, K]`. Every channel is independent, so this is a
//! per-channel dot product over the last `K` tokens, not a matmul:
//!
//! ```text
//! y[t, c] = sum(k in 0..K) w[c, k] * x[t + k, c]
//! ```
//!
//! **History is a second input, not a prefix.** The convolution takes `hist` of
//! `[K - 1, C]` and `x` of `[T, C]` separately, and reads
//! `[hist; x]` as one `[T + K - 1, C]` stack. The caller keeps `hist` the way it
//! keeps the KV cache, with zeros at the start of a sequence.
//!
//! Taking them apart matters: inside a decoder layer the convolution's input is
//! produced by the graph itself (the attention output, the MoE output), so it cannot
//! be a host tensor with history already glued on. Only the history is host-side.
//!
//! Grid: `ceil(C / BLOCK)` by `T`.

use teeny_core::dtype::{AddOffsets, Comparison, Float, Tensor};
use teeny_macros::kernel;
use teeny_triton::triton::*;

/// The causal depthwise convolution above.
#[allow(clippy::multiple_bound_locations)]
#[kernel]
pub fn sconv_forward<T: Triton, D: Float, const BLOCK: i32, const K: i32>(
    h_ptr: InPtr<T::Pointer<D>>,
    x_ptr: InPtr<T::Pointer<D>>,
    w_ptr: InPtr<T::Pointer<D>>,
    y_ptr: OutPtr<T::Pointer<D>>,
    C: i32,
) where
    T: Triton<I32Tensor = <T as Triton>::Tensor<i32>>,
    T::I32Tensor: Tensor<i32, 1>,
    T::I32Tensor: core::ops::Add<i32, Output = T::I32Tensor>,
    T::I32Tensor: core::ops::Mul<i32, Output = T::I32Tensor>,
    T::I32Tensor: Comparison<i32, BoolTensor = T::BoolTensor>,
    T::Pointer<D>: AddOffsets<i32, 1, T::I32Tensor, Output = T::Tensor<T::Pointer<D>>>,
{
    let cid = T::program_id(Axis::X);
    let t = T::program_id(Axis::Y);
    let offs = T::arange(0, BLOCK) + cid * BLOCK;
    let m = offs.lt(C);
    let zeros = T::zeros::<D>(&[BLOCK]);

    let mut acc = T::zeros::<D>(&[BLOCK]);
    // K is a const generic, so this unrolls: K loads and K multiply-adds per channel.
    for k in 0..K {
        // Row `t + k` of the virtual [hist; x] stack: the first K - 1 rows are
        // history, the rest are this step's tokens.
        let row = t + k;
        let in_hist = row < K - 1;
        let hrow = if in_hist { row } else { 0 };
        let xrow = if in_hist { 0 } else { row - (K - 1) };
        let x = if in_hist {
            T::load(
                h_ptr.add_offsets(offs + hrow * C),
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
                x_ptr.add_offsets(offs + xrow * C),
                Some(m),
                Some(zeros),
                &[],
                None,
                None,
                None,
                false,
            )
        };
        let w = T::load(
            w_ptr.add_offsets(offs * K + k),
            Some(m),
            Some(zeros),
            &[],
            None,
            None,
            None,
            false,
        );
        acc = acc + x * w;
    }
    T::store(
        y_ptr.add_offsets(offs + t * C),
        acc,
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

/// Graph-level short convolution: inputs `hist` `[K - 1, C]` and `x` `[T, C]`,
/// output `[T, C]`.
pub struct SconvOp<D: FloatBytes + Send + Sync + 'static> {
    /// Channels.
    pub channels: usize,
    /// Filter width. The shipped checkpoints use 4.
    pub width: usize,
    _phantom: PhantomData<D>,
}

impl<D: FloatBytes + Send + Sync + 'static> SconvOp<D> {
    /// A convolution of `width` taps over `channels` channels.
    pub fn new(channels: usize, width: usize) -> Self {
        assert_eq!(width, 4, "sconv: only the shipped width 4 is compiled");
        Self {
            channels,
            width,
            _phantom: PhantomData,
        }
    }
}

impl<D: FloatBytes + Send + Sync + 'static> CustomOp for SconvOp<D> {
    fn name(&self) -> &str {
        "llm.sconv"
    }

    fn infer_output_shape(&self, input_shapes: &[&Shape]) -> Shape {
        // Shaped by `x`, the second input; the history adds no rows of its own.
        vec![input_shapes[1][0], Some(self.channels)]
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn lower(&self) -> Option<(String, String, String, Arc<dyn RuntimeOp>)> {
        let kernel = SconvForward::<D>::new(BLOCK, 4);
        Some((
            "sconv_forward".to_string(),
            kernel.source,
            "entry_point".to_string(),
            Arc::new(SconvRuntimeOp::<D> {
                channels: self.channels,
                width: self.width,
                _phantom: PhantomData,
            }),
        ))
    }
}

/// Runtime dispatch for [`SconvOp`].
pub struct SconvRuntimeOp<D: FloatBytes + Send + Sync + 'static> {
    channels: usize,
    width: usize,
    _phantom: PhantomData<D>,
}

impl<D: FloatBytes + Send + Sync + 'static> RuntimeOp for SconvRuntimeOp<D> {
    fn n_activation_inputs(&self) -> usize {
        2
    }

    fn param_shapes(&self, _input_shapes: &[&[usize]], _output_shape: &[usize]) -> Vec<Vec<usize>> {
        vec![vec![self.channels, self.width]]
    }

    fn param_names(&self) -> &'static [&'static str] {
        &["weight"]
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
        assert_eq!(inputs[1].1[1], self.channels, "sconv: input channels");
        assert_eq!(
            inputs[0].1[0],
            self.width - 1,
            "sconv: history must be {} rows, not {}",
            self.width - 1,
            inputs[0].1[0]
        );
        visitor.visit_ptr(inputs[0].0);
        visitor.visit_ptr(inputs[1].0);
        visitor.visit_ptr(params[0]);
        visitor.visit_ptr(output);
        visitor.visit_i32(self.channels as i32);
    }

    fn grid(&self, output_shape: &[usize]) -> [u32; 3] {
        [
            self.channels.div_ceil(BLOCK as usize) as u32,
            output_shape[0] as u32,
            1,
        ]
    }
}
