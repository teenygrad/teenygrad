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

//! The expert gather with **MXFP8** weights, for layers 20 to 23.
//!
//! Those layers quantise their experts differently from the rest, and the layout is
//! the inverse of [`super::expert_gather_nvfp4`]:
//!
//! | | NVFP4 (layers 0-19) | MXFP8 (layers 20-23) |
//! |---|---|---|
//! | weight | `U8`, two E2M1 nibbles per byte | `F8_E4M3`, one byte per weight |
//! | scale | `F8_E4M3`, one per **16** | `U8` E8M0, one per **32** |
//! | per expert | `amax` and a denominator | none |
//!
//! ```text
//! y[s, n] = sum(k) x[token(s), k] * e4m3(w[expert(s), n, k]) * 2^(scale[k / 32] - 127)
//! ```
//!
//! E8M0 is exponent-only: the byte is a biased power of two, so the scale needs no
//! mantissa work, just `exp2(b - 127)`. There is no nibble unpacking either, which
//! makes this the simpler of the two decoders, though at one byte per weight the
//! layers are twice the size of an NVFP4 one.
//!
//! **Packed, four bytes to an f32 element**, as in
//! [`super::expert_gather_nvfp4`]: the weight bytes and the E8M0 scales both travel
//! inside f32 elements and are masked back out after a bitcast. One byte per weight
//! here, so these layers are twice the size of an NVFP4 one even packed.

use teeny_core::dtype::{AddOffsets, Comparison, Float, Tensor};
use teeny_macros::kernel;
use teeny_triton::triton::*;

/// The gathered projection above.
#[allow(clippy::multiple_bound_locations)]
#[kernel]
pub fn expert_gather_mxfp8_forward<T: Triton, D: Float, const BLOCK: i32>(
    x_ptr: InPtr<T::Pointer<D>>,
    b_ptr: InPtr<T::Pointer<D>>,
    s_ptr: InPtr<T::Pointer<D>>,
    ids_ptr: InPtr<T::Pointer<D>>,
    y_ptr: OutPtr<T::Pointer<D>>,
    N: i32,
    K: i32,
    TOP_K: i32,
    IDS_STRIDE: i32,
    IDS_OFFSET: i32,
    BWORDS: i32,
    SWORDS: i32,
) where
    T: Triton<I32Tensor = <T as Triton>::Tensor<i32>>,
    T::I32Tensor: Tensor<i32, 1>,
    T::I32Tensor: core::ops::Add<i32, Output = T::I32Tensor>,
    T::I32Tensor: core::ops::Add<T::I32Tensor, Output = T::I32Tensor>,
    T::I32Tensor: core::ops::Sub<T::I32Tensor, Output = T::I32Tensor>,
    T::I32Tensor: core::ops::Div<i32, Output = T::I32Tensor>,
    T::I32Tensor: core::ops::BitAnd<i32, Output = T::I32Tensor>,
    T::I32Tensor: core::ops::Mul<i32, Output = T::I32Tensor>,
    T::I32Tensor: Comparison<i32, BoolTensor = T::BoolTensor>,
    T::Pointer<D>: AddOffsets<i32, 1, T::I32Tensor, Output = T::Tensor<T::Pointer<D>>>,
{
    let n = T::program_id(Axis::X);
    let slot = T::program_id(Axis::Y);
    let token = slot / TOP_K;
    let which = slot - token * TOP_K;

    let offs = T::arange(0, BLOCK);
    let zeros = T::zeros::<D>(&[BLOCK]);

    // The id sits in the router's packed row as a float; broadcast it so it can take
    // part in the block-wide address arithmetic below.
    let id = T::load(
        ids_ptr.add_offsets(T::zeros::<i32>(&[BLOCK]) + (token * IDS_STRIDE + IDS_OFFSET + which)),
        None,
        Some(zeros),
        &[],
        None,
        None,
        None,
        false,
    );
    let expert = T::cast::<D, i32>(id, None, false);

    let mut acc = T::zeros::<D>(&[BLOCK]);
    let one = T::full(&[BLOCK], D::from_f64(1.0));
    let eighth = T::full(&[BLOCK], D::from_f64(0.125));
    let seven = T::full(&[BLOCK], D::from_f64(7.0));
    let sub6 = T::full(&[BLOCK], D::from_f64(0.015_625));
    let bias = T::full(&[BLOCK], D::from_f64(127.0));
    let n_chunks = (K + BLOCK - 1) / BLOCK;
    for i in 0..n_chunks {
        let k = offs + i * BLOCK;
        let km = k.lt(K);
        let x = T::load(
            x_ptr.add_offsets(k + token * K),
            Some(km),
            Some(zeros),
            &[],
            None,
            None,
            None,
            false,
        );
        // One byte per weight and one E8M0 scale per 32, packed four to an element.
        let brow = expert * (N * BWORDS) + (n * BWORDS);
        let srow = expert * (N * SWORDS) + (n * SWORDS);

        let wword = T::cast::<D, i32>(
            T::load(
                b_ptr.add_offsets(k / 4 + brow),
                Some(km),
                Some(zeros),
                &[],
                None,
                None,
                None,
                true,
            ),
            None,
            true,
        );
        let bsel = k - ((k / 4) * 4);
        let b = {
            let b0 = wword & 0xFF;
            let b1 = (wword & 0xFF00) / 256;
            let b2 = (wword & 0x00FF_0000) / 65536;
            let low7 = (wword & 0x7F00_0000) / 16_777_216;
            let b3 = T::where_(wword.lt(0), low7 + 128, low7 + 0);
            T::where_(
                bsel.eq(0),
                b0,
                T::where_(bsel.eq(1), b1, T::where_(bsel.eq(2), b2, b3)),
            )
        };
        // E4M3: s:1 e:4 m:3.
        let sgn = b / 128;
        let rest = b - sgn * 128;
        let e = rest / 8;
        let mant = T::cast::<i32, D>(rest - e * 8, None, false);
        let ef = T::cast::<i32, D>(e, None, false);
        let mag = T::where_(
            e.eq(0),
            mant * eighth * sub6,
            (one + mant * eighth) * T::exp2(ef - seven),
        );
        let val = T::where_(sgn.eq(1), zeros - mag, mag);

        // E8M0 is exponent-only: the byte is a biased power of two.
        let sword = T::cast::<D, i32>(
            T::load(
                s_ptr.add_offsets(k / 128 + srow),
                Some(km),
                Some(zeros),
                &[],
                None,
                None,
                None,
                true,
            ),
            None,
            true,
        );
        let ssel = (k / 32) - ((k / 128) * 4);
        let sb = {
            let s0 = sword & 0xFF;
            let s1 = (sword & 0xFF00) / 256;
            let s2 = (sword & 0x00FF_0000) / 65536;
            let slow7 = (sword & 0x7F00_0000) / 16_777_216;
            let s3 = T::where_(sword.lt(0), slow7 + 128, slow7 + 0);
            T::where_(
                ssel.eq(0),
                s0,
                T::where_(ssel.eq(1), s1, T::where_(ssel.eq(2), s2, s3)),
            )
        };
        let sval = T::exp2(T::cast::<i32, D>(sb, None, false) - bias);

        acc = acc + x * (val * sval);
    }
    // MXFP8 has no per-expert scale: the block scales are the whole story.
    let out = T::sum(acc, Some(0), true);
    T::store(
        y_ptr.add_offsets(T::zeros::<i32>(&[BLOCK]) + (slot * N + n)),
        out,
        Some(offs.lt(1)),
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

/// Graph-level expert gather. Inputs are `x` of `[T, K]` and the router's packed
/// `[T, 2 * top_k]`; the output is `[T * top_k, N]`.
pub struct ExpertGatherMxfp8Op<D: FloatBytes + Send + Sync + 'static> {
    /// Experts the parameter holds.
    pub experts: usize,
    /// Outputs per expert row.
    pub n: usize,
    /// Reduction width.
    pub k: usize,
    /// Experts firing per token.
    pub top_k: usize,
    _phantom: PhantomData<D>,
}

impl<D: FloatBytes + Send + Sync + 'static> ExpertGatherMxfp8Op<D> {
    /// A gather over `experts` planes of `[n, k]`, with `top_k` slots per token.
    pub fn new(experts: usize, n: usize, k: usize, top_k: usize) -> Self {
        assert!(
            top_k > 0 && top_k <= experts,
            "expert gather: top_k out of range"
        );
        Self {
            experts,
            n,
            k,
            top_k,
            _phantom: PhantomData,
        }
    }
}

impl<D: FloatBytes + Send + Sync + 'static> CustomOp for ExpertGatherMxfp8Op<D> {
    fn name(&self) -> &str {
        "llm.expert_gather_mxfp8"
    }

    fn infer_output_shape(&self, input_shapes: &[&Shape]) -> Shape {
        vec![input_shapes[0][0].map(|t| t * self.top_k), Some(self.n)]
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn lower(&self) -> Option<(String, String, String, Arc<dyn RuntimeOp>)> {
        let kernel = ExpertGatherMxfp8Forward::<D>::new(BLOCK);
        Some((
            "expert_gather_mxfp8_forward".to_string(),
            kernel.source,
            "entry_point".to_string(),
            Arc::new(ExpertGatherMxfp8RuntimeOp::<D> {
                experts: self.experts,
                n: self.n,
                k: self.k,
                top_k: self.top_k,
                _phantom: PhantomData,
            }),
        ))
    }
}

/// Runtime dispatch for [`ExpertGatherMxfp8Op`].
pub struct ExpertGatherMxfp8RuntimeOp<D: FloatBytes + Send + Sync + 'static> {
    experts: usize,
    n: usize,
    k: usize,
    top_k: usize,
    _phantom: PhantomData<D>,
}

impl<D: FloatBytes + Send + Sync + 'static> RuntimeOp for ExpertGatherMxfp8RuntimeOp<D> {
    fn n_activation_inputs(&self) -> usize {
        // x and the router's packed row; the weights are parameters.
        2
    }

    fn param_shapes(&self, _input_shapes: &[&[usize]], _output_shape: &[usize]) -> Vec<Vec<usize>> {
        vec![
            vec![self.experts, self.n, self.k.div_ceil(4)],
            vec![self.experts, self.n, (self.k / 32).div_ceil(4)],
        ]
    }

    fn param_names(&self) -> &'static [&'static str] {
        &["weight", "scale"]
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
        assert_eq!(inputs[0].1[1], self.k, "expert gather: input width");
        let ids_stride = inputs[1].1[1];
        assert!(
            ids_stride >= self.top_k,
            "expert gather: router row of {ids_stride} cannot hold {} ids",
            self.top_k
        );
        visitor.visit_ptr(inputs[0].0);
        visitor.visit_ptr(params[0]);
        visitor.visit_ptr(params[1]);
        visitor.visit_ptr(inputs[1].0);
        visitor.visit_ptr(output);
        visitor.visit_i32(self.n as i32);
        visitor.visit_i32(self.k as i32);
        visitor.visit_i32(self.top_k as i32);
        visitor.visit_i32(ids_stride as i32);
        // The ids sit after the weights in the router's packed row.
        visitor.visit_i32((ids_stride - self.top_k) as i32);
        // Words per row, four bytes each. div_ceil, because a row shorter than
        // one word still needs one.
        visitor.visit_i32(self.k.div_ceil(4) as i32);
        visitor.visit_i32((self.k / 32).div_ceil(4) as i32);
    }

    fn grid(&self, output_shape: &[usize]) -> [u32; 3] {
        [self.n as u32, output_shape[0] as u32, 1]
    }
}
