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

//! The expert gather with NVFP4 weights unpacked in its inner loop.
//!
//! Same shape as [`super::expert_gather`]:
//!
//! ```text
//! y[s, n] = sum(k) x[token(s), k] * W[expert(s), n, k]
//! ```
//!
//! but `W` is never materialised as f32. Each weight is decoded where it is used:
//! an E2M1 nibble, an E4M3 scale per 16 weights, and one f32 per expert, exactly as
//! [`super::nvfp4`] does it and as the RVV kernel in `experiments/nvfp4-riscv`
//! measures it (1.9x to 3.1x over the unfused form on a SpacemiT X100).
//!
//! This is what makes the model fit at all: the experts are 20.2 GB packed and about
//! 160 GB in f32, against a board's 31 GB.
//!
//! **Packed, four bytes to an f32 element.** Parameter buffers take the graph's
//! dtype, so the packed nibbles travel inside f32 elements and are taken apart here:
//! `T::cast` bitcasts one to an i32 and the byte is masked out of it. That is
//! 0.5 bytes per weight, the true NVFP4 density.
//!
//! Which of the four bytes a lane wants varies along `k`, so it is shifted out with
//! `wword >> (bsel * 8)` rather than branched on. That needed teenyc to lower `Shr`
//! on tensors, which it now does; before that the kernel computed all four bytes
//! with masks and divides and selected between them, and because `Div` truncates
//! towards zero the top byte had to be masked before being divided down and its sign
//! bit added back by hand.

use teeny_core::dtype::{AddOffsets, Comparison, Float, Tensor};
use teeny_macros::kernel;
use teeny_triton::triton::*;

/// The gathered projection above.
#[allow(clippy::multiple_bound_locations)]
#[kernel]
pub fn expert_gather_nvfp4_forward<T: Triton, D: Float, const BLOCK: i32>(
    x_ptr: InPtr<T::Pointer<D>>,
    b_ptr: InPtr<T::Pointer<D>>,
    s_ptr: InPtr<T::Pointer<D>>,
    ids_ptr: InPtr<T::Pointer<f32>>,
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
    T::I32Tensor: core::ops::Shr<i32, Output = T::I32Tensor>,
    T::I32Tensor: core::ops::Shr<T::I32Tensor, Output = T::I32Tensor>,
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

    let mut acc = T::zeros::<D>(&[BLOCK]);

    // The sixteen E2M1 values, built once per program and then read per lane with
    // `T::gather`, which is one `vrgather.vv` on RVV. Code c is sign (c >> 3), a
    // two-bit exponent and a one-bit mantissa: 0, 0.5, 1, 1.5, 2, 3, 4, 6, then
    // the same negated.
    let code = T::arange(0, 16);
    let half = T::full(&[16], D::from_f64(0.5));
    let one = T::full(&[16], D::from_f64(1.0));
    let two = T::full(&[16], D::from_f64(2.0));
    let four = T::full(&[16], D::from_f64(4.0));
    let crest = code & 7;
    let ce = crest >> 1;
    let cmant = T::cast::<i32, D>(crest & 1, None, false);
    let cp = T::where_(ce.eq(1), one, T::where_(ce.eq(2), two, four));
    let cmag = T::where_(ce.eq(0), cmant * half, (one + cmant * half) * cp);
    let lut = T::where_((code >> 3).eq(1), T::zeros::<D>(&[16]) - cmag, cmag);

    // Lane k of a chunk wants word k / 8, nibble k % 8 of it, and scale k / 16.
    // The words and scales a chunk needs are contiguous, so they are loaded as
    // BLOCK / 8 and BLOCK / 16 lanes and spread across the BLOCK lanes with an
    // in-register gather; a per-lane address would be an indexed load.
    let o8 = offs >> 3;
    let o16 = offs >> 4;
    let shift = (offs & 7) * 4;
    let words_off = T::arange(0, BLOCK / 8);
    let scales_off = T::arange(0, BLOCK / 16);
    // The expert id sits in the router's packed row as a float. Read as a scalar,
    // it makes every row offset below a scalar too, so each load's offsets are
    // `arange + scalar` and the compiler can prove them contiguous; a loaded tensor
    // would leave it per-lane indexed loads.
    let expert = T::load_scalar_f32_as_i32(ids_ptr, token * IDS_STRIDE + IDS_OFFSET + which);
    let brow = expert * (N * BWORDS) + n * BWORDS;
    let srow = expert * (N * SWORDS) + n * SWORDS;

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

        let wo = words_off + i * (BLOCK / 8);
        let words = T::cast::<D, i32>(
            T::load(
                b_ptr.add_offsets(wo + brow),
                Some(wo.lt(BWORDS)),
                Some(T::zeros::<D>(&[BLOCK / 8])),
                &[],
                None,
                None,
                None,
                true,
            ),
            None,
            true,
        );
        let nib = (T::gather(words, o8, 0) >> shift) & 0xF;
        let val = T::gather(lut, nib, 0);

        let so = scales_off + i * (BLOCK / 16);
        let scales = T::load(
            s_ptr.add_offsets(so + srow),
            Some(so.lt(SWORDS)),
            Some(T::zeros::<D>(&[BLOCK / 16])),
            &[],
            None,
            None,
            None,
            false,
        );
        let sval = T::gather(scales, o16, 0);
        acc = acc + x * (val * sval);
    }
    // One scalar per program: reduce the lanes, then write a single element. The
    // per-expert scale is already folded into the block scale.
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

// 32 lanes, not a whole vector register file's worth. At `e32` with VLEN=256 a
// 256-lane tensor is 32 vector registers, so each of the dequant's ~15 live
// intermediates needs the entire register file and almost all of them spill: 229
// whole-register spill/fill operations against ~27 arithmetic ones. At 32 that is 6.
// Smaller still (16) saves little more and shortens the vector ops, so 32 it is.
const BLOCK: i32 = 32;

/// Graph-level expert gather. Inputs are `x` of `[T, K]` and the router's packed
/// `[T, 2 * top_k]`; the output is `[T * top_k, N]`.
pub struct ExpertGatherNvfp4Op<D: FloatBytes + Send + Sync + 'static> {
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

impl<D: FloatBytes + Send + Sync + 'static> ExpertGatherNvfp4Op<D> {
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

impl<D: FloatBytes + Send + Sync + 'static> CustomOp for ExpertGatherNvfp4Op<D> {
    fn name(&self) -> &str {
        "llm.expert_gather_nvfp4"
    }

    fn infer_output_shape(&self, input_shapes: &[&Shape]) -> Shape {
        vec![input_shapes[0][0].map(|t| t * self.top_k), Some(self.n)]
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn lower(&self) -> Option<(String, String, String, Arc<dyn RuntimeOp>)> {
        let kernel = ExpertGatherNvfp4Forward::<D>::new(BLOCK);
        Some((
            "expert_gather_nvfp4_forward".to_string(),
            kernel.source,
            "entry_point".to_string(),
            Arc::new(ExpertGatherNvfp4RuntimeOp::<D> {
                experts: self.experts,
                n: self.n,
                k: self.k,
                top_k: self.top_k,
                _phantom: PhantomData,
            }),
        ))
    }
}

/// Runtime dispatch for [`ExpertGatherNvfp4Op`].
pub struct ExpertGatherNvfp4RuntimeOp<D: FloatBytes + Send + Sync + 'static> {
    experts: usize,
    n: usize,
    k: usize,
    top_k: usize,
    _phantom: PhantomData<D>,
}

impl<D: FloatBytes + Send + Sync + 'static> RuntimeOp for ExpertGatherNvfp4RuntimeOp<D> {
    fn n_activation_inputs(&self) -> usize {
        // x and the router's packed row; the weights are parameters.
        2
    }

    fn param_shapes(&self, _input_shapes: &[&[usize]], _output_shape: &[usize]) -> Vec<Vec<usize>> {
        vec![
            vec![self.experts, self.n, self.k.div_ceil(8)],
            // One f32 per 16-weight block: the E4M3 block scale and the per-expert
            // scale multiplied together at load time.
            vec![self.experts, self.n, self.k / 16],
        ]
    }

    fn param_names(&self) -> &'static [&'static str] {
        &["weight", "folded_scale"]
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
        visitor.visit_i32(self.k.div_ceil(8) as i32);
        visitor.visit_i32((self.k / 16) as i32);
    }

    fn grid(&self, output_shape: &[usize]) -> [u32; 3] {
        [self.n as u32, output_shape[0] as u32, 1]
    }
}
