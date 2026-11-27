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

//! The MXFP8 expert projection (layers 20 to 23) with output rows as lanes.
//!
//! ```text
//! y[s, n] = sum(k) x[token(s), k] * e4m3(w[expert(s), n, k]) * 2^(scale[n, k / 32] - 127)
//! ```
//!
//! as [`super::expert_gather_mxfp8`] computes it, laid out like
//! [`super::expert_gemv_nvfp4`]: a program computes [`ROWS`] rows of one expert,
//! and the reduction runs down `k` with one activation per step broadcast across
//! the rows, so there is no horizontal reduction and every load is contiguous.
//!
//! - **Weights**, one E4M3 byte each, are interleaved at load time so the [`ROWS`]
//!   rows' bytes at each `k` are adjacent ([`super::expert_gemv_nvfp4::interleave_row_bytes`]).
//!   A byte becomes its f32 exactly by bit construction: a normal value's bits are
//!   `((b & 0x7F) + 960) << 20` (the exponent rebiased from 7 to 127, the mantissa
//!   moved to the top), and a subnormal one, exponent 0, is `m * 2^-9`.
//! - **Scales**, one E8M0 byte per row per 32 elements, are interleaved the same way
//!   and applied once per 32-element block: `2^(s - 127)` is `s << 23` as f32 bits.
//! - **Activations stay f32**: an E4M3 value is not an integer at any useful scale,
//!   and the cost here is the decode, not the multiply.

use teeny_core::dtype::{AddOffsets, Comparison, Float, Tensor};
use teeny_macros::kernel;
use teeny_triton::triton::*;

/// Output rows per program: one vector register of bytes, four of f32 sums, at
/// VLEN=256.
pub const ROWS: usize = 32;

/// Elements per E8M0 scale.
pub const SBLOCK: usize = 32;

/// The projection above, over [`ROWS`] rows of one routed expert per program.
#[allow(clippy::multiple_bound_locations)]
#[kernel]
pub fn expert_gemv_mxfp8_forward<T: Triton, D: Float, const ROWS: i32>(
    x_ptr: InPtr<T::Pointer<D>>,
    w_ptr: InPtr<T::Pointer<u8>>,
    s_ptr: InPtr<T::Pointer<u8>>,
    ids_ptr: InPtr<T::Pointer<f32>>,
    y_ptr: OutPtr<T::Pointer<D>>,
    N: i32,
    K: i32,
    TOP_K: i32,
    IDS_STRIDE: i32,
    IDS_OFFSET: i32,
) where
    T: Triton<I32Tensor = <T as Triton>::Tensor<i32>>,
    T::I32Tensor: Tensor<i32, 1>,
    T::I32Tensor: core::ops::Add<i32, Output = T::I32Tensor>,
    T::I32Tensor: core::ops::BitAnd<i32, Output = T::I32Tensor>,
    T::I32Tensor: core::ops::Shl<i32, Output = T::I32Tensor>,
    T::I32Tensor: Comparison<i32, BoolTensor = T::BoolTensor>,
    T::Pointer<D>: AddOffsets<i32, 1, T::I32Tensor, Output = T::Tensor<T::Pointer<D>>>,
    T::Pointer<u8>: AddOffsets<i32, 1, T::I32Tensor, Output = T::Tensor<T::Pointer<u8>>>,
{
    let rb = T::program_id(Axis::X);
    let slot = T::program_id(Axis::Y);
    let token = slot / TOP_K;
    let which = slot - token * TOP_K;
    let expert = T::load_scalar_f32_as_i32(ids_ptr, token * IDS_STRIDE + IDS_OFFSET + which);

    let lanes = T::arange(0, ROWS);
    let zeros = T::zeros::<D>(&[ROWS]);
    // The subnormal E4M3 step, m * 2^-9, and E8M0's zero, 2^-127, which bit
    // construction cannot produce.
    let sub9 = T::full(&[ROWS], D::from_f64(0.001_953_125));
    let tiny = T::full(&[ROWS], D::from_f64(5.877_471_754_111_438e-39));

    // [expert][row block][k][row] bytes, and [expert][row block][k / 32][row].
    let blocks = K / 32;
    let row_block = expert * (N / ROWS) + rb;
    let wbase = row_block * K * ROWS;
    let sbase = row_block * blocks * ROWS;
    let xbase = token * K;

    let mut acc = T::zeros::<D>(&[ROWS]);
    for b in 0..blocks {
        let mut bsum = T::zeros::<D>(&[ROWS]);
        for t in 0..32 {
            let k = b * 32 + t;
            let byte = T::cast::<u8, i32>(
                T::load(
                    w_ptr.add_offsets(lanes + (wbase + k * ROWS)),
                    None,
                    None,
                    &[],
                    None,
                    None,
                    None,
                    false,
                ),
                None,
                false,
            );
            let rest = byte & 0x7F;
            let normal = T::cast::<i32, D>((rest + 960) << 20, None, true);
            let mag = T::where_(
                rest.lt(8),
                T::cast::<i32, D>(rest, None, false) * sub9,
                normal,
            );
            let w = T::where_(byte.gt(127), zeros - mag, mag);
            let xk = T::load(
                x_ptr.add_offsets(T::zeros::<i32>(&[ROWS]) + (xbase + k)),
                None,
                None,
                &[],
                None,
                None,
                None,
                false,
            );
            bsum = bsum + w * xk;
        }
        let s = T::cast::<u8, i32>(
            T::load(
                s_ptr.add_offsets(lanes + (sbase + b * ROWS)),
                None,
                None,
                &[],
                None,
                None,
                None,
                false,
            ),
            None,
            false,
        );
        let scale = T::where_(s.eq(0), tiny, T::cast::<i32, D>(s << 23, None, true));
        acc = acc + bsum * scale;
    }
    T::store(
        y_ptr.add_offsets(lanes + (slot * N + rb * ROWS)),
        acc,
        None,
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

/// The MXFP8 expert projection. Inputs are `x` of `[T, k]` and the router's packed
/// `[T, >= 2 * top_k]` row; the output is `[T * top_k, n]`, as
/// [`super::expert_gather_mxfp8::ExpertGatherMxfp8Op`]'s is.
pub struct ExpertGemvMxfp8Op<D: FloatBytes + Send + Sync + 'static> {
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

impl<D: FloatBytes + Send + Sync + 'static> ExpertGemvMxfp8Op<D> {
    /// A projection over `experts` planes of `[n, k]`, with `top_k` slots per token.
    pub fn new(experts: usize, n: usize, k: usize, top_k: usize) -> Self {
        assert!(top_k > 0 && top_k <= experts, "mxfp8 gemv: top_k out of range");
        assert_eq!(n % ROWS, 0, "mxfp8 gemv: {n} rows are not a multiple of {ROWS}");
        assert_eq!(k % (4 * SBLOCK), 0, "mxfp8 gemv: {k} is not a multiple of {}", 4 * SBLOCK);
        Self { experts, n, k, top_k, _phantom: PhantomData }
    }
}

impl<D: FloatBytes + Send + Sync + 'static> CustomOp for ExpertGemvMxfp8Op<D> {
    fn name(&self) -> &str {
        "llm.expert_gemv_mxfp8"
    }

    fn infer_output_shape(&self, input_shapes: &[&Shape]) -> Shape {
        vec![input_shapes[0][0].map(|t| t * self.top_k), Some(self.n)]
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn lower(&self) -> Option<(String, String, String, Arc<dyn RuntimeOp>)> {
        let kernel = ExpertGemvMxfp8Forward::<D>::new(ROWS as i32);
        Some((
            "expert_gemv_mxfp8_forward".to_string(),
            kernel.source,
            "entry_point".to_string(),
            Arc::new(ExpertGemvMxfp8RuntimeOp::<D> {
                experts: self.experts,
                n: self.n,
                k: self.k,
                top_k: self.top_k,
                _phantom: PhantomData,
            }),
        ))
    }
}

/// Runtime dispatch for [`ExpertGemvMxfp8Op`].
pub struct ExpertGemvMxfp8RuntimeOp<D: FloatBytes + Send + Sync + 'static> {
    experts: usize,
    n: usize,
    k: usize,
    top_k: usize,
    _phantom: PhantomData<D>,
}

impl<D: FloatBytes + Send + Sync + 'static> RuntimeOp for ExpertGemvMxfp8RuntimeOp<D> {
    fn n_activation_inputs(&self) -> usize {
        2
    }

    /// One row per (token, slot): `top_k` times the input's rows, which a graph that
    /// leaves the token count unknown cannot say until the rows are known.
    fn compute_concrete_output_shape(&self, input_shapes: &[&[usize]], _resolved: &[usize]) -> Vec<usize> {
        vec![input_shapes[0][0] * self.top_k, self.n]
    }

    fn param_shapes(&self, _input_shapes: &[&[usize]], _output_shape: &[usize]) -> Vec<Vec<usize>> {
        // Bytes four to a word, as the row-major gather's, interleaved by rows at
        // load time.
        vec![
            vec![self.experts, self.n, self.k / 4],
            vec![self.experts, self.n, self.k / SBLOCK / 4],
        ]
    }

    fn param_names(&self) -> &'static [&'static str] {
        &["weight_rows", "scale_rows"]
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
        assert_eq!(inputs[0].1[1], self.k, "mxfp8 gemv: input width");
        let ids_stride = inputs[1].1[1];
        assert!(
            ids_stride >= self.top_k,
            "mxfp8 gemv: router row of {ids_stride} cannot hold {} ids",
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
    }

    fn grid(&self, output_shape: &[usize]) -> [u32; 3] {
        [(self.n / ROWS) as u32, output_shape[0] as u32, 1]
    }
}
