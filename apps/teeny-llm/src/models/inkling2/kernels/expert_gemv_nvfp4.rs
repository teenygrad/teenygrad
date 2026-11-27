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

//! The NVFP4 expert projection in integer arithmetic.
//!
//! ```text
//! y[s, n] = sum(k) x[token(s), k] * W[expert(s), n, k]
//! ```
//!
//! as [`super::expert_gather_nvfp4`] computes it, but with neither operand in f32
//! inside the reduction:
//!
//! - **Weights are exact int8.** An E2M1 value times two is an integer --
//!   0, ±1, ±2, ±3, ±4, ±6, ±8, ±12 -- so a nibble becomes its weight through a
//!   16-entry integer table, one `vrgather.vv` on RVV, and the half goes into the
//!   block scale at load time. Nothing about the weights is approximated.
//! - **Activations are int8 per 16-element block**, from [`QuantRowsQ8Op`]:
//!   `q = round(x * 127 / amax)` with the block's `amax / 127` kept beside them.
//!   This is the one approximation, at most half a step of 1/127 of the block's
//!   largest value.
//!
//! A 16-element block's products are summed as integers, and the block's two scales
//! are applied once, in f32: the weight block scale (a vector over rows) times the
//! activation block scale (a scalar). With `|w| <= 12` and `|q| <= 127` a block sums
//! to at most 24384, so the integer sum is exact.
//!
//! **Lanes are output rows, not reduction elements.** A program computes [`ROWS`]
//! rows of one expert, and the reduction runs down `k` with one activation per step
//! broadcast across the rows -- a vector-times-scalar multiply-accumulate, with no
//! horizontal reduction anywhere. That needs the weights of [`ROWS`] rows at the
//! same `k` to be adjacent, which [`interleave_row_bytes`] arranges at load time.

use teeny_core::dtype::{AddOffsets, Comparison, Float, Tensor};
use teeny_macros::kernel;
use teeny_triton::triton::*;

/// Output rows per program, and the interleave width of the repacked weights.
///
/// The kernel works a byte at a time, in 8-bit lanes for the weights and 16-bit
/// lanes for the sums: 32 rows are one vector register of bytes and two of sums at
/// VLEN=256.
pub const ROWS: usize = 32;

/// Elements per activation block, and per weight scale.
pub const QBLOCK: usize = 16;

/// Quantises each row of `x` to int8 per 16-element block.
///
/// Writes `[q | scales]` per row: the K quantised values, as exact integers in f32
/// elements, then the K/16 block scales `amax / 127`. A block of zeros has scale 0
/// and quantises to zeros.
#[allow(clippy::multiple_bound_locations)]
#[kernel]
pub fn quant_rows_q8_forward<T: Triton, D: Float, const BLOCK: i32>(
    x_ptr: InPtr<T::Pointer<D>>,
    y_ptr: OutPtr<T::Pointer<D>>,
    K: i32,
    OUT_STRIDE: i32,
) where
    T: Triton<I32Tensor = <T as Triton>::Tensor<i32>>,
    T::I32Tensor: Tensor<i32, 1>,
    T::I32Tensor: core::ops::Add<i32, Output = T::I32Tensor>,
    T::I32Tensor: Comparison<i32, BoolTensor = T::BoolTensor>,
    T::Pointer<D>: AddOffsets<i32, 1, T::I32Tensor, Output = T::Tensor<T::Pointer<D>>>,
{
    let b = T::program_id(Axis::X);
    let token = T::program_id(Axis::Y);
    let offs = T::arange(0, BLOCK);
    let zeros = T::zeros::<D>(&[BLOCK]);
    let x = T::load(
        x_ptr.add_offsets(offs + (token * K + b * BLOCK)),
        None,
        Some(zeros),
        &[],
        None,
        None,
        None,
        false,
    );
    let amax = T::broadcast_to(T::max(T::abs(x), Some(0), true), &[BLOCK]);
    let one = T::full(&[BLOCK], D::from_f64(1.0));
    let half = T::full(&[BLOCK], D::from_f64(0.5));
    let c127 = T::full(&[BLOCK], D::from_f64(127.0));
    let nonzero = T::gt(amax, zeros);
    let safe = T::where_(nonzero, amax, one);
    let v = x * c127 / safe;
    // Half away from zero: the cast truncates towards zero.
    let r = T::where_(T::lt(v, zeros), v - half, v + half);
    let q = T::cast::<i32, D>(T::cast::<D, i32>(r, None, false), None, false);
    T::store(
        y_ptr.add_offsets(offs + (token * OUT_STRIDE + b * BLOCK)),
        q,
        None,
        &[],
        None,
        None,
    );
    T::store(
        y_ptr.add_offsets(T::zeros::<i32>(&[BLOCK]) + (token * OUT_STRIDE + K + b)),
        T::where_(nonzero, amax / c127, zeros),
        Some(offs.lt(1)),
        &[],
        None,
        None,
    );
}

/// The projection above, over [`ROWS`] rows of one routed expert per program.
#[allow(clippy::multiple_bound_locations)]
#[kernel]
pub fn expert_gemv_nvfp4_forward<T: Triton, D: Float, const ROWS: i32>(
    xq_ptr: InPtr<T::Pointer<f32>>,
    w_ptr: InPtr<T::Pointer<u8>>,
    s_ptr: InPtr<T::Pointer<u8>>,
    es_ptr: InPtr<T::Pointer<D>>,
    ids_ptr: InPtr<T::Pointer<f32>>,
    y_ptr: OutPtr<T::Pointer<D>>,
    N: i32,
    K: i32,
    TOP_K: i32,
    IDS_STRIDE: i32,
    IDS_OFFSET: i32,
    X_STRIDE: i32,
    SLOTS: i32,
    ES_OFFSET: i32,
) where
    T: Triton<I32Tensor = <T as Triton>::Tensor<i32>>,
    T::I32Tensor: Tensor<i32, 1>,
    T::I32Tensor: core::ops::Add<i32, Output = T::I32Tensor>,
    T::I32Tensor: core::ops::Add<T::I32Tensor, Output = T::I32Tensor>,
    T::I32Tensor: core::ops::Sub<T::I32Tensor, Output = T::I32Tensor>,
    T::I32Tensor: core::ops::Mul<i32, Output = T::I32Tensor>,
    T::I32Tensor: core::ops::BitAnd<i32, Output = T::I32Tensor>,
    T::I32Tensor: core::ops::Shr<i32, Output = T::I32Tensor>,
    T::I32Tensor: core::ops::Shl<i32, Output = T::I32Tensor>,
    T::I32Tensor: Comparison<i32, BoolTensor = T::BoolTensor>,
    T::Pointer<D>: AddOffsets<i32, 1, T::I32Tensor, Output = T::Tensor<T::Pointer<D>>>,
    T::Pointer<f32>: AddOffsets<i32, 1, T::I32Tensor, Output = T::Tensor<T::Pointer<f32>>>,
    T::Pointer<u8>: AddOffsets<i32, 1, T::I32Tensor, Output = T::Tensor<T::Pointer<u8>>>,
    T::Tensor<u8>: core::ops::BitAnd<u8, Output = T::Tensor<u8>>,
    T::Tensor<u8>: core::ops::Shr<u8, Output = T::Tensor<u8>>,
    T::Tensor<i16>: core::ops::Add<T::Tensor<i16>, Output = T::Tensor<i16>>,
    T::Tensor<i16>: core::ops::Mul<i16, Output = T::Tensor<i16>>,
{
    let rb = T::program_id(Axis::X);
    let slot = T::program_id(Axis::Y);
    let token = slot / TOP_K;
    let which = slot - token * TOP_K;
    let expert = T::load_scalar_f32_as_i32(ids_ptr, token * IDS_STRIDE + IDS_OFFSET + which);

    // The first slot routed to an expert computes every slot routed to it, one after
    // another, so a batch reads that expert's weights from memory once and from cache
    // for the rest; a later slot with the same expert has nothing left to do. With
    // one token the router's choices are distinct and every slot is first.
    let mut first = 1;
    for earlier in 0..slot {
        let t = earlier / TOP_K;
        let id = T::load_scalar_f32_as_i32(ids_ptr, t * IDS_STRIDE + IDS_OFFSET + (earlier - t * TOP_K));
        if id == expert {
            first = 0;
        }
    }

    if first == 1 {
        let lanes = T::arange(0, ROWS);
        // The expert's outer scale, with E2M1's half folded in, after the block
        // scale bytes in the same buffer.
        let es = T::load(
            es_ptr.add_offsets(T::zeros::<i32>(&[ROWS]) + (ES_OFFSET + expert)),
            None,
            None,
            &[],
            None,
            None,
            None,
            false,
        );
        // E4M3 block scales: a normal byte's f32 bits are `((b & 0x7F) + 960) << 20`
        // (exponent rebiased from 7 to 127, mantissa moved up); exponent 0 is
        // `m * 2^-9`. Scales are never negative.
        let sub9 = T::full(&[ROWS], D::from_f64(0.001_953_125));

        // E2M1 times two, by code: sign (c >> 3), exponent e and mantissa m in the low
        // three bits. e == 0 is m; otherwise (2 + m) << (e - 1).
        let code = T::arange(0, 16);
        let rest = code & 7;
        let e = rest >> 1;
        let m = rest & 1;
        let normal = m + 2;
        let mag = T::where_(
            e.eq(0),
            m,
            T::where_(e.eq(1), normal, T::where_(e.eq(2), normal * 2, normal * 4)),
        );
        let lut = T::cast::<i32, i8>(
            T::where_((code >> 3).eq(1), T::zeros::<i32>(&[16]) - mag, mag),
            None,
            false,
        );

        // The rows' bytes for byte j are adjacent: [expert][row block][byte][row]. A
        // byte holds a row's elements 2j (low nibble) and 2j + 1 (high nibble).
        let bytes_per_row = K / 2;
        let blocks = K / 16;
        let row_block = expert * (N / ROWS) + rb;
        let wbase = row_block * bytes_per_row * ROWS;
        let sbase = row_block * blocks * ROWS;
        for s in slot..SLOTS {
            let st = s / TOP_K;
            let sid = T::load_scalar_f32_as_i32(ids_ptr, st * IDS_STRIDE + IDS_OFFSET + (s - st * TOP_K));
            if sid == expert {
                let xbase = st * X_STRIDE;

                let mut acc = T::zeros::<D>(&[ROWS]);
                for b in 0..blocks {
                    // A block sums to at most 16 * 12 * 127 = 24384, so 16 bits hold it exactly.
                    let mut iacc = T::zeros::<i16>(&[ROWS]);
                    for t in 0..8 {
                        let j = b * 8 + t;
                        let byte = T::load(
                            w_ptr.add_offsets(lanes + (wbase + j * ROWS)),
                            None,
                            None,
                            &[],
                            None,
                            None,
                            None,
                            false,
                        );
                        let lo = T::cast::<u8, i32>(byte & 15u8, None, false);
                        let hi = T::cast::<u8, i32>(byte >> 4u8, None, false);
                        let wl = T::cast::<i8, i16>(T::gather(lut, lo, 0), None, false);
                        let wh = T::cast::<i8, i16>(T::gather(lut, hi, 0), None, false);
                        // Through i8 so the compiler can see both factors are 8-bit and use one
                        // widening multiply-accumulate (vwmacc) instead of extending each weight.
                        let x0 = T::load_scalar_f32_as_i32(xq_ptr, xbase + j * 2) as i8 as i16;
                        let x1 = T::load_scalar_f32_as_i32(xq_ptr, xbase + j * 2 + 1) as i8 as i16;
                        iacc = iacc + wl * x0 + wh * x1;
                    }
                    let sb = T::cast::<u8, i32>(
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
                    ) & 0x7F;
                    let ws = T::where_(
                        sb.lt(8),
                        T::cast::<i32, D>(sb, None, false) * sub9,
                        T::cast::<i32, D>((sb + 960) << 20, None, true),
                    ) * es;
                    let xs = T::load(
                        xq_ptr.add_offsets(T::zeros::<i32>(&[ROWS]) + (xbase + K + b)),
                        None,
                        None,
                        &[],
                        None,
                        None,
                        None,
                        false,
                    );
                    acc = acc + T::cast::<i16, D>(iacc, None, false) * ws * T::cast::<f32, D>(xs, None, false);
                }
                T::store(
                    y_ptr.add_offsets(lanes + (s * N + rb * ROWS)),
                    acc,
                    None,
                    &[],
                    None,
                    None,
                );
            }
        }
    }
}

/// Words in [`byte_scale_rows`]'s buffer: the block scale bytes four to a word, then
/// one f32 per expert.
pub fn byte_scale_words(experts: usize, n: usize, k: usize) -> usize {
    experts * n * (k / QBLOCK) / 4 + experts
}

/// The scale buffer [`expert_gemv_nvfp4_forward`] reads: the E4M3 block scale bytes
/// (`[experts, n, k / 16]`, one value per byte) interleaved by blocks of [`ROWS`]
/// rows as the weights are, four to a word, then each expert's outer scale times
/// E2M1's half (`scale2[e] * 0.5`).
///
/// A byte a block instead of the folded f32's four: a third of every NVFP4 expert's
/// bytes, read once a token in decode and once at every load.
pub fn byte_scale_rows(bytes: &[f32], scale2: &[f32], experts: usize, n: usize, k: usize) -> Vec<f32> {
    let blocks = k / QBLOCK;
    assert_eq!(bytes.len(), experts * n * blocks, "byte scales: size");
    assert_eq!(scale2.len(), experts, "byte scales: one outer scale per expert");
    let mut out = interleave_row_bytes(&super::super::weights::pack_rows(bytes, blocks), experts, n, blocks, ROWS);
    out.truncate(experts * n * blocks / 4);
    out.extend(scale2.iter().map(|s| s * 0.5));
    out
}

/// Interleaves the packed NVFP4 bytes of `[experts, n]` rows, `row_bytes` each and
/// stored four to an f32 word, by blocks of `rows` rows, so the `rows` bytes at each
/// byte column are adjacent: `[experts, n / rows, row_bytes, rows]` bytes, again four
/// to a word.
///
/// This is the layout [`expert_gemv_nvfp4_forward`] reads with one contiguous byte
/// load per column.
pub fn interleave_row_bytes(
    words: &[f32],
    experts: usize,
    n: usize,
    row_bytes: usize,
    rows: usize,
) -> Vec<f32> {
    let row_words = row_bytes.div_ceil(4);
    assert_eq!(words.len(), experts * n * row_words, "interleave bytes: size");
    assert_eq!(n % rows, 0, "interleave bytes: {n} rows do not split into blocks of {rows}");
    // Every row's bytes once, so the transposition below is plain indexing. This
    // runs over the whole checkpoint at load, so it is written for speed.
    let src: Vec<u8> = words.iter().flat_map(|w| w.to_bits().to_le_bytes()).collect();
    let row_stride = row_words * 4;
    let mut out = vec![0u8; (experts * n * row_bytes).div_ceil(4) * 4];
    for (block, dst) in out[..experts * n * row_bytes]
        .chunks_exact_mut(row_bytes * rows)
        .enumerate()
    {
        let first_row = block * rows;
        for r in 0..rows {
            let row = &src[(first_row + r) * row_stride..][..row_bytes];
            for (b, v) in row.iter().enumerate() {
                dst[b * rows + r] = *v;
            }
        }
    }
    out.chunks_exact(4)
        .map(|c| f32::from_bits(u32::from_le_bytes([c[0], c[1], c[2], c[3]])))
        .collect()
}

/// Interleaves `rows`-row blocks of `[experts, n, width]` so the `rows` values at
/// each column are adjacent: `[experts, n / rows, width, rows]`.
///
/// Applied to the folded scales, this is the layout [`expert_gemv_nvfp4_forward`]
/// reads with one contiguous load per block; the weights take
/// [`interleave_row_bytes`].
pub fn interleave_rows(src: &[f32], experts: usize, n: usize, width: usize, rows: usize) -> Vec<f32> {
    assert_eq!(src.len(), experts * n * width, "interleave: size");
    assert_eq!(n % rows, 0, "interleave: {n} rows do not split into blocks of {rows}");
    let mut out = vec![0.0f32; src.len()];
    for e in 0..experts {
        for rb in 0..n / rows {
            for j in 0..width {
                for r in 0..rows {
                    let from = (e * n + rb * rows + r) * width + j;
                    let to = ((e * (n / rows) + rb) * width + j) * rows + r;
                    out[to] = src[from];
                }
            }
        }
    }
    out
}

// ---------------------------------------------------------------------------
// Graph-level ops
// ---------------------------------------------------------------------------

use core::marker::PhantomData;
use std::any::Any;
use std::sync::Arc;
use teeny_core::device::program::ArgVisitor;
use teeny_core::dtype::FloatBytes;
use teeny_core::graph::{CustomOp, Shape};
use teeny_core::model::{RawPtr, RuntimeOp};

/// Quantises `[T, k]` to the `[T, k + k / 16]` that [`ExpertGemvNvfp4Op`] reads.
pub struct QuantRowsQ8Op<D: FloatBytes + Send + Sync + 'static> {
    /// Row width.
    pub k: usize,
    _phantom: PhantomData<D>,
}

impl<D: FloatBytes + Send + Sync + 'static> QuantRowsQ8Op<D> {
    /// Quantises rows of `k`, which must be a multiple of 16.
    pub fn new(k: usize) -> Self {
        assert_eq!(k % QBLOCK, 0, "q8 rows: {k} is not a multiple of {QBLOCK}");
        Self { k, _phantom: PhantomData }
    }
}

impl<D: FloatBytes + Send + Sync + 'static> CustomOp for QuantRowsQ8Op<D> {
    fn name(&self) -> &str {
        "llm.quant_rows_q8"
    }

    fn infer_output_shape(&self, input_shapes: &[&Shape]) -> Shape {
        vec![input_shapes[0][0], Some(self.k + self.k / QBLOCK)]
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn lower(&self) -> Option<(String, String, String, Arc<dyn RuntimeOp>)> {
        let kernel = QuantRowsQ8Forward::<D>::new(QBLOCK as i32);
        Some((
            "quant_rows_q8_forward".to_string(),
            kernel.source,
            "entry_point".to_string(),
            Arc::new(QuantRowsQ8RuntimeOp::<D> { k: self.k, _phantom: PhantomData }),
        ))
    }
}

/// Runtime dispatch for [`QuantRowsQ8Op`].
pub struct QuantRowsQ8RuntimeOp<D: FloatBytes + Send + Sync + 'static> {
    k: usize,
    _phantom: PhantomData<D>,
}

impl<D: FloatBytes + Send + Sync + 'static> RuntimeOp for QuantRowsQ8RuntimeOp<D> {
    fn n_activation_inputs(&self) -> usize {
        1
    }

    fn param_shapes(&self, _input_shapes: &[&[usize]], _output_shape: &[usize]) -> Vec<Vec<usize>> {
        Vec::new()
    }

    fn pack_args(
        &self,
        inputs: &[(RawPtr, &[usize])],
        _params: &[RawPtr],
        output: RawPtr,
        _output_shape: &[usize],
        _output_row_stride: i32,
        visitor: &mut dyn ArgVisitor,
    ) {
        assert_eq!(inputs[0].1[1], self.k, "q8 rows: input width");
        visitor.visit_ptr(inputs[0].0);
        visitor.visit_ptr(output);
        visitor.visit_i32(self.k as i32);
        visitor.visit_i32((self.k + self.k / QBLOCK) as i32);
    }

    fn grid(&self, output_shape: &[usize]) -> [u32; 3] {
        [(self.k / QBLOCK) as u32, output_shape[0] as u32, 1]
    }
}

/// The integer NVFP4 expert projection. Inputs are the `[T, k + k / 16]` from
/// [`QuantRowsQ8Op`] and the router's packed `[T, >= 2 * top_k]` row; the output is
/// `[T * top_k, n]`, as [`super::expert_gather_nvfp4::ExpertGatherNvfp4Op`]'s is.
pub struct ExpertGemvNvfp4Op<D: FloatBytes + Send + Sync + 'static> {
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

impl<D: FloatBytes + Send + Sync + 'static> ExpertGemvNvfp4Op<D> {
    /// A projection over `experts` planes of `[n, k]`, with `top_k` slots per token.
    pub fn new(experts: usize, n: usize, k: usize, top_k: usize) -> Self {
        assert!(top_k > 0 && top_k <= experts, "expert gemv: top_k out of range");
        assert_eq!(n % ROWS, 0, "expert gemv: {n} rows are not a multiple of {ROWS}");
        assert_eq!(k % QBLOCK, 0, "expert gemv: {k} is not a multiple of {QBLOCK}");
        Self { experts, n, k, top_k, _phantom: PhantomData }
    }
}

impl<D: FloatBytes + Send + Sync + 'static> CustomOp for ExpertGemvNvfp4Op<D> {
    fn name(&self) -> &str {
        "llm.expert_gemv_nvfp4"
    }

    fn infer_output_shape(&self, input_shapes: &[&Shape]) -> Shape {
        vec![input_shapes[0][0].map(|t| t * self.top_k), Some(self.n)]
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn lower(&self) -> Option<(String, String, String, Arc<dyn RuntimeOp>)> {
        let kernel = ExpertGemvNvfp4Forward::<D>::new(ROWS as i32);
        Some((
            "expert_gemv_nvfp4_forward".to_string(),
            kernel.source,
            "entry_point".to_string(),
            Arc::new(ExpertGemvNvfp4RuntimeOp::<D> {
                experts: self.experts,
                n: self.n,
                k: self.k,
                top_k: self.top_k,
                _phantom: PhantomData,
            }),
        ))
    }
}

/// Runtime dispatch for [`ExpertGemvNvfp4Op`].
pub struct ExpertGemvNvfp4RuntimeOp<D: FloatBytes + Send + Sync + 'static> {
    experts: usize,
    n: usize,
    k: usize,
    top_k: usize,
    _phantom: PhantomData<D>,
}

impl<D: FloatBytes + Send + Sync + 'static> RuntimeOp for ExpertGemvNvfp4RuntimeOp<D> {
    fn n_activation_inputs(&self) -> usize {
        2
    }

    /// One row per (token, slot): `top_k` times the input's rows, which a graph that
    /// leaves the token count unknown cannot say until the rows are known.
    fn compute_concrete_output_shape(&self, input_shapes: &[&[usize]], _resolved: &[usize]) -> Vec<usize> {
        vec![input_shapes[0][0] * self.top_k, self.n]
    }

    fn param_shapes(&self, _input_shapes: &[&[usize]], _output_shape: &[usize]) -> Vec<Vec<usize>> {
        // The same element counts as the row-major gather's, interleaved by
        // `interleave_rows` at load time; the scales also carry E2M1's factor of 1/2.
        vec![vec![self.experts, self.n, self.k / 8], vec![byte_scale_words(self.experts, self.n, self.k)]]
    }

    fn param_names(&self) -> &'static [&'static str] {
        &["weight_rows", "byte_scale_rows"]
    }

    fn pack_args(
        &self,
        inputs: &[(RawPtr, &[usize])],
        params: &[RawPtr],
        output: RawPtr,
        output_shape: &[usize],
        _output_row_stride: i32,
        visitor: &mut dyn ArgVisitor,
    ) {
        let x_stride = self.k + self.k / QBLOCK;
        assert_eq!(inputs[0].1[1], x_stride, "expert gemv: quantised input width");
        let ids_stride = inputs[1].1[1];
        assert!(
            ids_stride >= self.top_k,
            "expert gemv: router row of {ids_stride} cannot hold {} ids",
            self.top_k
        );
        visitor.visit_ptr(inputs[0].0);
        visitor.visit_ptr(params[0]);
        // The block scale bytes, then the per-expert scales after them.
        visitor.visit_ptr(params[1]);
        visitor.visit_ptr(params[1]);
        visitor.visit_ptr(inputs[1].0);
        visitor.visit_ptr(output);
        visitor.visit_i32(self.n as i32);
        visitor.visit_i32(self.k as i32);
        visitor.visit_i32(self.top_k as i32);
        visitor.visit_i32(ids_stride as i32);
        // The ids sit after the weights in the router's packed row.
        visitor.visit_i32((ids_stride - self.top_k) as i32);
        visitor.visit_i32(x_stride as i32);
        visitor.visit_i32(output_shape[0] as i32);
        visitor.visit_i32((self.experts * self.n * (self.k / QBLOCK) / 4) as i32);
    }

    fn grid(&self, output_shape: &[usize]) -> [u32; 3] {
        [(self.n / ROWS) as u32, output_shape[0] as u32, 1]
    }
}

#[cfg(test)]
mod tests {
    use super::{interleave_row_bytes, interleave_rows};

    #[test]
    fn interleave_bytes_puts_each_byte_column_of_a_row_block_together() {
        // One expert, four rows of eight bytes (two words), in blocks of two rows.
        let row = |r: u8| -> Vec<f32> {
            (0..2u8)
                .map(|w| {
                    let b: [u8; 4] = core::array::from_fn(|i| r * 16 + w * 4 + i as u8);
                    f32::from_bits(u32::from_le_bytes(b))
                })
                .collect()
        };
        let words: Vec<f32> = (0..4).flat_map(row).collect();
        let out = interleave_row_bytes(&words, 1, 4, 8, 2);
        let bytes: Vec<u8> = out.iter().flat_map(|w| w.to_bits().to_le_bytes()).collect();
        // Rows 0 and 1, byte by byte: (0, 16), (1, 17), ...
        assert_eq!(&bytes[..6], &[0, 16, 1, 17, 2, 18]);
        // Rows 2 and 3 start at byte 16: (32, 48), (33, 49), ...
        assert_eq!(&bytes[16..20], &[32, 48, 33, 49]);
    }

    #[test]
    fn interleave_puts_each_column_of_a_row_block_together() {
        // Two experts, four rows of three columns, in blocks of two rows.
        let src: Vec<f32> = (0..24).map(|v| v as f32).collect();
        let out = interleave_rows(&src, 2, 4, 3, 2);
        // Expert 0, rows 0-1: column 0 is (0, 3), column 1 is (1, 4), ...
        assert_eq!(&out[..6], &[0.0, 3.0, 1.0, 4.0, 2.0, 5.0]);
        // Expert 1, rows 2-3 start at row 6 of the source: (18, 21), (19, 22), ...
        assert_eq!(&out[18..], &[18.0, 21.0, 19.0, 22.0, 20.0, 23.0]);
    }
}
