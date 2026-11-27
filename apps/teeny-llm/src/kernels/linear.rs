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

//! Bias-free `nn.Linear`: `y[m, n] = sum_k x[m, k] * w[n, k]`, with PyTorch's weight
//! layout `[N, K]`.
//!
//! A portable kernel for both backends (teeny's `LinearForward` uses CUDA tensor
//! descriptors). Each program computes four consecutive outputs of one row, so every
//! load of `x` feeds four accumulators. Lanes run along `K` (contiguous in both `x` and
//! `w`), and each accumulator is reduced across lanes at the end.
//!
//! Grid: `M * N / 4` (`N` must be a multiple of 4).

use teeny_core::dtype::{AddOffsets, Comparison, Float, Tensor};
use teeny_macros::kernel;
use teeny_triton::triton::*;

/// The linear layer above.
#[allow(clippy::multiple_bound_locations)]
#[kernel]
pub fn linear4_forward<T: Triton, D: Float, const BLOCK: i32>(
    x_ptr: InPtr<T::Pointer<D>>,
    w_ptr: InPtr<T::Pointer<D>>,
    y_ptr: OutPtr<T::Pointer<D>>,
    N: i32,
    K: i32,
) where
    T: Triton<I32Tensor = <T as Triton>::Tensor<i32>>,
    T::I32Tensor: Tensor<i32, 1>,
    T::I32Tensor: core::ops::Add<i32, Output = T::I32Tensor>,
    T::I32Tensor: core::ops::Mul<i32, Output = T::I32Tensor>,
    T::I32Tensor: Comparison<i32, BoolTensor = T::BoolTensor>,
    T::Pointer<D>: AddOffsets<i32, 1, T::I32Tensor, Output = T::Tensor<T::Pointer<D>>>,
{
    let pid = T::program_id(Axis::X);
    let groups = N / 4;
    let m = pid / groups;
    let n = (pid % groups) * 4;

    let offs = T::arange(0, BLOCK);
    let zeros = T::zeros::<D>(&[BLOCK]);
    let mut a0 = T::zeros::<D>(&[BLOCK]);
    let mut a1 = T::zeros::<D>(&[BLOCK]);
    let mut a2 = T::zeros::<D>(&[BLOCK]);
    let mut a3 = T::zeros::<D>(&[BLOCK]);
    let n_chunks = (K + BLOCK - 1) / BLOCK;
    for i in 0..n_chunks {
        let k = offs + i * BLOCK;
        let km = k.lt(K);
        let x = T::load(
            x_ptr.add_offsets(k + m * K),
            Some(km),
            Some(zeros),
            &[],
            None,
            None,
            None,
            false,
        );
        let w0 = T::load(
            w_ptr.add_offsets(k + n * K),
            Some(km),
            Some(zeros),
            &[],
            None,
            None,
            None,
            false,
        );
        a0 = a0 + x * w0;
        let w1 = T::load(
            w_ptr.add_offsets(k + (n + 1) * K),
            Some(km),
            Some(zeros),
            &[],
            None,
            None,
            None,
            false,
        );
        a1 = a1 + x * w1;
        let w2 = T::load(
            w_ptr.add_offsets(k + (n + 2) * K),
            Some(km),
            Some(zeros),
            &[],
            None,
            None,
            None,
            false,
        );
        a2 = a2 + x * w2;
        let w3 = T::load(
            w_ptr.add_offsets(k + (n + 3) * K),
            Some(km),
            Some(zeros),
            &[],
            None,
            None,
            None,
            false,
        );
        a3 = a3 + x * w3;
    }
    // Each sum is a one-element tensor; store it through a one-lane mask.
    let one = offs.lt(1);
    let base = m * N + n;
    T::store(
        y_ptr.add_offsets(offs + base),
        T::sum(a0, Some(0), true) + zeros,
        Some(one),
        &[],
        None,
        None,
    );
    T::store(
        y_ptr.add_offsets(offs + (base + 1)),
        T::sum(a1, Some(0), true) + zeros,
        Some(one),
        &[],
        None,
        None,
    );
    T::store(
        y_ptr.add_offsets(offs + (base + 2)),
        T::sum(a2, Some(0), true) + zeros,
        Some(one),
        &[],
        None,
        None,
    );
    T::store(
        y_ptr.add_offsets(offs + (base + 3)),
        T::sum(a3, Some(0), true) + zeros,
        Some(one),
        &[],
        None,
        None,
    );
}

/// [`linear4_forward`] reading bf16 weights, two to a 32-bit word.
///
/// Word `j` of a row holds element `j` in its low half and element `j + K / 2` in
/// its high half, so both halves pair with a contiguous run of `x`. A bf16 is the
/// top half of an f32, so each half becomes its f32 exactly with a shift or a mask
/// and a bitcast: half the bytes of [`linear4_forward`] for the same arithmetic.
///
/// Program ids run down the rows, as [`linear4_rows_forward`]'s do: consecutive ids
/// compute the same four outputs for consecutive rows, and the launcher hands each
/// thread a run of consecutive ids, so a batch of rows reads those outputs' weights
/// from memory once and from cache for the rest. With one row it is the same order.
///
/// Grid: `M * N / 4`.
#[allow(clippy::multiple_bound_locations)]
#[kernel]
pub fn linear4_bf16_forward<T: Triton, D: Float, const BLOCK: i32>(
    x_ptr: InPtr<T::Pointer<D>>,
    w_ptr: InPtr<T::Pointer<D>>,
    y_ptr: OutPtr<T::Pointer<D>>,
    M: i32,
    N: i32,
    K: i32,
) where
    T: Triton<I32Tensor = <T as Triton>::Tensor<i32>>,
    T::I32Tensor: Tensor<i32, 1>,
    T::I32Tensor: core::ops::Add<i32, Output = T::I32Tensor>,
    T::I32Tensor: core::ops::Mul<i32, Output = T::I32Tensor>,
    T::I32Tensor: core::ops::BitAnd<i32, Output = T::I32Tensor>,
    T::I32Tensor: core::ops::Shl<i32, Output = T::I32Tensor>,
    T::I32Tensor: Comparison<i32, BoolTensor = T::BoolTensor>,
    T::Pointer<D>: AddOffsets<i32, 1, T::I32Tensor, Output = T::Tensor<T::Pointer<D>>>,
{
    let pid = T::program_id(Axis::X);
    let m = pid % M;
    let n = (pid / M) * 4;
    let half_k = K / 2;

    let offs = T::arange(0, BLOCK);
    let zeros = T::zeros::<D>(&[BLOCK]);
    let mut a0 = T::zeros::<D>(&[BLOCK]);
    let mut a1 = T::zeros::<D>(&[BLOCK]);
    let mut a2 = T::zeros::<D>(&[BLOCK]);
    let mut a3 = T::zeros::<D>(&[BLOCK]);
    let n_chunks = (half_k + BLOCK - 1) / BLOCK;
    for i in 0..n_chunks {
        let j = offs + i * BLOCK;
        let jm = j.lt(half_k);
        let xlo = T::load(
            x_ptr.add_offsets(j + m * K),
            Some(jm),
            Some(zeros),
            &[],
            None,
            None,
            None,
            false,
        );
        let xhi = T::load(
            x_ptr.add_offsets(j + (m * K + half_k)),
            Some(jm),
            Some(zeros),
            &[],
            None,
            None,
            None,
            false,
        );
        let w0 = T::cast::<D, i32>(
            T::load(
                w_ptr.add_offsets(j + n * half_k),
                Some(jm),
                Some(zeros),
                &[],
                None,
                None,
                None,
                false,
            ),
            None,
            true,
        );
        a0 = a0
            + xlo * T::cast::<i32, D>(w0 << 16, None, true)
            + xhi * T::cast::<i32, D>(w0 & -65536, None, true);
        let w1 = T::cast::<D, i32>(
            T::load(
                w_ptr.add_offsets(j + (n + 1) * half_k),
                Some(jm),
                Some(zeros),
                &[],
                None,
                None,
                None,
                false,
            ),
            None,
            true,
        );
        a1 = a1
            + xlo * T::cast::<i32, D>(w1 << 16, None, true)
            + xhi * T::cast::<i32, D>(w1 & -65536, None, true);
        let w2 = T::cast::<D, i32>(
            T::load(
                w_ptr.add_offsets(j + (n + 2) * half_k),
                Some(jm),
                Some(zeros),
                &[],
                None,
                None,
                None,
                false,
            ),
            None,
            true,
        );
        a2 = a2
            + xlo * T::cast::<i32, D>(w2 << 16, None, true)
            + xhi * T::cast::<i32, D>(w2 & -65536, None, true);
        let w3 = T::cast::<D, i32>(
            T::load(
                w_ptr.add_offsets(j + (n + 3) * half_k),
                Some(jm),
                Some(zeros),
                &[],
                None,
                None,
                None,
                false,
            ),
            None,
            true,
        );
        a3 = a3
            + xlo * T::cast::<i32, D>(w3 << 16, None, true)
            + xhi * T::cast::<i32, D>(w3 & -65536, None, true);
    }
    let one = offs.lt(1);
    let base = m * N + n;
    T::store(
        y_ptr.add_offsets(offs + base),
        T::sum(a0, Some(0), true) + zeros,
        Some(one),
        &[],
        None,
        None,
    );
    T::store(
        y_ptr.add_offsets(offs + (base + 1)),
        T::sum(a1, Some(0), true) + zeros,
        Some(one),
        &[],
        None,
        None,
    );
    T::store(
        y_ptr.add_offsets(offs + (base + 2)),
        T::sum(a2, Some(0), true) + zeros,
        Some(one),
        &[],
        None,
        None,
    );
    T::store(
        y_ptr.add_offsets(offs + (base + 3)),
        T::sum(a3, Some(0), true) + zeros,
        Some(one),
        &[],
        None,
        None,
    );
}

/// [`linear4_forward`] with program ids numbered down the rows: consecutive ids compute
/// the same four outputs for consecutive rows. The launcher hands each thread a run of
/// consecutive ids, so a thread reuses those outputs' weights (`4 * K` values) from
/// cache across rows instead of reading them from memory once per row.
///
/// Grid: `M * N / 4`.
#[allow(clippy::multiple_bound_locations)]
#[kernel]
pub fn linear4_rows_forward<T: Triton, D: Float, const BLOCK: i32>(
    x_ptr: InPtr<T::Pointer<D>>,
    w_ptr: InPtr<T::Pointer<D>>,
    y_ptr: OutPtr<T::Pointer<D>>,
    M: i32,
    N: i32,
    K: i32,
) where
    T: Triton<I32Tensor = <T as Triton>::Tensor<i32>>,
    T::I32Tensor: Tensor<i32, 1>,
    T::I32Tensor: core::ops::Add<i32, Output = T::I32Tensor>,
    T::I32Tensor: core::ops::Mul<i32, Output = T::I32Tensor>,
    T::I32Tensor: Comparison<i32, BoolTensor = T::BoolTensor>,
    T::Pointer<D>: AddOffsets<i32, 1, T::I32Tensor, Output = T::Tensor<T::Pointer<D>>>,
{
    let pid = T::program_id(Axis::X);
    let m = pid % M;
    let n = (pid / M) * 4;

    let offs = T::arange(0, BLOCK);
    let zeros = T::zeros::<D>(&[BLOCK]);
    let mut a0 = T::zeros::<D>(&[BLOCK]);
    let mut a1 = T::zeros::<D>(&[BLOCK]);
    let mut a2 = T::zeros::<D>(&[BLOCK]);
    let mut a3 = T::zeros::<D>(&[BLOCK]);
    let n_chunks = (K + BLOCK - 1) / BLOCK;
    for i in 0..n_chunks {
        let k = offs + i * BLOCK;
        let km = k.lt(K);
        let x = T::load(
            x_ptr.add_offsets(k + m * K),
            Some(km),
            Some(zeros),
            &[],
            None,
            None,
            None,
            false,
        );
        let w0 = T::load(
            w_ptr.add_offsets(k + n * K),
            Some(km),
            Some(zeros),
            &[],
            None,
            None,
            None,
            false,
        );
        a0 = a0 + x * w0;
        let w1 = T::load(
            w_ptr.add_offsets(k + (n + 1) * K),
            Some(km),
            Some(zeros),
            &[],
            None,
            None,
            None,
            false,
        );
        a1 = a1 + x * w1;
        let w2 = T::load(
            w_ptr.add_offsets(k + (n + 2) * K),
            Some(km),
            Some(zeros),
            &[],
            None,
            None,
            None,
            false,
        );
        a2 = a2 + x * w2;
        let w3 = T::load(
            w_ptr.add_offsets(k + (n + 3) * K),
            Some(km),
            Some(zeros),
            &[],
            None,
            None,
            None,
            false,
        );
        a3 = a3 + x * w3;
    }
    // Each sum is a one-element tensor; store it through a one-lane mask.
    let one = offs.lt(1);
    let base = m * N + n;
    T::store(
        y_ptr.add_offsets(offs + base),
        T::sum(a0, Some(0), true) + zeros,
        Some(one),
        &[],
        None,
        None,
    );
    T::store(
        y_ptr.add_offsets(offs + (base + 1)),
        T::sum(a1, Some(0), true) + zeros,
        Some(one),
        &[],
        None,
        None,
    );
    T::store(
        y_ptr.add_offsets(offs + (base + 2)),
        T::sum(a2, Some(0), true) + zeros,
        Some(one),
        &[],
        None,
        None,
    );
    T::store(
        y_ptr.add_offsets(offs + (base + 3)),
        T::sum(a3, Some(0), true) + zeros,
        Some(one),
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

/// Graph-level bias-free linear layer, `[M, K]` to `[M, N]`. Weight slot `weight`
/// `[N, K]`.
pub struct Linear4Op<D: FloatBytes + Send + Sync + 'static> {
    /// Input features `K`.
    pub k: usize,
    /// Output features `N` (a multiple of 4).
    pub n: usize,
    /// Lanes along `K`.
    pub block: i32,
    /// Number program ids down the rows ([`linear4_rows_forward`]) rather than along
    /// them, so weights are reused from cache across rows.
    pub weight_reuse: bool,
    _phantom: PhantomData<D>,
}

impl<D: FloatBytes + Send + Sync + 'static> Linear4Op<D> {
    /// The op with [`Self::weight_reuse`] set.
    pub fn with_weight_reuse(mut self) -> Self {
        self.weight_reuse = true;
        self
    }

    /// Creates the op.
    pub fn new(k: usize, n: usize, block: i32) -> Self {
        assert!(
            n.is_multiple_of(4),
            "linear4: out features must be a multiple of 4"
        );
        Self {
            k,
            n,
            block,
            weight_reuse: false,
            _phantom: PhantomData,
        }
    }
}

impl<D: FloatBytes + Send + Sync + 'static> CustomOp for Linear4Op<D> {
    fn name(&self) -> &str {
        "llm.linear4"
    }

    fn infer_output_shape(&self, input_shapes: &[&Shape]) -> Shape {
        vec![input_shapes[0][0], Some(self.n)]
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn lower(&self) -> Option<(String, String, String, Arc<dyn RuntimeOp>)> {
        let (name, source) = if self.weight_reuse {
            (
                "linear4_rows_forward",
                Linear4RowsForward::<D>::new(self.block).source,
            )
        } else {
            (
                "linear4_forward",
                Linear4Forward::<D>::new(self.block).source,
            )
        };
        Some((
            name.to_string(),
            source,
            "entry_point".to_string(),
            Arc::new(Linear4RuntimeOp::<D> {
                k: self.k,
                n: self.n,
                weight_reuse: self.weight_reuse,
                _phantom: PhantomData,
            }),
        ))
    }
}

/// Runtime dispatch for [`Linear4Op`].
pub struct Linear4RuntimeOp<D: FloatBytes + Send + Sync + 'static> {
    k: usize,
    n: usize,
    weight_reuse: bool,
    _phantom: PhantomData<D>,
}

impl<D: FloatBytes + Send + Sync + 'static> RuntimeOp for Linear4RuntimeOp<D> {
    fn n_activation_inputs(&self) -> usize {
        1
    }

    fn param_shapes(&self, _input_shapes: &[&[usize]], _output_shape: &[usize]) -> Vec<Vec<usize>> {
        vec![vec![self.n, self.k]]
    }

    fn param_names(&self) -> &'static [&'static str] {
        &["weight"]
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
        assert_eq!(inputs[0].1[1], self.k, "linear4: input features");
        visitor.visit_ptr(inputs[0].0);
        visitor.visit_ptr(params[0]);
        visitor.visit_ptr(output);
        if self.weight_reuse {
            visitor.visit_i32(output_shape[0] as i32);
        }
        visitor.visit_i32(self.n as i32);
        visitor.visit_i32(self.k as i32);
    }

    fn grid(&self, output_shape: &[usize]) -> [u32; 3] {
        [(output_shape[0] * self.n / 4) as u32, 1, 1]
    }
}

/// Packs a row-major `[rows, cols]` matrix of bf16-valued f32s into the
/// `[rows, cols / 2]` words [`linear4_bf16_forward`] reads: word `j` holds element
/// `j` in its low half and element `j + cols / 2` in its high half.
///
/// Fails if a value is not exactly a bf16, so a weight that was really f32 is
/// never silently rounded.
pub fn pack_bf16_halves(values: &[f32], rows: usize, cols: usize) -> Result<Vec<f32>, String> {
    if cols % 2 != 0 || values.len() != rows * cols {
        return Err(format!(
            "bf16 packing: {} values are not {rows} rows of an even {cols}",
            values.len()
        ));
    }
    let half = cols / 2;
    let mut out = Vec::with_capacity(rows * half);
    for (r, row) in values.chunks(cols).enumerate() {
        for j in 0..half {
            let (lo, hi) = (row[j].to_bits(), row[j + half].to_bits());
            for (bits, col) in [(lo, j), (hi, j + half)] {
                if bits & 0xFFFF != 0 {
                    return Err(format!(
                        "bf16 packing: row {r} column {col} is {}, which is not a bf16",
                        f32::from_bits(bits)
                    ));
                }
            }
            out.push(f32::from_bits((lo >> 16) | (hi & 0xFFFF_0000)));
        }
    }
    Ok(out)
}

/// [`Linear4Op`] with its weight stored as bf16, two to a word, for
/// [`linear4_bf16_forward`]. Weight slot `weight_bf16`, `[N, K / 2]` words, filled by
/// [`pack_bf16_halves`].
pub struct Linear4Bf16Op<D: FloatBytes + Send + Sync + 'static> {
    /// Input features `K` (even).
    pub k: usize,
    /// Output features `N` (a multiple of 4).
    pub n: usize,
    /// Lanes along `K / 2`.
    pub block: i32,
    _phantom: PhantomData<D>,
}

impl<D: FloatBytes + Send + Sync + 'static> Linear4Bf16Op<D> {
    /// Creates the op.
    pub fn new(k: usize, n: usize, block: i32) -> Self {
        assert!(
            n.is_multiple_of(4),
            "linear4 bf16: out features must be a multiple of 4"
        );
        assert!(
            k.is_multiple_of(2),
            "linear4 bf16: in features must be even"
        );
        Self {
            k,
            n,
            block,
            _phantom: PhantomData,
        }
    }
}

impl<D: FloatBytes + Send + Sync + 'static> CustomOp for Linear4Bf16Op<D> {
    fn name(&self) -> &str {
        "linear4_bf16"
    }

    fn infer_output_shape(&self, input_shapes: &[&Shape]) -> Shape {
        vec![input_shapes[0][0], Some(self.n)]
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn lower(&self) -> Option<(String, String, String, Arc<dyn RuntimeOp>)> {
        let kernel = Linear4Bf16Forward::<D>::new(self.block);
        Some((
            "linear4_bf16_forward".to_string(),
            kernel.source,
            "entry_point".to_string(),
            Arc::new(Linear4Bf16RuntimeOp::<D> {
                k: self.k,
                n: self.n,
                _phantom: PhantomData,
            }),
        ))
    }
}

/// Runtime dispatch for [`Linear4Bf16Op`].
pub struct Linear4Bf16RuntimeOp<D: FloatBytes + Send + Sync + 'static> {
    k: usize,
    n: usize,
    _phantom: PhantomData<D>,
}

impl<D: FloatBytes + Send + Sync + 'static> RuntimeOp for Linear4Bf16RuntimeOp<D> {
    fn n_activation_inputs(&self) -> usize {
        1
    }

    fn param_shapes(&self, _input_shapes: &[&[usize]], _output_shape: &[usize]) -> Vec<Vec<usize>> {
        vec![vec![self.n, self.k / 2]]
    }

    fn param_names(&self) -> &'static [&'static str] {
        &["weight_bf16"]
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
        assert_eq!(inputs[0].1[1], self.k, "linear4 bf16: input features");
        visitor.visit_ptr(inputs[0].0);
        visitor.visit_ptr(params[0]);
        visitor.visit_ptr(output);
        visitor.visit_i32(output_shape[0] as i32);
        visitor.visit_i32(self.n as i32);
        visitor.visit_i32(self.k as i32);
    }

    fn grid(&self, output_shape: &[usize]) -> [u32; 3] {
        [(output_shape[0] * self.n / 4) as u32, 1, 1]
    }
}

#[cfg(test)]
mod bf16_tests {
    use super::pack_bf16_halves;

    #[test]
    fn packing_puts_the_two_halves_in_one_word_exactly() {
        let a = f32::from_bits(0x3F80_0000); // 1.0
        let b = f32::from_bits(0xC040_0000); // -3.0
        let packed = pack_bf16_halves(&[a, 0.0, b, 0.0], 1, 4).unwrap();
        let w0 = packed[0].to_bits();
        assert_eq!(f32::from_bits(w0 << 16), a);
        assert_eq!(f32::from_bits(w0 & 0xFFFF_0000), b);
    }

    #[test]
    fn packing_refuses_a_value_that_is_not_a_bf16() {
        assert!(pack_bf16_halves(&[1.0, 1.0 + f32::EPSILON], 1, 2).is_err());
    }
}

// ---------------------------------------------------------------------------
// Output features as lanes
// ---------------------------------------------------------------------------

/// Output features per program of [`linear_lanes_bf16_forward`]: one f32 vector at
/// LMUL 2 on a VLEN=256 core, so four row accumulators fit with room to spare.
pub const LANES: usize = 16;

/// Rows a program of [`linear_lanes_bf16_forward`] computes at once.
pub const ROW_TILE: usize = 4;

/// [`linear4_bf16_forward`] with output features as lanes, for batches of rows.
///
/// The reduction runs down `K` with one input value per step broadcast across
/// [`LANES`] outputs, so a weight word is loaded and unpacked once and feeds up to
/// [`ROW_TILE`] rows' accumulators, with no horizontal sum. The lanes-along-`K`
/// kernel unpacks every word again for every row, which made a four-row step cost
/// twice a one-row step for the same weight bytes.
///
/// Weights are [`pack_bf16_halves`] words interleaved by blocks of [`LANES`]
/// outputs ([`interleave_lanes`]): `[N / LANES][K / 2][LANES]`. A tile with one
/// row left takes a one-accumulator loop, so a single token pays nothing for the
/// tiling; a partial tile of two or three rows repeats its last row.
///
/// Grid: `(N / LANES, ceil(M / ROW_TILE))`.
#[allow(clippy::multiple_bound_locations)]
#[kernel]
pub fn linear_lanes_bf16_forward<T: Triton, D: Float, const LANES: i32>(
    x_ptr: InPtr<T::Pointer<D>>,
    w_ptr: InPtr<T::Pointer<D>>,
    y_ptr: OutPtr<T::Pointer<D>>,
    M: i32,
    N: i32,
    K: i32,
) where
    T: Triton<I32Tensor = <T as Triton>::Tensor<i32>>,
    T::I32Tensor: Tensor<i32, 1>,
    T::I32Tensor: core::ops::Add<i32, Output = T::I32Tensor>,
    T::I32Tensor: core::ops::BitAnd<i32, Output = T::I32Tensor>,
    T::I32Tensor: core::ops::Shl<i32, Output = T::I32Tensor>,
    T::Pointer<D>: AddOffsets<i32, 1, T::I32Tensor, Output = T::Tensor<T::Pointer<D>>>,
{
    let nb = T::program_id(Axis::X);
    let rt = T::program_id(Axis::Y);
    let half_k = K / 2;
    let lanes = T::arange(0, LANES);
    let splat = T::zeros::<i32>(&[LANES]);
    let wbase = nb * half_k * LANES;
    let r0 = rt * 4;
    let col = nb * LANES;

    if M - r0 == 1 {
        let x0 = r0 * K;
        let mut a0 = T::zeros::<D>(&[LANES]);
        for j in 0..half_k {
            let w = T::cast::<D, i32>(
                T::load(
                    w_ptr.add_offsets(lanes + (wbase + j * LANES)),
                    None,
                    None,
                    &[],
                    None,
                    None,
                    None,
                    false,
                ),
                None,
                true,
            );
            let lo = T::cast::<i32, D>(w << 16, None, true);
            let hi = T::cast::<i32, D>(w & -65536, None, true);
            let xl = T::load(
                x_ptr.add_offsets(splat + (x0 + j)),
                None,
                None,
                &[],
                None,
                None,
                None,
                false,
            );
            let xh = T::load(
                x_ptr.add_offsets(splat + (x0 + half_k + j)),
                None,
                None,
                &[],
                None,
                None,
                None,
                false,
            );
            a0 = a0 + lo * xl + hi * xh;
        }
        T::store(
            y_ptr.add_offsets(lanes + (r0 * N + col)),
            a0,
            None,
            &[],
            None,
            None,
        );
    } else {
        // Rows past the end repeat the last one: they store the same values to the
        // same place, which costs a little arithmetic and no masks.
        let last = M - 1;
        let r1 = if r0 + 1 < M { r0 + 1 } else { last };
        let r2 = if r0 + 2 < M { r0 + 2 } else { last };
        let r3 = if r0 + 3 < M { r0 + 3 } else { last };
        let x0 = r0 * K;
        let x1 = r1 * K;
        let x2 = r2 * K;
        let x3 = r3 * K;
        let mut a0 = T::zeros::<D>(&[LANES]);
        let mut a1 = T::zeros::<D>(&[LANES]);
        let mut a2 = T::zeros::<D>(&[LANES]);
        let mut a3 = T::zeros::<D>(&[LANES]);
        for j in 0..half_k {
            let w = T::cast::<D, i32>(
                T::load(
                    w_ptr.add_offsets(lanes + (wbase + j * LANES)),
                    None,
                    None,
                    &[],
                    None,
                    None,
                    None,
                    false,
                ),
                None,
                true,
            );
            let lo = T::cast::<i32, D>(w << 16, None, true);
            let hi = T::cast::<i32, D>(w & -65536, None, true);
            let h = half_k + j;
            a0 = a0
                + lo * T::load(
                    x_ptr.add_offsets(splat + (x0 + j)),
                    None,
                    None,
                    &[],
                    None,
                    None,
                    None,
                    false,
                )
                + hi * T::load(
                    x_ptr.add_offsets(splat + (x0 + h)),
                    None,
                    None,
                    &[],
                    None,
                    None,
                    None,
                    false,
                );
            a1 = a1
                + lo * T::load(
                    x_ptr.add_offsets(splat + (x1 + j)),
                    None,
                    None,
                    &[],
                    None,
                    None,
                    None,
                    false,
                )
                + hi * T::load(
                    x_ptr.add_offsets(splat + (x1 + h)),
                    None,
                    None,
                    &[],
                    None,
                    None,
                    None,
                    false,
                );
            a2 = a2
                + lo * T::load(
                    x_ptr.add_offsets(splat + (x2 + j)),
                    None,
                    None,
                    &[],
                    None,
                    None,
                    None,
                    false,
                )
                + hi * T::load(
                    x_ptr.add_offsets(splat + (x2 + h)),
                    None,
                    None,
                    &[],
                    None,
                    None,
                    None,
                    false,
                );
            a3 = a3
                + lo * T::load(
                    x_ptr.add_offsets(splat + (x3 + j)),
                    None,
                    None,
                    &[],
                    None,
                    None,
                    None,
                    false,
                )
                + hi * T::load(
                    x_ptr.add_offsets(splat + (x3 + h)),
                    None,
                    None,
                    &[],
                    None,
                    None,
                    None,
                    false,
                );
        }
        T::store(
            y_ptr.add_offsets(lanes + (r0 * N + col)),
            a0,
            None,
            &[],
            None,
            None,
        );
        T::store(
            y_ptr.add_offsets(lanes + (r1 * N + col)),
            a1,
            None,
            &[],
            None,
            None,
        );
        T::store(
            y_ptr.add_offsets(lanes + (r2 * N + col)),
            a2,
            None,
            &[],
            None,
            None,
        );
        T::store(
            y_ptr.add_offsets(lanes + (r3 * N + col)),
            a3,
            None,
            &[],
            None,
            None,
        );
    }
}

/// Reorders [`pack_bf16_halves`] words `[n, words]` into the `[n / LANES][words][LANES]`
/// blocks [`linear_lanes_bf16_forward`] reads, so a step down `K` is one contiguous
/// load.
pub fn interleave_lanes(packed: &[f32], n: usize, words: usize) -> Result<Vec<f32>, String> {
    if n % LANES != 0 || packed.len() != n * words {
        return Err(format!(
            "lanes interleave: {} words are not {n} rows (a multiple of {LANES}) of {words}",
            packed.len()
        ));
    }
    let mut out = vec![0.0f32; packed.len()];
    for (r, row) in packed.chunks_exact(words).enumerate() {
        let (block, lane) = (r / LANES, r % LANES);
        for (j, &v) in row.iter().enumerate() {
            out[(block * words + j) * LANES + lane] = v;
        }
    }
    Ok(out)
}

/// A bias-free linear on bf16 weights with output features as lanes, for
/// [`linear_lanes_bf16_forward`]. Weight slot `weight_lanes`, `[N, K / 2]` words in
/// the order [`interleave_lanes`] gives.
pub struct LinearLanesBf16Op<D: FloatBytes + Send + Sync + 'static> {
    /// Input features `K` (even).
    pub k: usize,
    /// Output features `N` (a multiple of [`LANES`]).
    pub n: usize,
    _phantom: PhantomData<D>,
}

impl<D: FloatBytes + Send + Sync + 'static> LinearLanesBf16Op<D> {
    /// Creates the op.
    pub fn new(k: usize, n: usize) -> Self {
        assert!(
            n.is_multiple_of(LANES),
            "linear lanes: {n} outputs are not a multiple of {LANES}"
        );
        assert!(
            k.is_multiple_of(2),
            "linear lanes: in features must be even"
        );
        Self {
            k,
            n,
            _phantom: PhantomData,
        }
    }
}

impl<D: FloatBytes + Send + Sync + 'static> CustomOp for LinearLanesBf16Op<D> {
    fn name(&self) -> &str {
        "linear_lanes_bf16"
    }

    fn infer_output_shape(&self, input_shapes: &[&Shape]) -> Shape {
        vec![input_shapes[0][0], Some(self.n)]
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn lower(&self) -> Option<(String, String, String, Arc<dyn RuntimeOp>)> {
        let kernel = LinearLanesBf16Forward::<D>::new(LANES as i32);
        Some((
            "linear_lanes_bf16_forward".to_string(),
            kernel.source,
            "entry_point".to_string(),
            Arc::new(LinearLanesBf16RuntimeOp::<D> {
                k: self.k,
                n: self.n,
                _phantom: PhantomData,
            }),
        ))
    }
}

/// Runtime dispatch for [`LinearLanesBf16Op`].
pub struct LinearLanesBf16RuntimeOp<D: FloatBytes + Send + Sync + 'static> {
    k: usize,
    n: usize,
    _phantom: PhantomData<D>,
}

impl<D: FloatBytes + Send + Sync + 'static> RuntimeOp for LinearLanesBf16RuntimeOp<D> {
    fn n_activation_inputs(&self) -> usize {
        1
    }

    fn param_shapes(&self, _input_shapes: &[&[usize]], _output_shape: &[usize]) -> Vec<Vec<usize>> {
        vec![vec![self.n, self.k / 2]]
    }

    fn param_names(&self) -> &'static [&'static str] {
        &["weight_lanes"]
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
        assert_eq!(inputs[0].1[1], self.k, "linear lanes: input features");
        visitor.visit_ptr(inputs[0].0);
        visitor.visit_ptr(params[0]);
        visitor.visit_ptr(output);
        visitor.visit_i32(output_shape[0] as i32);
        visitor.visit_i32(self.n as i32);
        visitor.visit_i32(self.k as i32);
    }

    fn grid(&self, output_shape: &[usize]) -> [u32; 3] {
        [
            (self.n / LANES) as u32,
            output_shape[0].div_ceil(ROW_TILE) as u32,
            1,
        ]
    }
}

// ---------------------------------------------------------------------------
// Int8 weights, output features as lanes
// ---------------------------------------------------------------------------

/// Output features per program of [`linear_lanes_i8_forward`]. Twice [`LANES`]: a
/// step down `K` is then one 32-byte load, a widen and two multiply-adds per row for
/// 32 outputs, where 16 lanes spent the same instructions on half the bytes and left
/// the kernel short of memory bandwidth.
pub const I8_LANES: usize = 32;

/// The name a converted checkpoint stores a linear's int8 buffer under, beside its
/// weight. The lane width is in the name because it is in the byte order.
pub fn int8_name(weight: &str) -> String {
    format!("{weight}.i8x{I8_LANES}")
}

/// [`linear_lanes_bf16_forward`] on int8 weights with one f32 scale per output.
///
/// Decode reads every linear weight once per token and runs at memory bandwidth,
/// so a byte per weight instead of two is the speedup. A weight is stored as
/// `q + 128` in a byte (`q` in -127..=127) and becomes `q` with a widen and a
/// subtract; the row's scale multiplies the sum once at the end.
///
/// The buffer is `[N / I8_LANES][K][I8_LANES]` bytes, then `N` f32 scales; `w_ptr` and
/// `s_ptr` are the same buffer, read as bytes and as f32.
///
/// Grid: `(N / I8_LANES, ceil(M / ROW_TILE))`.
#[allow(clippy::multiple_bound_locations)]
#[kernel]
pub fn linear_lanes_i8_forward<T: Triton, D: Float, const LANES: i32>(
    x_ptr: InPtr<T::Pointer<D>>,
    w_ptr: InPtr<T::Pointer<u8>>,
    s_ptr: InPtr<T::Pointer<D>>,
    y_ptr: OutPtr<T::Pointer<D>>,
    M: i32,
    N: i32,
    K: i32,
) where
    T: Triton<I32Tensor = <T as Triton>::Tensor<i32>>,
    T::I32Tensor: Tensor<i32, 1>,
    T::I32Tensor: core::ops::Add<i32, Output = T::I32Tensor>,
    T::I32Tensor: core::ops::Sub<i32, Output = T::I32Tensor>,
    T::Pointer<D>: AddOffsets<i32, 1, T::I32Tensor, Output = T::Tensor<T::Pointer<D>>>,
    T::Pointer<u8>: AddOffsets<i32, 1, T::I32Tensor, Output = T::Tensor<T::Pointer<u8>>>,
{
    let nb = T::program_id(Axis::X);
    let rt = T::program_id(Axis::Y);
    let lanes = T::arange(0, LANES);
    let splat = T::zeros::<i32>(&[LANES]);
    let wbase = nb * K * LANES;
    let r0 = rt * 4;
    let col = nb * LANES;
    let scale = T::load(
        s_ptr.add_offsets(lanes + (N * K / 4 + col)),
        None,
        None,
        &[],
        None,
        None,
        None,
        false,
    );

    if M - r0 == 1 {
        let x0 = r0 * K;
        let mut a0 = T::zeros::<D>(&[LANES]);
        for k in 0..K {
            let q = T::cast::<i32, D>(
                T::cast::<u8, i32>(
                    T::load(
                        w_ptr.add_offsets(lanes + (wbase + k * LANES)),
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
                ) - 128,
                None,
                false,
            );
            a0 = a0
                + q * T::load(
                    x_ptr.add_offsets(splat + (x0 + k)),
                    None,
                    None,
                    &[],
                    None,
                    None,
                    None,
                    false,
                );
        }
        T::store(
            y_ptr.add_offsets(lanes + (r0 * N + col)),
            a0 * scale,
            None,
            &[],
            None,
            None,
        );
    } else {
        let last = M - 1;
        let r1 = if r0 + 1 < M { r0 + 1 } else { last };
        let r2 = if r0 + 2 < M { r0 + 2 } else { last };
        let r3 = if r0 + 3 < M { r0 + 3 } else { last };
        let x0 = r0 * K;
        let x1 = r1 * K;
        let x2 = r2 * K;
        let x3 = r3 * K;
        let mut a0 = T::zeros::<D>(&[LANES]);
        let mut a1 = T::zeros::<D>(&[LANES]);
        let mut a2 = T::zeros::<D>(&[LANES]);
        let mut a3 = T::zeros::<D>(&[LANES]);
        for k in 0..K {
            let q = T::cast::<i32, D>(
                T::cast::<u8, i32>(
                    T::load(
                        w_ptr.add_offsets(lanes + (wbase + k * LANES)),
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
                ) - 128,
                None,
                false,
            );
            a0 = a0
                + q * T::load(
                    x_ptr.add_offsets(splat + (x0 + k)),
                    None,
                    None,
                    &[],
                    None,
                    None,
                    None,
                    false,
                );
            a1 = a1
                + q * T::load(
                    x_ptr.add_offsets(splat + (x1 + k)),
                    None,
                    None,
                    &[],
                    None,
                    None,
                    None,
                    false,
                );
            a2 = a2
                + q * T::load(
                    x_ptr.add_offsets(splat + (x2 + k)),
                    None,
                    None,
                    &[],
                    None,
                    None,
                    None,
                    false,
                );
            a3 = a3
                + q * T::load(
                    x_ptr.add_offsets(splat + (x3 + k)),
                    None,
                    None,
                    &[],
                    None,
                    None,
                    None,
                    false,
                );
        }
        T::store(
            y_ptr.add_offsets(lanes + (r0 * N + col)),
            a0 * scale,
            None,
            &[],
            None,
            None,
        );
        T::store(
            y_ptr.add_offsets(lanes + (r1 * N + col)),
            a1 * scale,
            None,
            &[],
            None,
            None,
        );
        T::store(
            y_ptr.add_offsets(lanes + (r2 * N + col)),
            a2 * scale,
            None,
            &[],
            None,
            None,
        );
        T::store(
            y_ptr.add_offsets(lanes + (r3 * N + col)),
            a3 * scale,
            None,
            &[],
            None,
            None,
        );
    }
}

/// Quantises a row-major `[n, k]` matrix to the buffer [`linear_lanes_i8_forward`]
/// reads: symmetric int8 per output row (`scale = max |w| / 127`), stored as
/// `q + 128` in the `[n / I8_LANES][k][I8_LANES]` byte order, four bytes to a word, then
/// the `n` scales.
pub fn quantize_lanes_i8(values: &[f32], n: usize, k: usize) -> Result<Vec<f32>, String> {
    if n % I8_LANES != 0 || k % 4 != 0 || values.len() != n * k {
        return Err(format!(
            "int8 lanes: {} values are not {n} rows (a multiple of {I8_LANES}) of {k} (a multiple of 4)",
            values.len()
        ));
    }
    let mut bytes = vec![0u8; n * k];
    let mut scales = Vec::with_capacity(n);
    for (r, row) in values.chunks_exact(k).enumerate() {
        let max = row.iter().fold(0.0f32, |m, v| m.max(v.abs()));
        let scale = if max > 0.0 { max / 127.0 } else { 1.0 };
        let (block, lane) = (r / I8_LANES, r % I8_LANES);
        for (c, &v) in row.iter().enumerate() {
            let q = (v / scale).round().clamp(-127.0, 127.0) as i32;
            bytes[(block * k + c) * I8_LANES + lane] = (q + 128) as u8;
        }
        scales.push(scale);
    }
    let mut out: Vec<f32> = bytes
        .chunks_exact(4)
        .map(|b| f32::from_bits(u32::from_le_bytes([b[0], b[1], b[2], b[3]])))
        .collect();
    out.extend(scales);
    Ok(out)
}

/// A bias-free linear on int8 weights with output features as lanes, for
/// [`linear_lanes_i8_forward`]. Weight slot `weight_i8`: [`quantize_lanes_i8`]'s
/// buffer, declared `[N, K / 4 + 1]` words.
pub struct LinearLanesI8Op<D: FloatBytes + Send + Sync + 'static> {
    /// Input features `K` (a multiple of 4).
    pub k: usize,
    /// Output features `N` (a multiple of [`I8_LANES`]).
    pub n: usize,
    _phantom: PhantomData<D>,
}

impl<D: FloatBytes + Send + Sync + 'static> LinearLanesI8Op<D> {
    /// Creates the op.
    pub fn new(k: usize, n: usize) -> Self {
        assert!(
            n.is_multiple_of(I8_LANES),
            "int8 lanes: {n} outputs are not a multiple of {I8_LANES}"
        );
        assert!(
            k.is_multiple_of(4),
            "int8 lanes: in features must be a multiple of 4"
        );
        Self {
            k,
            n,
            _phantom: PhantomData,
        }
    }
}

impl<D: FloatBytes + Send + Sync + 'static> CustomOp for LinearLanesI8Op<D> {
    fn name(&self) -> &str {
        "linear_lanes_i8"
    }

    fn infer_output_shape(&self, input_shapes: &[&Shape]) -> Shape {
        vec![input_shapes[0][0], Some(self.n)]
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn lower(&self) -> Option<(String, String, String, Arc<dyn RuntimeOp>)> {
        let kernel = LinearLanesI8Forward::<D>::new(I8_LANES as i32);
        Some((
            "linear_lanes_i8_forward".to_string(),
            kernel.source,
            "entry_point".to_string(),
            Arc::new(LinearLanesI8RuntimeOp::<D> {
                k: self.k,
                n: self.n,
                _phantom: PhantomData,
            }),
        ))
    }
}

/// Runtime dispatch for [`LinearLanesI8Op`].
pub struct LinearLanesI8RuntimeOp<D: FloatBytes + Send + Sync + 'static> {
    k: usize,
    n: usize,
    _phantom: PhantomData<D>,
}

impl<D: FloatBytes + Send + Sync + 'static> RuntimeOp for LinearLanesI8RuntimeOp<D> {
    fn n_activation_inputs(&self) -> usize {
        1
    }

    fn param_shapes(&self, _input_shapes: &[&[usize]], _output_shape: &[usize]) -> Vec<Vec<usize>> {
        vec![vec![self.n, self.k / 4 + 1]]
    }

    fn param_names(&self) -> &'static [&'static str] {
        &["weight_i8"]
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
        assert_eq!(inputs[0].1[1], self.k, "int8 lanes: input features");
        visitor.visit_ptr(inputs[0].0);
        visitor.visit_ptr(params[0]);
        visitor.visit_ptr(params[0]);
        visitor.visit_ptr(output);
        visitor.visit_i32(output_shape[0] as i32);
        visitor.visit_i32(self.n as i32);
        visitor.visit_i32(self.k as i32);
    }

    fn grid(&self, output_shape: &[usize]) -> [u32; 3] {
        [
            (self.n / I8_LANES) as u32,
            output_shape[0].div_ceil(ROW_TILE) as u32,
            1,
        ]
    }
}
