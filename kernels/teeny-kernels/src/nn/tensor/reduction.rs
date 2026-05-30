/*
 * Copyright (c) 2026 Teenygrad.
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

//! Reduction kernels — each CTA handles one output element (one "row" of the
//! flattened [outer, inner] view).  The caller is responsible for reshaping
//! the input to `[n_outer, n_inner]` before invoking these kernels.
//!
//! Grid: `[n_outer, 1, 1]`
//! Block: `[BLOCK_INNER, 1, 1]`

#![allow(non_snake_case)]

use teeny_core::dtype::{Float, Num};
use teeny_macros::kernel;
use teeny_triton::triton::{
    types::{AddOffsets, Comparison},
    *,
};

// ── Helper macro for reduction RuntimeOp ─────────────────────────────────────

/// Standard reduction RuntimeOp: input shape [outer * inner], output shape [outer].
/// pack_args: x_ptr, y_ptr, n_inner, n_outer
macro_rules! impl_reduce_num_runtime_op {
    ($Fwd:ident) => {
        impl<D: Num + Send + Sync + 'static> teeny_core::model::RuntimeOp for $Fwd<D> {
            fn n_activation_inputs(&self) -> usize { 1 }
            fn param_shapes(&self, _: &[&[usize]], _: &[usize]) -> Vec<Vec<usize>> { vec![] }
            fn pack_args(
                &self,
                inputs: &[(teeny_core::model::RawPtr, &[usize])],
                _: &[teeny_core::model::RawPtr],
                output: teeny_core::model::RawPtr,
                output_shape: &[usize],
                _: i32,
                visitor: &mut dyn teeny_core::device::program::ArgVisitor,
            ) {
                // output_shape has been reduced; we need input_shape for n_inner.
                // n_outer = product of output dims
                // n_inner = product of input dims / n_outer
                let n_outer: usize = output_shape.iter().product::<usize>().max(1);
                let n_total: usize = inputs[0].1.iter().product();
                let n_inner: usize = if n_outer > 0 { n_total / n_outer } else { n_total };
                visitor.visit_ptr(inputs[0].0);
                visitor.visit_ptr(output);
                visitor.visit_i32(n_inner as i32);
                visitor.visit_i32(n_outer as i32);
            }
            fn block(&self) -> [u32; 3] { [self.block_inner as u32, 1, 1] }
            fn grid(&self, output_shape: &[usize]) -> [u32; 3] {
                let n_outer: usize = output_shape.iter().product::<usize>().max(1);
                [n_outer as u32, 1, 1]
            }
        }
    };
}

macro_rules! impl_reduce_float_runtime_op {
    ($Fwd:ident) => {
        impl<D: Float + Send + Sync + 'static> teeny_core::model::RuntimeOp for $Fwd<D> {
            fn n_activation_inputs(&self) -> usize { 1 }
            fn param_shapes(&self, _: &[&[usize]], _: &[usize]) -> Vec<Vec<usize>> { vec![] }
            fn pack_args(
                &self,
                inputs: &[(teeny_core::model::RawPtr, &[usize])],
                _: &[teeny_core::model::RawPtr],
                output: teeny_core::model::RawPtr,
                output_shape: &[usize],
                _: i32,
                visitor: &mut dyn teeny_core::device::program::ArgVisitor,
            ) {
                let n_outer: usize = output_shape.iter().product::<usize>().max(1);
                let n_total: usize = inputs[0].1.iter().product();
                let n_inner: usize = if n_outer > 0 { n_total / n_outer } else { n_total };
                visitor.visit_ptr(inputs[0].0);
                visitor.visit_ptr(output);
                visitor.visit_i32(n_inner as i32);
                visitor.visit_i32(n_outer as i32);
            }
            fn block(&self) -> [u32; 3] { [self.block_inner as u32, 1, 1] }
            fn grid(&self, output_shape: &[usize]) -> [u32; 3] {
                let n_outer: usize = output_shape.iter().product::<usize>().max(1);
                [n_outer as u32, 1, 1]
            }
        }
    };
}

// ── ReduceSum ─────────────────────────────────────────────────────────────────

/// Forward: y[row] = sum(x[row, :])
#[kernel]
pub fn reduce_sum_forward<T: Triton, D: Num, const BLOCK_INNER: i32>(
    x_ptr: T::Pointer<D>,
    y_ptr: T::Pointer<D>,
    n_inner: i32,
    n_outer: i32,
) where
    T::I32Tensor: types::Tensor<i32, 1>,
    T::I32Tensor: Comparison<i32, BoolTensor = T::BoolTensor>,
    T::Pointer<D>: AddOffsets<i32, 1, T::I32Tensor, Output = T::Tensor<T::Pointer<D>>>,
{
    let row = T::program_id(Axis::X);
    if row >= n_outer { return; }
    let col_offsets = T::arange(0, BLOCK_INNER);
    let offsets = col_offsets + row * n_inner;
    let mask = col_offsets.lt(n_inner);
    let x = T::load(
        x_ptr.add_offsets(offsets),
        Some(mask),
        Some(T::zeros::<D>(&[BLOCK_INNER])),
        &[], None, None, None, false,
    );
    let sum = T::sum(x, Some(0), true); // [1] or scalar
    let row_offsets = T::arange(0, 1) + row;
    T::store(y_ptr.add_offsets(row_offsets), sum, None, &[], None, None);
}

impl_reduce_num_runtime_op!(ReduceSumForward);

// ── ReduceMean ────────────────────────────────────────────────────────────────

/// Forward: y[row] = mean(x[row, :])
#[kernel]
pub fn reduce_mean_forward<T: Triton, D: Float, const BLOCK_INNER: i32>(
    x_ptr: T::Pointer<D>,
    y_ptr: T::Pointer<D>,
    n_inner: i32,
    n_outer: i32,
) where
    T::I32Tensor: types::Tensor<i32, 1>,
    T::I32Tensor: Comparison<i32, BoolTensor = T::BoolTensor>,
    T::Pointer<D>: AddOffsets<i32, 1, T::I32Tensor, Output = T::Tensor<T::Pointer<D>>>,
{
    let row = T::program_id(Axis::X);
    if row >= n_outer { return; }
    let col_offsets = T::arange(0, BLOCK_INNER);
    let offsets = col_offsets + row * n_inner;
    let mask = col_offsets.lt(n_inner);
    let x = T::load(
        x_ptr.add_offsets(offsets),
        Some(mask),
        Some(T::zeros::<D>(&[BLOCK_INNER])),
        &[], None, None, None, false,
    );
    let sum = T::sum(x, Some(0), true);
    let n_f = T::cast::<i32, D>(T::full::<i32>(&[1], n_inner), None, false);
    let mean = sum / n_f;
    let row_offsets = T::arange(0, 1) + row;
    T::store(y_ptr.add_offsets(row_offsets), mean, None, &[], None, None);
}

impl_reduce_float_runtime_op!(ReduceMeanForward);

// ── ReduceMax ─────────────────────────────────────────────────────────────────

/// Forward: y[row] = max(x[row, :])
#[kernel]
pub fn reduce_max_forward<T: Triton, D: Num, const BLOCK_INNER: i32>(
    x_ptr: T::Pointer<D>,
    y_ptr: T::Pointer<D>,
    n_inner: i32,
    n_outer: i32,
) where
    T::I32Tensor: types::Tensor<i32, 1>,
    T::I32Tensor: Comparison<i32, BoolTensor = T::BoolTensor>,
    T::Pointer<D>: AddOffsets<i32, 1, T::I32Tensor, Output = T::Tensor<T::Pointer<D>>>,
{
    let row = T::program_id(Axis::X);
    if row >= n_outer { return; }
    let col_offsets = T::arange(0, BLOCK_INNER);
    let offsets = col_offsets + row * n_inner;
    let mask = col_offsets.lt(n_inner);
    // Load with a very small fill value for masked lanes
    let neg_inf = T::cast::<f32, D>(T::full::<f32>(&[BLOCK_INNER], -3.4028235e38_f32), None, false);
    let x = T::load(
        x_ptr.add_offsets(offsets),
        Some(mask),
        Some(neg_inf),
        &[], None, None, None, false,
    );
    let val = T::max(x, Some(0), true);
    let row_offsets = T::arange(0, 1) + row;
    T::store(y_ptr.add_offsets(row_offsets), val, None, &[], None, None);
}

impl_reduce_num_runtime_op!(ReduceMaxForward);

// ── ReduceMin ─────────────────────────────────────────────────────────────────

/// Forward: y[row] = min(x[row, :])
#[kernel]
pub fn reduce_min_forward<T: Triton, D: Num, const BLOCK_INNER: i32>(
    x_ptr: T::Pointer<D>,
    y_ptr: T::Pointer<D>,
    n_inner: i32,
    n_outer: i32,
) where
    T::I32Tensor: types::Tensor<i32, 1>,
    T::I32Tensor: Comparison<i32, BoolTensor = T::BoolTensor>,
    T::Pointer<D>: AddOffsets<i32, 1, T::I32Tensor, Output = T::Tensor<T::Pointer<D>>>,
{
    let row = T::program_id(Axis::X);
    if row >= n_outer { return; }
    let col_offsets = T::arange(0, BLOCK_INNER);
    let offsets = col_offsets + row * n_inner;
    let mask = col_offsets.lt(n_inner);
    let pos_inf = T::cast::<f32, D>(T::full::<f32>(&[BLOCK_INNER], 3.4028235e38_f32), None, false);
    let x = T::load(
        x_ptr.add_offsets(offsets),
        Some(mask),
        Some(pos_inf),
        &[], None, None, None, false,
    );
    let val = T::min(x, Some(0), true);
    let row_offsets = T::arange(0, 1) + row;
    T::store(y_ptr.add_offsets(row_offsets), val, None, &[], None, None);
}

impl_reduce_num_runtime_op!(ReduceMinForward);

// ── ReduceL1 ──────────────────────────────────────────────────────────────────

/// Forward: y[row] = sum(|x[row, :]|)
#[kernel]
pub fn reduce_l1_forward<T: Triton, D: Num, const BLOCK_INNER: i32>(
    x_ptr: T::Pointer<D>,
    y_ptr: T::Pointer<D>,
    n_inner: i32,
    n_outer: i32,
) where
    T::I32Tensor: types::Tensor<i32, 1>,
    T::I32Tensor: Comparison<i32, BoolTensor = T::BoolTensor>,
    T::Pointer<D>: AddOffsets<i32, 1, T::I32Tensor, Output = T::Tensor<T::Pointer<D>>>,
{
    let row = T::program_id(Axis::X);
    if row >= n_outer { return; }
    let col_offsets = T::arange(0, BLOCK_INNER);
    let offsets = col_offsets + row * n_inner;
    let mask = col_offsets.lt(n_inner);
    let x = T::load(
        x_ptr.add_offsets(offsets),
        Some(mask),
        Some(T::zeros::<D>(&[BLOCK_INNER])),
        &[], None, None, None, false,
    );
    let val = T::sum(T::abs(x), Some(0), true);
    let row_offsets = T::arange(0, 1) + row;
    T::store(y_ptr.add_offsets(row_offsets), val, None, &[], None, None);
}

impl_reduce_num_runtime_op!(ReduceL1Forward);

// ── ReduceL2 ──────────────────────────────────────────────────────────────────

/// Forward: y[row] = sqrt(sum(x[row, :]^2))
#[kernel]
pub fn reduce_l2_forward<T: Triton, D: Float, const BLOCK_INNER: i32>(
    x_ptr: T::Pointer<D>,
    y_ptr: T::Pointer<D>,
    n_inner: i32,
    n_outer: i32,
) where
    T::I32Tensor: types::Tensor<i32, 1>,
    T::I32Tensor: Comparison<i32, BoolTensor = T::BoolTensor>,
    T::Pointer<D>: AddOffsets<i32, 1, T::I32Tensor, Output = T::Tensor<T::Pointer<D>>>,
{
    let row = T::program_id(Axis::X);
    if row >= n_outer { return; }
    let col_offsets = T::arange(0, BLOCK_INNER);
    let offsets = col_offsets + row * n_inner;
    let mask = col_offsets.lt(n_inner);
    let x = T::load(
        x_ptr.add_offsets(offsets),
        Some(mask),
        Some(T::zeros::<D>(&[BLOCK_INNER])),
        &[], None, None, None, false,
    );
    let sum_sq = T::sum(x * x, Some(0), true);
    let val = T::sqrt(sum_sq);
    let row_offsets = T::arange(0, 1) + row;
    T::store(y_ptr.add_offsets(row_offsets), val, None, &[], None, None);
}

impl_reduce_float_runtime_op!(ReduceL2Forward);

// ── ReduceSumSquare ───────────────────────────────────────────────────────────

/// Forward: y[row] = sum(x[row, :]^2)
#[kernel]
pub fn reduce_sum_square_forward<T: Triton, D: Num, const BLOCK_INNER: i32>(
    x_ptr: T::Pointer<D>,
    y_ptr: T::Pointer<D>,
    n_inner: i32,
    n_outer: i32,
) where
    T::I32Tensor: types::Tensor<i32, 1>,
    T::I32Tensor: Comparison<i32, BoolTensor = T::BoolTensor>,
    T::Pointer<D>: AddOffsets<i32, 1, T::I32Tensor, Output = T::Tensor<T::Pointer<D>>>,
{
    let row = T::program_id(Axis::X);
    if row >= n_outer { return; }
    let col_offsets = T::arange(0, BLOCK_INNER);
    let offsets = col_offsets + row * n_inner;
    let mask = col_offsets.lt(n_inner);
    let x = T::load(
        x_ptr.add_offsets(offsets),
        Some(mask),
        Some(T::zeros::<D>(&[BLOCK_INNER])),
        &[], None, None, None, false,
    );
    let val = T::sum(x * x, Some(0), true);
    let row_offsets = T::arange(0, 1) + row;
    T::store(y_ptr.add_offsets(row_offsets), val, None, &[], None, None);
}

impl_reduce_num_runtime_op!(ReduceSumSquareForward);

// ── ReduceLogSum ──────────────────────────────────────────────────────────────

/// Forward: y[row] = log(sum(x[row, :]))  (numerically unsafe; use ReduceLogSumExp for stable)
#[kernel]
pub fn reduce_log_sum_forward<T: Triton, D: Float, const BLOCK_INNER: i32>(
    x_ptr: T::Pointer<D>,
    y_ptr: T::Pointer<D>,
    n_inner: i32,
    n_outer: i32,
) where
    T::I32Tensor: types::Tensor<i32, 1>,
    T::I32Tensor: Comparison<i32, BoolTensor = T::BoolTensor>,
    T::Pointer<D>: AddOffsets<i32, 1, T::I32Tensor, Output = T::Tensor<T::Pointer<D>>>,
{
    let row = T::program_id(Axis::X);
    if row >= n_outer { return; }
    let col_offsets = T::arange(0, BLOCK_INNER);
    let offsets = col_offsets + row * n_inner;
    let mask = col_offsets.lt(n_inner);
    let x = T::load(
        x_ptr.add_offsets(offsets),
        Some(mask),
        Some(T::zeros::<D>(&[BLOCK_INNER])),
        &[], None, None, None, false,
    );
    let sum = T::sum(x, Some(0), true);
    let val = T::log(sum);
    let row_offsets = T::arange(0, 1) + row;
    T::store(y_ptr.add_offsets(row_offsets), val, None, &[], None, None);
}

impl_reduce_float_runtime_op!(ReduceLogSumForward);

// ── ReduceLogSumExp ───────────────────────────────────────────────────────────

/// Forward: y[row] = log(sum(exp(x[row, :]))) — numerically stable via max subtraction
#[kernel]
pub fn reduce_log_sum_exp_forward<T: Triton, D: Float, const BLOCK_INNER: i32>(
    x_ptr: T::Pointer<D>,
    y_ptr: T::Pointer<D>,
    n_inner: i32,
    n_outer: i32,
) where
    T::I32Tensor: types::Tensor<i32, 1>,
    T::I32Tensor: Comparison<i32, BoolTensor = T::BoolTensor>,
    T::Pointer<D>: AddOffsets<i32, 1, T::I32Tensor, Output = T::Tensor<T::Pointer<D>>>,
{
    let row = T::program_id(Axis::X);
    if row >= n_outer { return; }
    let col_offsets = T::arange(0, BLOCK_INNER);
    let offsets = col_offsets + row * n_inner;
    let mask = col_offsets.lt(n_inner);
    let neg_inf = T::cast::<f32, D>(T::full::<f32>(&[BLOCK_INNER], -3.4028235e38_f32), None, false);
    let x = T::load(
        x_ptr.add_offsets(offsets),
        Some(mask),
        Some(neg_inf),
        &[], None, None, None, false,
    );
    // Numerically stable: log(sum(exp(x))) = m + log(sum(exp(x - m)))
    // where m = max(x)
    let m = T::max(x, Some(0), true); // [1]
    let fill = T::cast::<f32, D>(T::full::<f32>(&[BLOCK_INNER], 0.0_f32), None, false);
    let x_adj = T::load(
        x_ptr.add_offsets(offsets),
        Some(mask),
        Some(fill),
        &[], None, None, None, false,
    );
    let sum_exp = T::sum(T::exp(x_adj - m), Some(0), true);
    let val = m + T::log(sum_exp);
    let row_offsets = T::arange(0, 1) + row;
    T::store(y_ptr.add_offsets(row_offsets), val, None, &[], None, None);
}

impl_reduce_float_runtime_op!(ReduceLogSumExpForward);

// ── ReduceProd ────────────────────────────────────────────────────────────────

/// Forward: y[row] = prod(x[row, :])
/// Note: implemented as exp(sum(log(x))) — only valid for positive x.
/// For general use this is a placeholder.
#[kernel]
pub fn reduce_prod_forward<T: Triton, D: Float, const BLOCK_INNER: i32>(
    x_ptr: T::Pointer<D>,
    y_ptr: T::Pointer<D>,
    n_inner: i32,
    n_outer: i32,
) where
    T::I32Tensor: types::Tensor<i32, 1>,
    T::I32Tensor: Comparison<i32, BoolTensor = T::BoolTensor>,
    T::Pointer<D>: AddOffsets<i32, 1, T::I32Tensor, Output = T::Tensor<T::Pointer<D>>>,
{
    let row = T::program_id(Axis::X);
    if row >= n_outer { return; }
    let col_offsets = T::arange(0, BLOCK_INNER);
    let offsets = col_offsets + row * n_inner;
    let mask = col_offsets.lt(n_inner);
    // Fill with 1.0 for masked-off lanes so they don't affect the product.
    let one_fill = T::full(&[BLOCK_INNER], D::ONE);
    let x = T::load(
        x_ptr.add_offsets(offsets),
        Some(mask),
        Some(one_fill),
        &[], None, None, None, false,
    );
    // exp(sum(log(x))) approximates product for positive x.
    let val = T::exp(T::sum(T::log(x), Some(0), true));
    let row_offsets = T::arange(0, 1) + row;
    T::store(y_ptr.add_offsets(row_offsets), val, None, &[], None, None);
}

impl_reduce_float_runtime_op!(ReduceProdForward);

// ── CumSum ────────────────────────────────────────────────────────────────────

/// Forward: y = cumsum(x, axis=0) over a 1-D block
/// Each CTA handles one complete row (n_inner elements).
#[kernel]
pub fn cum_sum_forward<T: Triton, D: Num, const BLOCK_INNER: i32>(
    x_ptr: T::Pointer<D>,
    y_ptr: T::Pointer<D>,
    n_inner: i32,
    n_outer: i32,
) where
    T::I32Tensor: types::Tensor<i32, 1>,
    T::I32Tensor: Comparison<i32, BoolTensor = T::BoolTensor>,
    T::Pointer<D>: AddOffsets<i32, 1, T::I32Tensor, Output = T::Tensor<T::Pointer<D>>>,
{
    let row = T::program_id(Axis::X);
    if row >= n_outer { return; }
    let col_offsets = T::arange(0, BLOCK_INNER);
    let offsets = col_offsets + row * n_inner;
    let mask = col_offsets.lt(n_inner);
    let x = T::load(
        x_ptr.add_offsets(offsets),
        Some(mask),
        Some(T::zeros::<D>(&[BLOCK_INNER])),
        &[], None, None, None, false,
    );
    // Use Triton's cumsum: axis=0 over the 1-D block, not reversed.
    let y = T::cumsum(x, 0, false);
    T::store(y_ptr.add_offsets(offsets), y, Some(mask), &[], None, None);
}

impl<D: Num + Send + Sync + 'static> teeny_core::model::RuntimeOp for CumSumForward<D> {
    fn n_activation_inputs(&self) -> usize { 1 }
    fn param_shapes(&self, _: &[&[usize]], _: &[usize]) -> Vec<Vec<usize>> { vec![] }
    fn pack_args(
        &self,
        inputs: &[(teeny_core::model::RawPtr, &[usize])],
        _: &[teeny_core::model::RawPtr],
        output: teeny_core::model::RawPtr,
        output_shape: &[usize],
        _: i32,
        visitor: &mut dyn teeny_core::device::program::ArgVisitor,
    ) {
        // For cumsum: output_shape == input_shape; n_outer = all dims except last
        let n_total: usize = output_shape.iter().product();
        let n_inner = output_shape.last().copied().unwrap_or(1);
        let n_outer = n_total / n_inner;
        visitor.visit_ptr(inputs[0].0);
        visitor.visit_ptr(output);
        visitor.visit_i32(n_inner as i32);
        visitor.visit_i32(n_outer as i32);
    }
    fn block(&self) -> [u32; 3] { [self.block_inner as u32, 1, 1] }
    fn grid(&self, output_shape: &[usize]) -> [u32; 3] {
        let n_total: usize = output_shape.iter().product();
        let n_inner = output_shape.last().copied().unwrap_or(1);
        let n_outer = n_total / n_inner;
        [n_outer as u32, 1, 1]
    }
}

// ── CumProd ───────────────────────────────────────────────────────────────────

/// Forward: y = cumprod(x, axis=0) over a 1-D block
#[kernel]
pub fn cum_prod_forward<T: Triton, D: Num, const BLOCK_INNER: i32>(
    x_ptr: T::Pointer<D>,
    y_ptr: T::Pointer<D>,
    n_inner: i32,
    n_outer: i32,
) where
    T::I32Tensor: types::Tensor<i32, 1>,
    T::I32Tensor: Comparison<i32, BoolTensor = T::BoolTensor>,
    T::Pointer<D>: AddOffsets<i32, 1, T::I32Tensor, Output = T::Tensor<T::Pointer<D>>>,
{
    let row = T::program_id(Axis::X);
    if row >= n_outer { return; }
    let col_offsets = T::arange(0, BLOCK_INNER);
    let offsets = col_offsets + row * n_inner;
    let mask = col_offsets.lt(n_inner);
    let x = T::load(
        x_ptr.add_offsets(offsets),
        Some(mask),
        Some(T::zeros::<D>(&[BLOCK_INNER])),
        &[], None, None, None, false,
    );
    let y = T::cumprod(x, 0, false);
    T::store(y_ptr.add_offsets(offsets), y, Some(mask), &[], None, None);
}

impl<D: Num + Send + Sync + 'static> teeny_core::model::RuntimeOp for CumProdForward<D> {
    fn n_activation_inputs(&self) -> usize { 1 }
    fn param_shapes(&self, _: &[&[usize]], _: &[usize]) -> Vec<Vec<usize>> { vec![] }
    fn pack_args(
        &self,
        inputs: &[(teeny_core::model::RawPtr, &[usize])],
        _: &[teeny_core::model::RawPtr],
        output: teeny_core::model::RawPtr,
        output_shape: &[usize],
        _: i32,
        visitor: &mut dyn teeny_core::device::program::ArgVisitor,
    ) {
        let n_total: usize = output_shape.iter().product();
        let n_inner = output_shape.last().copied().unwrap_or(1);
        let n_outer = n_total / n_inner;
        visitor.visit_ptr(inputs[0].0);
        visitor.visit_ptr(output);
        visitor.visit_i32(n_inner as i32);
        visitor.visit_i32(n_outer as i32);
    }
    fn block(&self) -> [u32; 3] { [self.block_inner as u32, 1, 1] }
    fn grid(&self, output_shape: &[usize]) -> [u32; 3] {
        let n_total: usize = output_shape.iter().product();
        let n_inner = output_shape.last().copied().unwrap_or(1);
        let n_outer = n_total / n_inner;
        [n_outer as u32, 1, 1]
    }
}

// ArgMax and ArgMin kernels are deferred — the Triton type system requires
// I32Tensor → Tensor<i32> coercion that isn't directly supported via #[kernel].
// These are handled as TODO in the lowering match arm.

// ── GlobalAvgPool ─────────────────────────────────────────────────────────────
//
// Treats input as [n_outer, n_inner] and averages over n_inner.
// For a [N, C, H, W] input: n_outer = N * C, n_inner = H * W.

/// Forward: y[row] = mean(x[row, :])  (same as ReduceMean)
#[kernel]
pub fn global_avg_pool_forward<T: Triton, D: Float, const BLOCK_INNER: i32>(
    x_ptr: T::Pointer<D>,
    y_ptr: T::Pointer<D>,
    n_inner: i32,
    n_outer: i32,
) where
    T::I32Tensor: types::Tensor<i32, 1>,
    T::I32Tensor: Comparison<i32, BoolTensor = T::BoolTensor>,
    T::Pointer<D>: AddOffsets<i32, 1, T::I32Tensor, Output = T::Tensor<T::Pointer<D>>>,
{
    let row = T::program_id(Axis::X);
    if row >= n_outer { return; }
    let col_offsets = T::arange(0, BLOCK_INNER);
    let offsets = col_offsets + row * n_inner;
    let mask = col_offsets.lt(n_inner);
    let x = T::load(
        x_ptr.add_offsets(offsets),
        Some(mask),
        Some(T::zeros::<D>(&[BLOCK_INNER])),
        &[], None, None, None, false,
    );
    let sum = T::sum(x, Some(0), true);
    let n_f = T::cast::<i32, D>(T::full::<i32>(&[1], n_inner), None, false);
    let mean = sum / n_f;
    let row_offsets = T::arange(0, 1) + row;
    T::store(y_ptr.add_offsets(row_offsets), mean, None, &[], None, None);
}

impl_reduce_float_runtime_op!(GlobalAvgPoolForward);

// ── GlobalMaxPool ─────────────────────────────────────────────────────────────

/// Forward: y[row] = max(x[row, :])  (same as ReduceMax)
#[kernel]
pub fn global_max_pool_forward<T: Triton, D: Float, const BLOCK_INNER: i32>(
    x_ptr: T::Pointer<D>,
    y_ptr: T::Pointer<D>,
    n_inner: i32,
    n_outer: i32,
) where
    T::I32Tensor: types::Tensor<i32, 1>,
    T::I32Tensor: Comparison<i32, BoolTensor = T::BoolTensor>,
    T::Pointer<D>: AddOffsets<i32, 1, T::I32Tensor, Output = T::Tensor<T::Pointer<D>>>,
{
    let row = T::program_id(Axis::X);
    if row >= n_outer { return; }
    let col_offsets = T::arange(0, BLOCK_INNER);
    let offsets = col_offsets + row * n_inner;
    let mask = col_offsets.lt(n_inner);
    let neg_inf = T::cast::<f32, D>(T::full::<f32>(&[BLOCK_INNER], -3.4028235e38_f32), None, false);
    let x = T::load(
        x_ptr.add_offsets(offsets),
        Some(mask),
        Some(neg_inf),
        &[], None, None, None, false,
    );
    let val = T::max(x, Some(0), true);
    let row_offsets = T::arange(0, 1) + row;
    T::store(y_ptr.add_offsets(row_offsets), val, None, &[], None, None);
}

impl_reduce_float_runtime_op!(GlobalMaxPoolForward);
