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

//! Reduction kernels — each CTA handles one output element (one "row" of the
//! flattened [outer, inner] view).  The caller is responsible for reshaping
//! the input to `[n_outer, n_inner]` before invoking these kernels.
//!
//! Grid: `[n_outer, 1, 1]`
//! Block: `[BLOCK_INNER, 1, 1]`

#![allow(non_snake_case)]

use teeny_core::dtype::{Float, Num};
use teeny_macros::{kernel, tiled_kernel};
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
            fn n_activation_inputs(&self) -> usize {
                1
            }
            fn param_shapes(&self, _: &[&[usize]], _: &[usize]) -> Vec<Vec<usize>> {
                vec![]
            }
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
                let n_inner: usize = if n_outer > 0 {
                    n_total / n_outer
                } else {
                    n_total
                };
                visitor.visit_ptr(inputs[0].0);
                visitor.visit_ptr(output);
                visitor.visit_i32(n_inner as i32);
                visitor.visit_i32(n_outer as i32);
            }
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
            fn n_activation_inputs(&self) -> usize {
                1
            }
            fn param_shapes(&self, _: &[&[usize]], _: &[usize]) -> Vec<Vec<usize>> {
                vec![]
            }
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
                let n_inner: usize = if n_outer > 0 {
                    n_total / n_outer
                } else {
                    n_total
                };
                visitor.visit_ptr(inputs[0].0);
                visitor.visit_ptr(output);
                visitor.visit_i32(n_inner as i32);
                visitor.visit_i32(n_outer as i32);
            }
            fn grid(&self, output_shape: &[usize]) -> [u32; 3] {
                let n_outer: usize = output_shape.iter().product::<usize>().max(1);
                [n_outer as u32, 1, 1]
            }
        }
    };
}

// ── ReduceSum ─────────────────────────────────────────────────────────────────

/// Forward: y[row] = sum(x[row, :])
// ANCHOR: reduce_sum_forward
#[tiled_kernel]
// teenygrad-1tl.9. Rank-reducing: x is [n_outer, n_inner], y is [n_outer].
// The reduced axis has no counterpart in the output, and that resolves
// correctly without any new vocabulary -- an axis with no binding keeps its
// full extent, which here is the truth rather than a fallback: one output row
// needs its whole input row. Seeding a 4-row output tile gives x a tile of
// 4 x n_inner.
//
// `BLOCK_INNER` is deliberately NOT bound to the inner axis. It is the load
// width that covers the whole row under a mask, not a tiling of it; binding it
// would claim the axis is chunked when it is read in one piece. Same reasoning
// as GEMM's BLOCK_K (teenygrad-1tl.10).
//
// `block = 1` on the outer axis records the kernel's own granularity -- one
// row per program, `row = program_id(Axis::X)`, and there is no BLOCK_OUTER to
// name. It is documentation: `resolve_inputs` takes the block from the
// propagated output tile, so a multi-row tile still resolves correctly.
pub fn reduce_sum_forward<T: Triton, D: Num, const BLOCK_INNER: i32>(
    #[tile(block = 1, extent = n_outer)]
    #[tile(block = BLOCK_INNER, extent = n_inner, reduce)]
    x: In<Tile<T, D>>,
    #[tile(block = 1, extent = n_outer)] y: Out<Tile<T, D>>,
    n_inner: i32,
    n_outer: i32,
) where
    T::I32Tensor: types::Tensor<i32, 1>,
    T::I32Tensor: Comparison<i32, BoolTensor = T::BoolTensor>,
    T::Pointer<D>: AddOffsets<i32, 1, T::I32Tensor, Output = T::Tensor<T::Pointer<D>>>,
{
    // The reduced axis's range, its mask and the row offset are all generated
    // (teenygrad-29qp); this is the reduction itself and the store.
    T::store(
        y.tensor,
        T::sum(x.tensor, Some(0), true),
        None,
        &[],
        None,
        None,
    );
}

// ANCHOR_END: reduce_sum_forward

impl_reduce_num_runtime_op!(ReduceSumForward);

// ── ReduceMean ────────────────────────────────────────────────────────────────

/// Forward: y[row] = mean(x[row, :])
#[tiled_kernel]
// teenygrad-1tl.9. Rank-reducing: x is [n_outer, n_inner], y is [n_outer].
// The reduced axis has no counterpart in the output, and that resolves
// correctly without any new vocabulary -- an axis with no binding keeps its
// full extent, which here is the truth rather than a fallback: one output row
// needs its whole input row. Seeding a 4-row output tile gives x a tile of
// 4 x n_inner.
//
// `BLOCK_INNER` is deliberately NOT bound to the inner axis. It is the load
// width that covers the whole row under a mask, not a tiling of it; binding it
// would claim the axis is chunked when it is read in one piece. Same reasoning
// as GEMM's BLOCK_K (teenygrad-1tl.10).
//
// `block = 1` on the outer axis records the kernel's own granularity -- one
// row per program, `row = program_id(Axis::X)`, and there is no BLOCK_OUTER to
// name. It is documentation: `resolve_inputs` takes the block from the
// propagated output tile, so a multi-row tile still resolves correctly.
pub fn reduce_mean_forward<T: Triton, D: Float, const BLOCK_INNER: i32>(
    #[tile(block = 1, extent = n_outer)]
    #[tile(block = BLOCK_INNER, extent = n_inner, reduce)]
    x: In<Tile<T, D>>,
    #[tile(block = 1, extent = n_outer)] y: Out<Tile<T, D>>,
    n_inner: i32,
    n_outer: i32,
) where
    T::I32Tensor: types::Tensor<i32, 1>,
    T::I32Tensor: Comparison<i32, BoolTensor = T::BoolTensor>,
    T::Pointer<D>: AddOffsets<i32, 1, T::I32Tensor, Output = T::Tensor<T::Pointer<D>>>,
{
    T::store(
        y.tensor,
        T::sum(x.tensor, Some(0), true)
            / T::cast::<i32, D>(T::full::<i32>(&[1], n_inner), None, false),
        None,
        &[],
        None,
        None,
    );
}

impl_reduce_float_runtime_op!(ReduceMeanForward);

// ── ReduceMax ─────────────────────────────────────────────────────────────────

/// Forward: y[row] = max(x[row, :])
#[tiled_kernel]
// teenygrad-1tl.9. Rank-reducing: x is [n_outer, n_inner], y is [n_outer].
// The reduced axis has no counterpart in the output, and that resolves
// correctly without any new vocabulary -- an axis with no binding keeps its
// full extent, which here is the truth rather than a fallback: one output row
// needs its whole input row. Seeding a 4-row output tile gives x a tile of
// 4 x n_inner.
//
// `BLOCK_INNER` is deliberately NOT bound to the inner axis. It is the load
// width that covers the whole row under a mask, not a tiling of it; binding it
// would claim the axis is chunked when it is read in one piece. Same reasoning
// as GEMM's BLOCK_K (teenygrad-1tl.10).
//
// `block = 1` on the outer axis records the kernel's own granularity -- one
// row per program, `row = program_id(Axis::X)`, and there is no BLOCK_OUTER to
// name. It is documentation: `resolve_inputs` takes the block from the
// propagated output tile, so a multi-row tile still resolves correctly.
pub fn reduce_max_forward<T: Triton, D: Num, const BLOCK_INNER: i32>(
    #[tile(block = 1, extent = n_outer)]
    #[tile(block = BLOCK_INNER, extent = n_inner, reduce, fill = neg_inf)]
    x: In<Tile<T, D>>,
    #[tile(block = 1, extent = n_outer)] y: Out<Tile<T, D>>,
    n_inner: i32,
    n_outer: i32,
) where
    T::I32Tensor: types::Tensor<i32, 1>,
    T::I32Tensor: Comparison<i32, BoolTensor = T::BoolTensor>,
    T::Pointer<D>: AddOffsets<i32, 1, T::I32Tensor, Output = T::Tensor<T::Pointer<D>>>,
{
    T::store(
        y.tensor,
        T::max(x.tensor, Some(0), true),
        None,
        &[],
        None,
        None,
    );
}

impl_reduce_num_runtime_op!(ReduceMaxForward);

// ── ReduceMin ─────────────────────────────────────────────────────────────────

/// Forward: y[row] = min(x[row, :])
#[tiled_kernel]
// teenygrad-1tl.9. Rank-reducing: x is [n_outer, n_inner], y is [n_outer].
// The reduced axis has no counterpart in the output, and that resolves
// correctly without any new vocabulary -- an axis with no binding keeps its
// full extent, which here is the truth rather than a fallback: one output row
// needs its whole input row. Seeding a 4-row output tile gives x a tile of
// 4 x n_inner.
//
// `BLOCK_INNER` is deliberately NOT bound to the inner axis. It is the load
// width that covers the whole row under a mask, not a tiling of it; binding it
// would claim the axis is chunked when it is read in one piece. Same reasoning
// as GEMM's BLOCK_K (teenygrad-1tl.10).
//
// `block = 1` on the outer axis records the kernel's own granularity -- one
// row per program, `row = program_id(Axis::X)`, and there is no BLOCK_OUTER to
// name. It is documentation: `resolve_inputs` takes the block from the
// propagated output tile, so a multi-row tile still resolves correctly.
pub fn reduce_min_forward<T: Triton, D: Num, const BLOCK_INNER: i32>(
    #[tile(block = 1, extent = n_outer)]
    #[tile(block = BLOCK_INNER, extent = n_inner, reduce, fill = pos_inf)]
    x: In<Tile<T, D>>,
    #[tile(block = 1, extent = n_outer)] y: Out<Tile<T, D>>,
    n_inner: i32,
    n_outer: i32,
) where
    T::I32Tensor: types::Tensor<i32, 1>,
    T::I32Tensor: Comparison<i32, BoolTensor = T::BoolTensor>,
    T::Pointer<D>: AddOffsets<i32, 1, T::I32Tensor, Output = T::Tensor<T::Pointer<D>>>,
{
    T::store(
        y.tensor,
        T::min(x.tensor, Some(0), true),
        None,
        &[],
        None,
        None,
    );
}

impl_reduce_num_runtime_op!(ReduceMinForward);

// ── ReduceL1 ──────────────────────────────────────────────────────────────────

/// Forward: y[row] = sum(|x[row, :]|)
#[tiled_kernel]
// teenygrad-1tl.9. Rank-reducing: x is [n_outer, n_inner], y is [n_outer].
// The reduced axis has no counterpart in the output, and that resolves
// correctly without any new vocabulary -- an axis with no binding keeps its
// full extent, which here is the truth rather than a fallback: one output row
// needs its whole input row. Seeding a 4-row output tile gives x a tile of
// 4 x n_inner.
//
// `BLOCK_INNER` is deliberately NOT bound to the inner axis. It is the load
// width that covers the whole row under a mask, not a tiling of it; binding it
// would claim the axis is chunked when it is read in one piece. Same reasoning
// as GEMM's BLOCK_K (teenygrad-1tl.10).
//
// `block = 1` on the outer axis records the kernel's own granularity -- one
// row per program, `row = program_id(Axis::X)`, and there is no BLOCK_OUTER to
// name. It is documentation: `resolve_inputs` takes the block from the
// propagated output tile, so a multi-row tile still resolves correctly.
pub fn reduce_l1_forward<T: Triton, D: Num, const BLOCK_INNER: i32>(
    #[tile(block = 1, extent = n_outer)]
    #[tile(block = BLOCK_INNER, extent = n_inner, reduce)]
    x: In<Tile<T, D>>,
    #[tile(block = 1, extent = n_outer)] y: Out<Tile<T, D>>,
    n_inner: i32,
    n_outer: i32,
) where
    T::I32Tensor: types::Tensor<i32, 1>,
    T::I32Tensor: Comparison<i32, BoolTensor = T::BoolTensor>,
    T::Pointer<D>: AddOffsets<i32, 1, T::I32Tensor, Output = T::Tensor<T::Pointer<D>>>,
{
    T::store(
        y.tensor,
        T::sum(T::abs(x.tensor), Some(0), true),
        None,
        &[],
        None,
        None,
    );
}

impl_reduce_num_runtime_op!(ReduceL1Forward);

// ── ReduceL2 ──────────────────────────────────────────────────────────────────

/// Forward: y[row] = sqrt(sum(x[row, :]^2))
#[tiled_kernel]
// teenygrad-1tl.9. Rank-reducing: x is [n_outer, n_inner], y is [n_outer].
// The reduced axis has no counterpart in the output, and that resolves
// correctly without any new vocabulary -- an axis with no binding keeps its
// full extent, which here is the truth rather than a fallback: one output row
// needs its whole input row. Seeding a 4-row output tile gives x a tile of
// 4 x n_inner.
//
// `BLOCK_INNER` is deliberately NOT bound to the inner axis. It is the load
// width that covers the whole row under a mask, not a tiling of it; binding it
// would claim the axis is chunked when it is read in one piece. Same reasoning
// as GEMM's BLOCK_K (teenygrad-1tl.10).
//
// `block = 1` on the outer axis records the kernel's own granularity -- one
// row per program, `row = program_id(Axis::X)`, and there is no BLOCK_OUTER to
// name. It is documentation: `resolve_inputs` takes the block from the
// propagated output tile, so a multi-row tile still resolves correctly.
pub fn reduce_l2_forward<T: Triton, D: Float, const BLOCK_INNER: i32>(
    #[tile(block = 1, extent = n_outer)]
    #[tile(block = BLOCK_INNER, extent = n_inner, reduce)]
    x: In<Tile<T, D>>,
    #[tile(block = 1, extent = n_outer)] y: Out<Tile<T, D>>,
    n_inner: i32,
    n_outer: i32,
) where
    T::I32Tensor: types::Tensor<i32, 1>,
    T::I32Tensor: Comparison<i32, BoolTensor = T::BoolTensor>,
    T::Pointer<D>: AddOffsets<i32, 1, T::I32Tensor, Output = T::Tensor<T::Pointer<D>>>,
{
    T::store(
        y.tensor,
        T::sqrt(T::sum(x.tensor * x.tensor, Some(0), true)),
        None,
        &[],
        None,
        None,
    );
}

impl_reduce_float_runtime_op!(ReduceL2Forward);

// ── ReduceSumSquare ───────────────────────────────────────────────────────────

/// Forward: y[row] = sum(x[row, :]^2)
#[tiled_kernel]
// teenygrad-1tl.9. Rank-reducing: x is [n_outer, n_inner], y is [n_outer].
// The reduced axis has no counterpart in the output, and that resolves
// correctly without any new vocabulary -- an axis with no binding keeps its
// full extent, which here is the truth rather than a fallback: one output row
// needs its whole input row. Seeding a 4-row output tile gives x a tile of
// 4 x n_inner.
//
// `BLOCK_INNER` is deliberately NOT bound to the inner axis. It is the load
// width that covers the whole row under a mask, not a tiling of it; binding it
// would claim the axis is chunked when it is read in one piece. Same reasoning
// as GEMM's BLOCK_K (teenygrad-1tl.10).
//
// `block = 1` on the outer axis records the kernel's own granularity -- one
// row per program, `row = program_id(Axis::X)`, and there is no BLOCK_OUTER to
// name. It is documentation: `resolve_inputs` takes the block from the
// propagated output tile, so a multi-row tile still resolves correctly.
pub fn reduce_sum_square_forward<T: Triton, D: Num, const BLOCK_INNER: i32>(
    #[tile(block = 1, extent = n_outer)]
    #[tile(block = BLOCK_INNER, extent = n_inner, reduce)]
    x: In<Tile<T, D>>,
    #[tile(block = 1, extent = n_outer)] y: Out<Tile<T, D>>,
    n_inner: i32,
    n_outer: i32,
) where
    T::I32Tensor: types::Tensor<i32, 1>,
    T::I32Tensor: Comparison<i32, BoolTensor = T::BoolTensor>,
    T::Pointer<D>: AddOffsets<i32, 1, T::I32Tensor, Output = T::Tensor<T::Pointer<D>>>,
{
    T::store(
        y.tensor,
        T::sum(x.tensor * x.tensor, Some(0), true),
        None,
        &[],
        None,
        None,
    );
}

impl_reduce_num_runtime_op!(ReduceSumSquareForward);

// ── ReduceLogSum ──────────────────────────────────────────────────────────────

/// Forward: y[row] = log(sum(x[row, :]))  (numerically unsafe; use ReduceLogSumExp for stable)
#[tiled_kernel]
// teenygrad-1tl.9. Rank-reducing: x is [n_outer, n_inner], y is [n_outer].
// The reduced axis has no counterpart in the output, and that resolves
// correctly without any new vocabulary -- an axis with no binding keeps its
// full extent, which here is the truth rather than a fallback: one output row
// needs its whole input row. Seeding a 4-row output tile gives x a tile of
// 4 x n_inner.
//
// `BLOCK_INNER` is deliberately NOT bound to the inner axis. It is the load
// width that covers the whole row under a mask, not a tiling of it; binding it
// would claim the axis is chunked when it is read in one piece. Same reasoning
// as GEMM's BLOCK_K (teenygrad-1tl.10).
//
// `block = 1` on the outer axis records the kernel's own granularity -- one
// row per program, `row = program_id(Axis::X)`, and there is no BLOCK_OUTER to
// name. It is documentation: `resolve_inputs` takes the block from the
// propagated output tile, so a multi-row tile still resolves correctly.
pub fn reduce_log_sum_forward<T: Triton, D: Float, const BLOCK_INNER: i32>(
    #[tile(block = 1, extent = n_outer)]
    #[tile(block = BLOCK_INNER, extent = n_inner, reduce)]
    x: In<Tile<T, D>>,
    #[tile(block = 1, extent = n_outer)] y: Out<Tile<T, D>>,
    n_inner: i32,
    n_outer: i32,
) where
    T::I32Tensor: types::Tensor<i32, 1>,
    T::I32Tensor: Comparison<i32, BoolTensor = T::BoolTensor>,
    T::Pointer<D>: AddOffsets<i32, 1, T::I32Tensor, Output = T::Tensor<T::Pointer<D>>>,
{
    T::store(
        y.tensor,
        T::log(T::sum(x.tensor, Some(0), true)),
        None,
        &[],
        None,
        None,
    );
}

impl_reduce_float_runtime_op!(ReduceLogSumForward);

// ── ReduceLogSumExp ───────────────────────────────────────────────────────────

/// Forward: y[row] = log(sum(exp(x[row, :]))) — numerically stable via max subtraction
#[tiled_kernel]
// teenygrad-1tl.9. Rank-reducing: x is [n_outer, n_inner], y is [n_outer].
// The reduced axis has no counterpart in the output, and that resolves
// correctly without any new vocabulary -- an axis with no binding keeps its
// full extent, which here is the truth rather than a fallback: one output row
// needs its whole input row. Seeding a 4-row output tile gives x a tile of
// 4 x n_inner.
//
// `BLOCK_INNER` is deliberately NOT bound to the inner axis. It is the load
// width that covers the whole row under a mask, not a tiling of it; binding it
// would claim the axis is chunked when it is read in one piece. Same reasoning
// as GEMM's BLOCK_K (teenygrad-1tl.10).
//
// `block = 1` on the outer axis records the kernel's own granularity -- one
// row per program, `row = program_id(Axis::X)`, and there is no BLOCK_OUTER to
// name. It is documentation: `resolve_inputs` takes the block from the
// propagated output tile, so a multi-row tile still resolves correctly.
pub fn reduce_log_sum_exp_forward<T: Triton, D: Float, const BLOCK_INNER: i32>(
    #[tile(block = 1, extent = n_outer)]
    #[tile(block = BLOCK_INNER, extent = n_inner, reduce, fill = neg_inf)]
    x: In<Tile<T, D>>,
    #[tile(block = 1, extent = n_outer)] y: Out<Tile<T, D>>,
    n_inner: i32,
    n_outer: i32,
) where
    T::I32Tensor: types::Tensor<i32, 1>,
    T::I32Tensor: Comparison<i32, BoolTensor = T::BoolTensor>,
    T::Pointer<D>: AddOffsets<i32, 1, T::I32Tensor, Output = T::Tensor<T::Pointer<D>>>,
{
    // One load, read twice -- possible since `Tile` became `Copy`
    // (teenygrad-y8aa). The hand-written form loaded the row twice, the second
    // time with a 0.0 fill, which made a masked lane contribute `exp(0 - m)`
    // to the sum rather than nothing. With a single `-inf`-filled load a masked
    // lane gives `exp(-inf - m) == 0`, which is what the reduction wants. The
    // difference is invisible to the existing tests, whose `n_inner` equals
    // `BLOCK_INNER` -- see the partial-tile test added for exactly this
    // (teenygrad-29qp).
    let m = T::max(x.tensor, Some(0), true);
    let sum_exp = T::sum(T::exp(x.tensor - m), Some(0), true);
    T::store(y.tensor, m + T::log(sum_exp), None, &[], None, None);
}

impl_reduce_float_runtime_op!(ReduceLogSumExpForward);

// ── ReduceProd ────────────────────────────────────────────────────────────────

/// Forward: y[row] = prod(x[row, :])
/// Note: implemented as exp(sum(log(x))) — only valid for positive x.
/// For general use this is a placeholder.
#[tiled_kernel]
// teenygrad-1tl.9. Rank-reducing: x is [n_outer, n_inner], y is [n_outer].
// The reduced axis has no counterpart in the output, and that resolves
// correctly without any new vocabulary -- an axis with no binding keeps its
// full extent, which here is the truth rather than a fallback: one output row
// needs its whole input row. Seeding a 4-row output tile gives x a tile of
// 4 x n_inner.
//
// `BLOCK_INNER` is deliberately NOT bound to the inner axis. It is the load
// width that covers the whole row under a mask, not a tiling of it; binding it
// would claim the axis is chunked when it is read in one piece. Same reasoning
// as GEMM's BLOCK_K (teenygrad-1tl.10).
//
// `block = 1` on the outer axis records the kernel's own granularity -- one
// row per program, `row = program_id(Axis::X)`, and there is no BLOCK_OUTER to
// name. It is documentation: `resolve_inputs` takes the block from the
// propagated output tile, so a multi-row tile still resolves correctly.
pub fn reduce_prod_forward<T: Triton, D: Float, const BLOCK_INNER: i32>(
    #[tile(block = 1, extent = n_outer)]
    #[tile(block = BLOCK_INNER, extent = n_inner, reduce, fill = one)]
    x: In<Tile<T, D>>,
    #[tile(block = 1, extent = n_outer)] y: Out<Tile<T, D>>,
    n_inner: i32,
    n_outer: i32,
) where
    T::I32Tensor: types::Tensor<i32, 1>,
    T::I32Tensor: Comparison<i32, BoolTensor = T::BoolTensor>,
    T::Pointer<D>: AddOffsets<i32, 1, T::I32Tensor, Output = T::Tensor<T::Pointer<D>>>,
{
    T::store(
        y.tensor,
        T::exp(T::sum(T::log(x.tensor), Some(0), true)),
        None,
        &[],
        None,
        None,
    );
}

impl_reduce_float_runtime_op!(ReduceProdForward);

// ── CumSum ────────────────────────────────────────────────────────────────────

/// Forward: y = cumsum(x, axis=0) over a 1-D block
/// Each CTA handles one complete row (n_inner elements).
#[tiled_kernel]
// teenygrad-1tl.9. Rank-*preserving* but sequentially dependent: every output
// element depends on all preceding ones along the scan axis, so the inner axis
// cannot be tiled -- and it is untiled on both sides here, which is what the
// spec says by binding neither. The kernel loads the whole row, scans it and
// stores the whole row.
//
// So this is declarable rather than the recorded route-3 reason the rung
// expected. Note what the spec does not carry: "untiled" says the axis is not
// tiled, not *why*. The reason is the prefix dependency, and it lives here.
//
// No `reduce` flag -- nothing is reduced; the output has the axis.
pub fn cum_sum_forward<T: Triton, D: Num, const BLOCK_INNER: i32>(
    #[tile(block = 1, extent = n_outer)]
    #[tile(extent = n_inner)]
    x_ptr: In<T::Pointer<D>>,
    #[tile(block = 1, extent = n_outer)]
    #[tile(extent = n_inner)]
    y_ptr: Out<T::Pointer<D>>,
    n_inner: i32,
    n_outer: i32,
) where
    T::I32Tensor: types::Tensor<i32, 1>,
    T::I32Tensor: Comparison<i32, BoolTensor = T::BoolTensor>,
    T::Pointer<D>: AddOffsets<i32, 1, T::I32Tensor, Output = T::Tensor<T::Pointer<D>>>,
{
    let row = T::program_id(Axis::X);
    if row >= n_outer {
        return;
    }
    let col_offsets = T::arange(0, BLOCK_INNER);
    let offsets = col_offsets + row * n_inner;
    let mask = col_offsets.lt(n_inner);
    let x = T::load(
        x_ptr.add_offsets(offsets),
        Some(mask),
        Some(T::zeros::<D>(&[BLOCK_INNER])),
        &[],
        None,
        None,
        None,
        false,
    );
    // Use Triton's cumsum: axis=0 over the 1-D block, not reversed.
    let y = T::cumsum(x, 0, false);
    T::store(y_ptr.add_offsets(offsets), y, Some(mask), &[], None, None);
}

impl<D: Num + Send + Sync + 'static> teeny_core::model::RuntimeOp for CumSumForward<D> {
    fn n_activation_inputs(&self) -> usize {
        1
    }
    fn param_shapes(&self, _: &[&[usize]], _: &[usize]) -> Vec<Vec<usize>> {
        vec![]
    }
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
    fn grid(&self, output_shape: &[usize]) -> [u32; 3] {
        let n_total: usize = output_shape.iter().product();
        let n_inner = output_shape.last().copied().unwrap_or(1);
        let n_outer = n_total / n_inner;
        [n_outer as u32, 1, 1]
    }
}

// ── CumProd ───────────────────────────────────────────────────────────────────

/// Forward: y = cumprod(x, axis=0) over a 1-D block
#[tiled_kernel]
// teenygrad-1tl.9. Rank-*preserving* but sequentially dependent: every output
// element depends on all preceding ones along the scan axis, so the inner axis
// cannot be tiled -- and it is untiled on both sides here, which is what the
// spec says by binding neither. The kernel loads the whole row, scans it and
// stores the whole row.
//
// So this is declarable rather than the recorded route-3 reason the rung
// expected. Note what the spec does not carry: "untiled" says the axis is not
// tiled, not *why*. The reason is the prefix dependency, and it lives here.
//
// No `reduce` flag -- nothing is reduced; the output has the axis.
pub fn cum_prod_forward<T: Triton, D: Num, const BLOCK_INNER: i32>(
    #[tile(block = 1, extent = n_outer)]
    #[tile(extent = n_inner)]
    x_ptr: In<T::Pointer<D>>,
    #[tile(block = 1, extent = n_outer)]
    #[tile(extent = n_inner)]
    y_ptr: Out<T::Pointer<D>>,
    n_inner: i32,
    n_outer: i32,
) where
    T::I32Tensor: types::Tensor<i32, 1>,
    T::I32Tensor: Comparison<i32, BoolTensor = T::BoolTensor>,
    T::Pointer<D>: AddOffsets<i32, 1, T::I32Tensor, Output = T::Tensor<T::Pointer<D>>>,
{
    let row = T::program_id(Axis::X);
    if row >= n_outer {
        return;
    }
    let col_offsets = T::arange(0, BLOCK_INNER);
    let offsets = col_offsets + row * n_inner;
    let mask = col_offsets.lt(n_inner);
    let x = T::load(
        x_ptr.add_offsets(offsets),
        Some(mask),
        Some(T::zeros::<D>(&[BLOCK_INNER])),
        &[],
        None,
        None,
        None,
        false,
    );
    let y = T::cumprod(x, 0, false);
    T::store(y_ptr.add_offsets(offsets), y, Some(mask), &[], None, None);
}

impl<D: Num + Send + Sync + 'static> teeny_core::model::RuntimeOp for CumProdForward<D> {
    fn n_activation_inputs(&self) -> usize {
        1
    }
    fn param_shapes(&self, _: &[&[usize]], _: &[usize]) -> Vec<Vec<usize>> {
        vec![]
    }
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
#[tiled_kernel]
// teenygrad-1tl.9. Rank-reducing: x is [n_outer, n_inner], y is [n_outer].
// The reduced axis has no counterpart in the output, and that resolves
// correctly without any new vocabulary -- an axis with no binding keeps its
// full extent, which here is the truth rather than a fallback: one output row
// needs its whole input row. Seeding a 4-row output tile gives x a tile of
// 4 x n_inner.
//
// `BLOCK_INNER` is deliberately NOT bound to the inner axis. It is the load
// width that covers the whole row under a mask, not a tiling of it; binding it
// would claim the axis is chunked when it is read in one piece. Same reasoning
// as GEMM's BLOCK_K (teenygrad-1tl.10).
//
// `block = 1` on the outer axis records the kernel's own granularity -- one
// row per program, `row = program_id(Axis::X)`, and there is no BLOCK_OUTER to
// name. It is documentation: `resolve_inputs` takes the block from the
// propagated output tile, so a multi-row tile still resolves correctly.
pub fn global_avg_pool_forward<T: Triton, D: Float, const BLOCK_INNER: i32>(
    #[tile(block = 1, extent = n_outer)]
    #[tile(block = BLOCK_INNER, extent = n_inner, reduce)]
    x: In<Tile<T, D>>,
    #[tile(block = 1, extent = n_outer)] y: Out<Tile<T, D>>,
    n_inner: i32,
    n_outer: i32,
) where
    T::I32Tensor: types::Tensor<i32, 1>,
    T::I32Tensor: Comparison<i32, BoolTensor = T::BoolTensor>,
    T::Pointer<D>: AddOffsets<i32, 1, T::I32Tensor, Output = T::Tensor<T::Pointer<D>>>,
{
    T::store(
        y.tensor,
        T::sum(x.tensor, Some(0), true)
            / T::cast::<i32, D>(T::full::<i32>(&[1], n_inner), None, false),
        None,
        &[],
        None,
        None,
    );
}

impl_reduce_float_runtime_op!(GlobalAvgPoolForward);

// ── GlobalMaxPool ─────────────────────────────────────────────────────────────

/// Forward: y[row] = max(x[row, :])  (same as ReduceMax)
#[tiled_kernel]
// teenygrad-1tl.9. Rank-reducing: x is [n_outer, n_inner], y is [n_outer].
// The reduced axis has no counterpart in the output, and that resolves
// correctly without any new vocabulary -- an axis with no binding keeps its
// full extent, which here is the truth rather than a fallback: one output row
// needs its whole input row. Seeding a 4-row output tile gives x a tile of
// 4 x n_inner.
//
// `BLOCK_INNER` is deliberately NOT bound to the inner axis. It is the load
// width that covers the whole row under a mask, not a tiling of it; binding it
// would claim the axis is chunked when it is read in one piece. Same reasoning
// as GEMM's BLOCK_K (teenygrad-1tl.10).
//
// `block = 1` on the outer axis records the kernel's own granularity -- one
// row per program, `row = program_id(Axis::X)`, and there is no BLOCK_OUTER to
// name. It is documentation: `resolve_inputs` takes the block from the
// propagated output tile, so a multi-row tile still resolves correctly.
pub fn global_max_pool_forward<T: Triton, D: Float, const BLOCK_INNER: i32>(
    #[tile(block = 1, extent = n_outer)]
    #[tile(block = BLOCK_INNER, extent = n_inner, reduce, fill = neg_inf)]
    x: In<Tile<T, D>>,
    #[tile(block = 1, extent = n_outer)] y: Out<Tile<T, D>>,
    n_inner: i32,
    n_outer: i32,
) where
    T::I32Tensor: types::Tensor<i32, 1>,
    T::I32Tensor: Comparison<i32, BoolTensor = T::BoolTensor>,
    T::Pointer<D>: AddOffsets<i32, 1, T::I32Tensor, Output = T::Tensor<T::Pointer<D>>>,
{
    T::store(
        y.tensor,
        T::max(x.tensor, Some(0), true),
        None,
        &[],
        None,
        None,
    );
}

impl_reduce_float_runtime_op!(GlobalMaxPoolForward);
