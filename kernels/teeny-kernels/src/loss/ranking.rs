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

#![allow(non_snake_case)]

use teeny_macros::kernel;
use teeny_triton::triton::{
    types::{AddOffsets, Comparison},
    *,
};

// ── MarginRankingLoss ─────────────────────────────────────────────────────────

/// Margin ranking loss forward (element-wise).
///
/// `out[i] = max(0, -y[i] * (x1[i] - x2[i]) + margin)`
///
/// Grid: `[ceil(n / BLOCK_SIZE), 1, 1]`
#[kernel]
pub fn margin_ranking_loss_forward<T: Triton, const BLOCK_SIZE: i32>(
    x1_ptr: T::Pointer<f32>,
    x2_ptr: T::Pointer<f32>,
    y_ptr: T::Pointer<f32>,
    out_ptr: T::Pointer<f32>,
    n_elements: i32,
    margin: f32,
) where
    T::I32Tensor: types::Tensor<i32, 1>,
    T::I32Tensor: Comparison<i32, BoolTensor = T::BoolTensor>,
    T::Pointer<f32>: AddOffsets<i32, 1, T::I32Tensor, Output = T::Tensor<T::Pointer<f32>>>,
{
    let pid = T::program_id(Axis::X);
    let block_start = pid * BLOCK_SIZE;
    let offsets = T::arange(0, BLOCK_SIZE) + block_start;
    let in_bounds = offsets.lt(n_elements);
    let zeros = T::zeros::<f32>(&[BLOCK_SIZE]);

    let x1 = T::load(x1_ptr.add_offsets(offsets), Some(in_bounds), Some(zeros), &[], None, None, None, false);
    let x2 = T::load(x2_ptr.add_offsets(offsets), Some(in_bounds), Some(zeros), &[], None, None, None, false);
    let y  = T::load(y_ptr.add_offsets(offsets),  Some(in_bounds), Some(zeros), &[], None, None, None, false);

    let margin_t = T::full(&[BLOCK_SIZE], margin);
    let neg_one  = T::full(&[BLOCK_SIZE], -1.0_f32);
    // hinge = max(0, -y*(x1-x2) + margin)
    let hinge = T::maximum(neg_one * y * (x1 - x2) + margin_t, zeros);
    T::store(out_ptr.add_offsets(offsets), hinge, Some(in_bounds), &[], None, None);
}

/// Margin ranking loss backward (element-wise).
///
/// `dx1[i] = -y[i] * dy[i]` if hinge > 0, else 0.
/// `dx2[i] =  y[i] * dy[i]` if hinge > 0, else 0.
#[kernel]
pub fn margin_ranking_loss_backward<T: Triton, const BLOCK_SIZE: i32>(
    dy_ptr: T::Pointer<f32>,
    x1_ptr: T::Pointer<f32>,
    x2_ptr: T::Pointer<f32>,
    y_ptr: T::Pointer<f32>,
    dx1_ptr: T::Pointer<f32>,
    dx2_ptr: T::Pointer<f32>,
    n_elements: i32,
    margin: f32,
) where
    T::I32Tensor: types::Tensor<i32, 1>,
    T::I32Tensor: Comparison<i32, BoolTensor = T::BoolTensor>,
    T::Pointer<f32>: AddOffsets<i32, 1, T::I32Tensor, Output = T::Tensor<T::Pointer<f32>>>,
{
    let pid = T::program_id(Axis::X);
    let block_start = pid * BLOCK_SIZE;
    let offsets = T::arange(0, BLOCK_SIZE) + block_start;
    let in_bounds = offsets.lt(n_elements);
    let zeros = T::zeros::<f32>(&[BLOCK_SIZE]);

    let dy = T::load(dy_ptr.add_offsets(offsets), Some(in_bounds), Some(zeros), &[], None, None, None, false);
    let x1 = T::load(x1_ptr.add_offsets(offsets), Some(in_bounds), Some(zeros), &[], None, None, None, false);
    let x2 = T::load(x2_ptr.add_offsets(offsets), Some(in_bounds), Some(zeros), &[], None, None, None, false);
    let y  = T::load(y_ptr.add_offsets(offsets),  Some(in_bounds), Some(zeros), &[], None, None, None, false);

    let margin_t = T::full(&[BLOCK_SIZE], margin);
    let neg_one  = T::full(&[BLOCK_SIZE], -1.0_f32);
    let pre_hinge = neg_one * y * (x1 - x2) + margin_t;
    let active = T::gt(pre_hinge, zeros);

    let dx1 = T::where_(active, neg_one * y * dy, zeros);
    let dx2 = T::where_(active, y * dy, zeros);
    T::store(dx1_ptr.add_offsets(offsets), dx1, Some(in_bounds), &[], None, None);
    T::store(dx2_ptr.add_offsets(offsets), dx2, Some(in_bounds), &[], None, None);
}

// ── HingeEmbeddingLoss ────────────────────────────────────────────────────────

/// Hinge embedding loss forward (element-wise).
///
/// ```text
/// out[i] = x[i]                     if y[i] ==  1
///        = max(0, margin - x[i])    if y[i] == -1
/// ```
///
/// Grid: `[ceil(n / BLOCK_SIZE), 1, 1]`
#[kernel]
pub fn hinge_embedding_loss_forward<T: Triton, const BLOCK_SIZE: i32>(
    inp_ptr: T::Pointer<f32>,
    y_ptr: T::Pointer<f32>,
    out_ptr: T::Pointer<f32>,
    n_elements: i32,
    margin: f32,
) where
    T::I32Tensor: types::Tensor<i32, 1>,
    T::I32Tensor: Comparison<i32, BoolTensor = T::BoolTensor>,
    T::Pointer<f32>: AddOffsets<i32, 1, T::I32Tensor, Output = T::Tensor<T::Pointer<f32>>>,
{
    let pid = T::program_id(Axis::X);
    let block_start = pid * BLOCK_SIZE;
    let offsets = T::arange(0, BLOCK_SIZE) + block_start;
    let in_bounds = offsets.lt(n_elements);
    let zeros = T::zeros::<f32>(&[BLOCK_SIZE]);

    let inp = T::load(inp_ptr.add_offsets(offsets), Some(in_bounds), Some(zeros), &[], None, None, None, false);
    let y   = T::load(y_ptr.add_offsets(offsets),   Some(in_bounds), Some(zeros), &[], None, None, None, false);

    let margin_t = T::full(&[BLOCK_SIZE], margin);
    // y > 0 means y == 1
    let y_is_pos = T::gt(y, zeros);
    let hinge = T::maximum(margin_t - inp, zeros);
    let out = T::where_(y_is_pos, inp, hinge);
    T::store(out_ptr.add_offsets(offsets), out, Some(in_bounds), &[], None, None);
}

/// Hinge embedding loss backward (element-wise).
///
/// ```text
/// dx[i] = dy[i]     if y[i] ==  1
///       = -dy[i]    if y[i] == -1 and margin - x[i] > 0
///       = 0         otherwise
/// ```
#[kernel]
pub fn hinge_embedding_loss_backward<T: Triton, const BLOCK_SIZE: i32>(
    dy_ptr: T::Pointer<f32>,
    inp_ptr: T::Pointer<f32>,
    y_ptr: T::Pointer<f32>,
    dx_ptr: T::Pointer<f32>,
    n_elements: i32,
    margin: f32,
) where
    T::I32Tensor: types::Tensor<i32, 1>,
    T::I32Tensor: Comparison<i32, BoolTensor = T::BoolTensor>,
    T::Pointer<f32>: AddOffsets<i32, 1, T::I32Tensor, Output = T::Tensor<T::Pointer<f32>>>,
{
    let pid = T::program_id(Axis::X);
    let block_start = pid * BLOCK_SIZE;
    let offsets = T::arange(0, BLOCK_SIZE) + block_start;
    let in_bounds = offsets.lt(n_elements);
    let zeros = T::zeros::<f32>(&[BLOCK_SIZE]);

    let dy  = T::load(dy_ptr.add_offsets(offsets),  Some(in_bounds), Some(zeros), &[], None, None, None, false);
    let inp = T::load(inp_ptr.add_offsets(offsets),  Some(in_bounds), Some(zeros), &[], None, None, None, false);
    let y   = T::load(y_ptr.add_offsets(offsets),    Some(in_bounds), Some(zeros), &[], None, None, None, false);

    let margin_t = T::full(&[BLOCK_SIZE], margin);
    let neg_one  = T::full(&[BLOCK_SIZE], -1.0_f32);
    let y_is_pos = T::gt(y, zeros);
    // Active for y == -1: margin - x > 0
    let neg_active = T::gt(margin_t - inp, zeros);
    let dx_neg = T::where_(neg_active, neg_one * dy, zeros);
    let dx = T::where_(y_is_pos, dy, dx_neg);
    T::store(dx_ptr.add_offsets(offsets), dx, Some(in_bounds), &[], None, None);
}

// ── MultiMarginLoss ───────────────────────────────────────────────────────────

/// Multi-margin loss forward (per-row).
///
/// `out[n] = (1/n_cols) * sum_{j != target[n]} max(0, margin - x[n,target] + x[n,j])`
///
/// Computed as:
/// `(sum_all max(0, margin - x_t + x_j) - max(0, margin)) / n_cols`
///
/// Grid: `[n_rows, 1, 1]` — one CTA per row.
/// `BLOCK_SIZE` must equal `next_power_of_two(n_cols)`.
#[kernel]
pub fn multi_margin_loss_forward<T: Triton, const BLOCK_SIZE: i32>(
    input_ptr: T::Pointer<f32>,
    targets_ptr: T::Pointer<i32>,
    out_ptr: T::Pointer<f32>,
    _n_rows: i32,
    n_cols: i32,
    margin: f32,
) where
    T::I32Tensor: types::Tensor<i32, 1>,
    T::I32Tensor: Comparison<i32, BoolTensor = T::BoolTensor>,
    T::Pointer<f32>: AddOffsets<i32, 1, T::I32Tensor, Output = T::Tensor<T::Pointer<f32>>>,
    T::Pointer<i32>: AddOffsets<i32, 1, T::I32Tensor, Output = T::Tensor<T::Pointer<i32>>>,
    T::Tensor<i32>: types::Tensor<i32, 1>,
    T::Pointer<f32>: AddOffsets<i32, 1, T::Tensor<i32>, Output = T::Tensor<T::Pointer<f32>>>,
{
    let pid = T::program_id(Axis::X);
    let row_base = pid * n_cols;
    let col_offs: T::I32Tensor = T::arange(0, BLOCK_SIZE);
    let row_offs: T::I32Tensor = col_offs + row_base;
    let in_row = col_offs.lt(n_cols);
    let zeros = T::zeros::<f32>(&[BLOCK_SIZE]);

    let row = T::load(
        input_ptr.add_offsets(row_offs),
        Some(in_row), Some(zeros), &[], None, None, None, false,
    );

    // Load target index
    let tgt_off: T::I32Tensor = T::arange(0, 1) + pid;
    let tgt: T::Tensor<i32> = T::load(
        targets_ptr.add_offsets(tgt_off),
        None, None, &[], None, None, None, false,
    );

    // Load x[target] using flat index
    let base: T::Tensor<i32> = T::full::<i32>(&[1], row_base);
    let flat_off: T::Tensor<i32> = base + tgt;
    let x_t: T::Tensor<f32> = T::load(
        input_ptr.add_offsets(flat_off),
        None, None, &[], None, None, None, false,
    );

    // hinge for all j (including t): max(0, margin - x_t + x_j)
    let margin_t = T::full(&[BLOCK_SIZE], margin);
    let x_t_bcast = T::broadcast_to(x_t, &[BLOCK_SIZE]);
    let hinge_all = T::maximum(margin_t - x_t_bcast + row, zeros);

    // sum all hinges, subtract the target's own contribution: max(0, margin)
    let sum_all = T::sum(hinge_all, Some(0), true); // shape [1]
    // max(0, margin) as a tensor to avoid scalar f32 comparison in kernel context
    let tgt_contrib = T::maximum(T::full::<f32>(&[1], margin), T::zeros::<f32>(&[1]));
    // Cast n_cols to f32 via tensor cast (scalar `as f32` produces ub.poison in kernels)
    let n_cols_f = T::cast::<i32, f32>(T::full::<i32>(&[1], n_cols), None, false);
    let loss = (sum_all - tgt_contrib) / n_cols_f;

    let out_off: T::I32Tensor = T::arange(0, 1) + pid;
    T::store(out_ptr.add_offsets(out_off), loss, None, &[], None, None);
}

/// Multi-margin loss backward (per-row).
///
/// Grid: `[n_rows, 1, 1]`.  `BLOCK_SIZE` must equal `next_power_of_two(n_cols)`.
///
/// Two-step write:
/// 1. Store `(dy / n_cols) * active_j` for all j in the row.
/// 2. Atomic-add correction at target: `-(dy / n_cols) * sum(active_all)`.
#[kernel]
pub fn multi_margin_loss_backward<T: Triton, const BLOCK_SIZE: i32>(
    dy_ptr: T::Pointer<f32>,
    input_ptr: T::Pointer<f32>,
    targets_ptr: T::Pointer<i32>,
    dx_ptr: T::Pointer<f32>,
    _n_rows: i32,
    n_cols: i32,
    margin: f32,
) where
    T::I32Tensor: types::Tensor<i32, 1>,
    T::I32Tensor: Comparison<i32, BoolTensor = T::BoolTensor>,
    T::Pointer<f32>: AddOffsets<i32, 1, T::I32Tensor, Output = T::Tensor<T::Pointer<f32>>>,
    T::Pointer<i32>: AddOffsets<i32, 1, T::I32Tensor, Output = T::Tensor<T::Pointer<i32>>>,
    T::Tensor<i32>: types::Tensor<i32, 1>,
    T::Pointer<f32>: AddOffsets<i32, 1, T::Tensor<i32>, Output = T::Tensor<T::Pointer<f32>>>,
{
    let pid = T::program_id(Axis::X);
    let row_base = pid * n_cols;
    let col_offs: T::I32Tensor = T::arange(0, BLOCK_SIZE);
    let row_offs: T::I32Tensor = col_offs + row_base;
    let in_row = col_offs.lt(n_cols);
    let zeros = T::zeros::<f32>(&[BLOCK_SIZE]);

    // Load upstream gradient
    let dy_off: T::I32Tensor = T::arange(0, 1) + pid;
    let dy: T::Tensor<f32> = T::load(
        dy_ptr.add_offsets(dy_off),
        None, None, &[], None, None, None, false,
    );

    let row = T::load(
        input_ptr.add_offsets(row_offs),
        Some(in_row), Some(zeros), &[], None, None, None, false,
    );

    // Load target index
    let tgt_off: T::I32Tensor = T::arange(0, 1) + pid;
    let tgt: T::Tensor<i32> = T::load(
        targets_ptr.add_offsets(tgt_off),
        None, None, &[], None, None, None, false,
    );

    // Load x[target] using flat index
    let base: T::Tensor<i32> = T::full::<i32>(&[1], row_base);
    let flat_off: T::Tensor<i32> = base + tgt;
    let x_t: T::Tensor<f32> = T::load(
        input_ptr.add_offsets(flat_off),
        None, None, &[], None, None, None, false,
    );

    let margin_t = T::full(&[BLOCK_SIZE], margin);
    let x_t_bcast = T::broadcast_to(x_t, &[BLOCK_SIZE]);
    let ones = T::full(&[BLOCK_SIZE], 1.0_f32);

    // active[j] = (margin - x_t + x_j > 0)
    let active = T::gt(margin_t - x_t_bcast + row, zeros);
    let active_f = T::where_(active, ones, zeros);

    // sum of all active (including target position)
    let sum_active = T::sum(active_f, Some(0), true); // shape [1]

    let n_cols_f = T::cast::<i32, f32>(T::full::<i32>(&[1], n_cols), None, false);
    let dy_over_n = dy / n_cols_f;

    // Step 1: store (dy/n_cols) * active_f for the whole row
    let dy_bcast = T::broadcast_to(dy_over_n, &[BLOCK_SIZE]);
    let dx_row = dy_bcast * active_f;
    T::store(
        dx_ptr.add_offsets(row_offs),
        dx_row,
        Some(in_row), &[], None, None,
    );

    // Step 2: atomic_add correction at target to fix target position
    // After step 1, position t has (dy/n_cols)*active_t; correct value is -(dy/n_cols)*count_not_t
    // correction = -(dy/n_cols)*sum_active_all
    let neg_one = T::full(&[1], -1.0_f32);
    let correction = neg_one * dy_over_n * sum_active;
    T::atomic_add(
        dx_ptr.add_offsets(flat_off),
        correction,
        None, None, None,
    );
}
