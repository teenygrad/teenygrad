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

// ── NLLLoss ───────────────────────────────────────────────────────────────────

/// NLL loss forward: `out[n] = -log_prob[n, target[n]]`.
///
/// Grid: `[n_rows, 1, 1]` — one CTA per batch element.
///
/// The flat index `pid * n_cols + target` is computed entirely in
/// `T::Tensor<i32>` space (from T::full + T::load arithmetic), avoiding any
/// conversion to `T::I32Tensor`.  A separate `AddOffsets` bound on
/// `T::Pointer<_>` for `T::Tensor<i32>` covers the indexed load and store.
#[kernel]
pub fn nll_loss_forward<T: Triton>(
    log_probs_ptr: T::Pointer<f32>,
    targets_ptr: T::Pointer<i32>,
    out_ptr: T::Pointer<f32>,
    _n_rows: i32,
    n_cols: i32,
) where
    T::I32Tensor: types::Tensor<i32, 1>,
    T::I32Tensor: Comparison<i32, BoolTensor = T::BoolTensor>,
    // For loading targets (offset = T::I32Tensor from arange)
    T::Pointer<i32>: AddOffsets<i32, 1, T::I32Tensor, Output = T::Tensor<T::Pointer<i32>>>,
    // For storing output (offset = T::I32Tensor from arange)
    T::Pointer<f32>: AddOffsets<i32, 1, T::I32Tensor, Output = T::Tensor<T::Pointer<f32>>>,
    // For indexed load of log_prob (offset = T::Tensor<i32> from full + load arithmetic)
    T::Tensor<i32>: types::Tensor<i32, 1>,
    T::Pointer<f32>: AddOffsets<i32, 1, T::Tensor<i32>, Output = T::Tensor<T::Pointer<f32>>>,
{
    let pid = T::program_id(Axis::X);

    // Load target class for this row (unmasked — pid is always a valid row index)
    let tgt_off: T::I32Tensor = T::arange(0, 1) + pid;
    let tgt: T::Tensor<i32> = T::load(
        targets_ptr.add_offsets(tgt_off),
        None, None, &[], None, None, None, false,
    );

    // flat index: pid * n_cols + target — purely in T::Tensor<i32> space
    let base: T::Tensor<i32> = T::full::<i32>(&[1], pid * n_cols);
    let flat_off: T::Tensor<i32> = base + tgt;

    // Load log_prob at target class (unmasked — index is always valid)
    let lp: T::Tensor<f32> = T::load(
        log_probs_ptr.add_offsets(flat_off),
        None, None, &[], None, None, None, false,
    );

    let loss = T::full(&[1], -1.0_f32) * lp;

    let out_off: T::I32Tensor = T::arange(0, 1) + pid;
    T::store(out_ptr.add_offsets(out_off), loss, None, &[], None, None);
}

/// NLL loss backward: `dx[n, target[n]] = -dy[n]`, zero elsewhere.
///
/// Grid: `[n_rows, 1, 1]` — one CTA per batch element.
/// The dx buffer must be zero-initialised before launch.
#[kernel]
pub fn nll_loss_backward<T: Triton>(
    dy_ptr: T::Pointer<f32>,
    targets_ptr: T::Pointer<i32>,
    dx_ptr: T::Pointer<f32>,
    _n_rows: i32,
    n_cols: i32,
) where
    T::I32Tensor: types::Tensor<i32, 1>,
    T::I32Tensor: Comparison<i32, BoolTensor = T::BoolTensor>,
    T::Pointer<f32>: AddOffsets<i32, 1, T::I32Tensor, Output = T::Tensor<T::Pointer<f32>>>,
    T::Pointer<i32>: AddOffsets<i32, 1, T::I32Tensor, Output = T::Tensor<T::Pointer<i32>>>,
    T::Tensor<i32>: types::Tensor<i32, 1>,
    T::Pointer<f32>: AddOffsets<i32, 1, T::Tensor<i32>, Output = T::Tensor<T::Pointer<f32>>>,
{
    let pid = T::program_id(Axis::X);

    let dy_off: T::I32Tensor = T::arange(0, 1) + pid;
    let dy: T::Tensor<f32> = T::load(
        dy_ptr.add_offsets(dy_off),
        None, None, &[], None, None, None, false,
    );

    let tgt_off: T::I32Tensor = T::arange(0, 1) + pid;
    let tgt: T::Tensor<i32> = T::load(
        targets_ptr.add_offsets(tgt_off),
        None, None, &[], None, None, None, false,
    );

    // flat index: pid * n_cols + target
    let base: T::Tensor<i32> = T::full::<i32>(&[1], pid * n_cols);
    let flat_off: T::Tensor<i32> = base + tgt;

    let neg_dy = T::full(&[1], -1.0_f32) * dy;
    T::store(dx_ptr.add_offsets(flat_off), neg_dy, None, &[], None, None);
}

// ── CrossEntropyLoss ──────────────────────────────────────────────────────────

/// Cross-entropy loss forward: `out[n] = log(sum_c exp(x[n,c])) - x[n, target[n]]`.
///
/// Numerically stable: subtracts row-max before exp (log-sum-exp trick).
///
/// Grid: `[n_rows, 1, 1]` — one CTA per row.
/// `BLOCK_SIZE` must equal `next_power_of_two(n_cols)`.
#[kernel]
pub fn cross_entropy_loss_forward<T: Triton, const BLOCK_SIZE: i32>(
    input_ptr: T::Pointer<f32>,
    targets_ptr: T::Pointer<i32>,
    out_ptr: T::Pointer<f32>,
    _n_rows: i32,
    n_cols: i32,
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
    // Large finite negative used as "neg-inf" mask: exp underflows to 0
    let neg_inf = T::full(&[BLOCK_SIZE], -3.4028235e38_f32);

    let row = T::load(
        input_ptr.add_offsets(row_offs),
        Some(in_row), Some(neg_inf), &[], None, None, None, false,
    );

    // log-sum-exp: subtract row max for numerical stability
    let row_max = T::max(row, Some(0), true);     // shape [1]
    let row_shifted = row - row_max;              // broadcast
    let exp_row = T::exp(row_shifted);
    let sum_exp = T::sum(exp_row, Some(0), true); // shape [1]
    let log_sum_exp = T::log(sum_exp) + row_max;  // shape [1]

    // Load target index (unmasked — always in-bounds)
    let tgt_off: T::I32Tensor = T::arange(0, 1) + pid;
    let tgt: T::Tensor<i32> = T::load(
        targets_ptr.add_offsets(tgt_off),
        None, None, &[], None, None, None, false,
    );

    // flat index for x[target]: pid * n_cols + target
    let base: T::Tensor<i32> = T::full::<i32>(&[1], row_base);
    let flat_off: T::Tensor<i32> = base + tgt;
    let x_target: T::Tensor<f32> = T::load(
        input_ptr.add_offsets(flat_off),
        None, None, &[], None, None, None, false,
    );

    // CE = log_sum_exp - x[target]
    let loss = log_sum_exp - x_target;

    let out_off: T::I32Tensor = T::arange(0, 1) + pid;
    T::store(out_ptr.add_offsets(out_off), loss, None, &[], None, None);
}

/// Cross-entropy loss backward.
///
/// `dx[n, c] = dy[n] * (softmax(x[n])[c] - indicator(c == target[n]))`
///
/// Grid: `[n_rows, 1, 1]` — one CTA per row.
/// `BLOCK_SIZE` must equal `next_power_of_two(n_cols)`.
#[kernel]
pub fn cross_entropy_loss_backward<T: Triton, const BLOCK_SIZE: i32>(
    dy_ptr: T::Pointer<f32>,
    input_ptr: T::Pointer<f32>,
    targets_ptr: T::Pointer<i32>,
    dx_ptr: T::Pointer<f32>,
    _n_rows: i32,
    n_cols: i32,
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

    // Load upstream gradient (unmasked — always in-bounds)
    let dy_off: T::I32Tensor = T::arange(0, 1) + pid;
    let dy: T::Tensor<f32> = T::load(
        dy_ptr.add_offsets(dy_off),
        None, None, &[], None, None, None, false,
    );

    // Load the row and compute softmax
    let neg_inf = T::full(&[BLOCK_SIZE], -3.4028235e38_f32);
    let row = T::load(
        input_ptr.add_offsets(row_offs),
        Some(in_row), Some(neg_inf), &[], None, None, None, false,
    );
    let sm = T::softmax(row, None, false, false);

    // Load target index (unmasked — always in-bounds)
    let tgt_off: T::I32Tensor = T::arange(0, 1) + pid;
    let tgt: T::Tensor<i32> = T::load(
        targets_ptr.add_offsets(tgt_off),
        None, None, &[], None, None, None, false,
    );

    // Step 1: store dy * softmax(x) to the full row (uses T::I32Tensor offsets)
    let dy_bcast = T::broadcast_to(dy, &[BLOCK_SIZE]);
    let dx_row = dy_bcast * sm;
    T::store(
        dx_ptr.add_offsets(row_offs),
        dx_row,
        Some(in_row), &[], None, None,
    );

    // Step 2: subtract dy at target position via atomic_add(-dy)
    // flat index: pid * n_cols + target (T::Tensor<i32> space)
    let base: T::Tensor<i32> = T::full::<i32>(&[1], row_base);
    let flat_off: T::Tensor<i32> = base + tgt;
    let neg_dy = T::full(&[1], -1.0_f32) * dy;
    T::atomic_add(
        dx_ptr.add_offsets(flat_off),
        neg_dy,
        None, None, None,
    );
}

// ── MultiLabelSoftMarginLoss ──────────────────────────────────────────────────

/// Multi-label soft-margin loss forward (element-wise).
///
/// Identical to `BCEWithLogitsLoss` per element:
/// ```text
/// out = max(x, 0) - x*y + log(1 + exp(-|x|))
/// ```
///
/// Grid: `[ceil(n / BLOCK_SIZE), 1, 1]`, block `[128, 1, 1]`.
#[kernel]
pub fn multilabel_soft_margin_loss_forward<T: Triton, const BLOCK_SIZE: i32>(
    input_ptr: T::Pointer<f32>,
    target_ptr: T::Pointer<f32>,
    out_ptr: T::Pointer<f32>,
    n_elements: i32,
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

    let inp = T::load(input_ptr.add_offsets(offsets),  Some(in_bounds), Some(zeros), &[], None, None, None, false);
    let tgt = T::load(target_ptr.add_offsets(offsets), Some(in_bounds), Some(zeros), &[], None, None, None, false);

    let one = T::full(&[BLOCK_SIZE], 1.0_f32);
    let neg_one = T::full(&[BLOCK_SIZE], -1.0_f32);
    // Numerically stable BCE-with-logits:  max(x,0) - x*t + log(1+exp(-|x|))
    let relu_x = T::maximum(inp, zeros);
    let neg_abs_x = neg_one * T::abs(inp);
    let loss = relu_x - inp * tgt + T::log(one + T::exp(neg_abs_x));
    T::store(out_ptr.add_offsets(offsets), loss, Some(in_bounds), &[], None, None);
}

/// Multi-label soft-margin loss backward (element-wise).
///
/// `dx = (sigmoid(x) - target) * dy`
#[kernel]
pub fn multilabel_soft_margin_loss_backward<T: Triton, const BLOCK_SIZE: i32>(
    dy_ptr: T::Pointer<f32>,
    input_ptr: T::Pointer<f32>,
    target_ptr: T::Pointer<f32>,
    dx_ptr: T::Pointer<f32>,
    n_elements: i32,
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

    let dy  = T::load(dy_ptr.add_offsets(offsets),     Some(in_bounds), Some(zeros), &[], None, None, None, false);
    let inp = T::load(input_ptr.add_offsets(offsets),  Some(in_bounds), Some(zeros), &[], None, None, None, false);
    let tgt = T::load(target_ptr.add_offsets(offsets), Some(in_bounds), Some(zeros), &[], None, None, None, false);

    let one = T::full(&[BLOCK_SIZE], 1.0_f32);
    let neg_one = T::full(&[BLOCK_SIZE], -1.0_f32);
    // sigmoid(x) manually to avoid __nv_sigmoidf
    let sig = one / (one + T::exp(neg_one * inp));
    let dx = (sig - tgt) * dy;
    T::store(dx_ptr.add_offsets(offsets), dx, Some(in_bounds), &[], None, None);
}
