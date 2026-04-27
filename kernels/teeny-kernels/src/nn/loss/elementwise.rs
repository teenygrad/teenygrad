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

// ── L1Loss ────────────────────────────────────────────────────────────────────

/// Element-wise L1 (MAE) loss forward: `out = |x - y|`.
///
/// Grid: `[ceil(n / BLOCK_SIZE), 1, 1]`, block `[128, 1, 1]`.
#[kernel]
pub fn l1_loss_forward<T: Triton, const BLOCK_SIZE: i32>(
    x_ptr: T::Pointer<f32>,
    y_ptr: T::Pointer<f32>,
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

    let x = T::load(x_ptr.add_offsets(offsets), Some(in_bounds), Some(zeros), &[], None, None, None, false);
    let y = T::load(y_ptr.add_offsets(offsets), Some(in_bounds), Some(zeros), &[], None, None, None, false);

    let loss = T::abs(x - y);
    T::store(out_ptr.add_offsets(offsets), loss, Some(in_bounds), &[], None, None);
}

/// Element-wise L1 backward: `dx = dy * sign(x - y)`.
///
/// `sign(0) = 0` by convention (no gradient at the kink).
#[kernel]
pub fn l1_loss_backward<T: Triton, const BLOCK_SIZE: i32>(
    dy_ptr: T::Pointer<f32>,
    x_ptr: T::Pointer<f32>,
    y_ptr: T::Pointer<f32>,
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

    let dy = T::load(dy_ptr.add_offsets(offsets), Some(in_bounds), Some(zeros), &[], None, None, None, false);
    let x  = T::load(x_ptr.add_offsets(offsets),  Some(in_bounds), Some(zeros), &[], None, None, None, false);
    let y  = T::load(y_ptr.add_offsets(offsets),  Some(in_bounds), Some(zeros), &[], None, None, None, false);

    let diff = x - y;
    let ones    = T::full(&[BLOCK_SIZE],  1.0_f32);
    let neg_one = T::full(&[BLOCK_SIZE], -1.0_f32);
    let pos = T::gt(diff, zeros);
    let neg = T::lt(diff, zeros);
    let sign = T::where_(pos, ones, T::where_(neg, neg_one, zeros));
    let dx = dy * sign;
    T::store(dx_ptr.add_offsets(offsets), dx, Some(in_bounds), &[], None, None);
}

// ── MSELoss ───────────────────────────────────────────────────────────────────

/// Element-wise MSE loss forward: `out = (x - y)^2`.
#[kernel]
pub fn mse_loss_forward<T: Triton, const BLOCK_SIZE: i32>(
    x_ptr: T::Pointer<f32>,
    y_ptr: T::Pointer<f32>,
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

    let x = T::load(x_ptr.add_offsets(offsets), Some(in_bounds), Some(zeros), &[], None, None, None, false);
    let y = T::load(y_ptr.add_offsets(offsets), Some(in_bounds), Some(zeros), &[], None, None, None, false);

    let diff = x - y;
    let loss = diff * diff;
    T::store(out_ptr.add_offsets(offsets), loss, Some(in_bounds), &[], None, None);
}

/// Element-wise MSE backward: `dx = 2 * (x - y) * dy`.
#[kernel]
pub fn mse_loss_backward<T: Triton, const BLOCK_SIZE: i32>(
    dy_ptr: T::Pointer<f32>,
    x_ptr: T::Pointer<f32>,
    y_ptr: T::Pointer<f32>,
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

    let dy = T::load(dy_ptr.add_offsets(offsets), Some(in_bounds), Some(zeros), &[], None, None, None, false);
    let x  = T::load(x_ptr.add_offsets(offsets),  Some(in_bounds), Some(zeros), &[], None, None, None, false);
    let y  = T::load(y_ptr.add_offsets(offsets),  Some(in_bounds), Some(zeros), &[], None, None, None, false);

    let two = T::full(&[BLOCK_SIZE], 2.0_f32);
    let dx = two * (x - y) * dy;
    T::store(dx_ptr.add_offsets(offsets), dx, Some(in_bounds), &[], None, None);
}

// ── HuberLoss ─────────────────────────────────────────────────────────────────

/// Element-wise Huber loss forward.
///
/// ```text
/// out = 0.5 * diff^2              if |diff| <= delta
///     = delta * (|diff| - 0.5*delta)   otherwise
/// ```
#[kernel]
pub fn huber_loss_forward<T: Triton, const BLOCK_SIZE: i32>(
    x_ptr: T::Pointer<f32>,
    y_ptr: T::Pointer<f32>,
    out_ptr: T::Pointer<f32>,
    n_elements: i32,
    delta: f32,
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

    let x = T::load(x_ptr.add_offsets(offsets), Some(in_bounds), Some(zeros), &[], None, None, None, false);
    let y = T::load(y_ptr.add_offsets(offsets), Some(in_bounds), Some(zeros), &[], None, None, None, false);

    let diff     = x - y;
    let abs_diff = T::abs(diff);
    let delta_t  = T::full(&[BLOCK_SIZE], delta);
    let half     = T::full(&[BLOCK_SIZE], 0.5_f32);

    // quadratic: 0.5 * diff^2
    let quad = half * diff * diff;
    // linear: delta * (|diff| - 0.5 * delta)
    let lin = delta_t * (abs_diff - half * delta_t);

    let in_quad = T::le(abs_diff, delta_t);
    let loss = T::where_(in_quad, quad, lin);
    T::store(out_ptr.add_offsets(offsets), loss, Some(in_bounds), &[], None, None);
}

/// Element-wise Huber loss backward.
///
/// ```text
/// dx = diff * dy            if |diff| <= delta
///    = delta * sign(diff) * dy    otherwise
/// ```
#[kernel]
pub fn huber_loss_backward<T: Triton, const BLOCK_SIZE: i32>(
    dy_ptr: T::Pointer<f32>,
    x_ptr: T::Pointer<f32>,
    y_ptr: T::Pointer<f32>,
    dx_ptr: T::Pointer<f32>,
    n_elements: i32,
    delta: f32,
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
    let x  = T::load(x_ptr.add_offsets(offsets),  Some(in_bounds), Some(zeros), &[], None, None, None, false);
    let y  = T::load(y_ptr.add_offsets(offsets),  Some(in_bounds), Some(zeros), &[], None, None, None, false);

    let diff     = x - y;
    let abs_diff = T::abs(diff);
    let delta_t  = T::full(&[BLOCK_SIZE], delta);
    let ones     = T::full(&[BLOCK_SIZE],  1.0_f32);
    let neg_one  = T::full(&[BLOCK_SIZE], -1.0_f32);

    let pos  = T::gt(diff, zeros);
    let neg  = T::lt(diff, zeros);
    let sign = T::where_(pos, ones, T::where_(neg, neg_one, zeros));

    let in_quad   = T::le(abs_diff, delta_t);
    // quadratic gradient: diff; linear gradient: delta * sign(diff)
    let grad = T::where_(in_quad, diff, delta_t * sign);
    let dx = grad * dy;
    T::store(dx_ptr.add_offsets(offsets), dx, Some(in_bounds), &[], None, None);
}

// ── SmoothL1Loss ──────────────────────────────────────────────────────────────

/// Element-wise SmoothL1 (Huber variant) forward.
///
/// PyTorch convention with `beta`:
/// ```text
/// out = 0.5 * diff^2 / beta     if |diff| < beta
///     = |diff| - 0.5 * beta     otherwise
/// ```
#[kernel]
pub fn smooth_l1_loss_forward<T: Triton, const BLOCK_SIZE: i32>(
    x_ptr: T::Pointer<f32>,
    y_ptr: T::Pointer<f32>,
    out_ptr: T::Pointer<f32>,
    n_elements: i32,
    beta: f32,
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

    let x = T::load(x_ptr.add_offsets(offsets), Some(in_bounds), Some(zeros), &[], None, None, None, false);
    let y = T::load(y_ptr.add_offsets(offsets), Some(in_bounds), Some(zeros), &[], None, None, None, false);

    let diff     = x - y;
    let abs_diff = T::abs(diff);
    let beta_t   = T::full(&[BLOCK_SIZE], beta);
    let half     = T::full(&[BLOCK_SIZE], 0.5_f32);

    // quadratic: 0.5 * diff^2 / beta
    let quad = half * diff * diff / beta_t;
    // linear: |diff| - 0.5 * beta
    let lin  = abs_diff - half * beta_t;

    let in_quad = T::lt(abs_diff, beta_t);
    let loss = T::where_(in_quad, quad, lin);
    T::store(out_ptr.add_offsets(offsets), loss, Some(in_bounds), &[], None, None);
}

/// Element-wise SmoothL1 backward.
///
/// ```text
/// dx = diff / beta * dy       if |diff| < beta
///    = sign(diff) * dy        otherwise
/// ```
#[kernel]
pub fn smooth_l1_loss_backward<T: Triton, const BLOCK_SIZE: i32>(
    dy_ptr: T::Pointer<f32>,
    x_ptr: T::Pointer<f32>,
    y_ptr: T::Pointer<f32>,
    dx_ptr: T::Pointer<f32>,
    n_elements: i32,
    beta: f32,
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
    let x  = T::load(x_ptr.add_offsets(offsets),  Some(in_bounds), Some(zeros), &[], None, None, None, false);
    let y  = T::load(y_ptr.add_offsets(offsets),  Some(in_bounds), Some(zeros), &[], None, None, None, false);

    let diff     = x - y;
    let abs_diff = T::abs(diff);
    let beta_t   = T::full(&[BLOCK_SIZE], beta);
    let ones     = T::full(&[BLOCK_SIZE],  1.0_f32);
    let neg_one  = T::full(&[BLOCK_SIZE], -1.0_f32);

    let pos  = T::gt(diff, zeros);
    let neg  = T::lt(diff, zeros);
    let sign = T::where_(pos, ones, T::where_(neg, neg_one, zeros));

    let in_quad = T::lt(abs_diff, beta_t);
    // quadratic gradient: diff / beta; linear gradient: sign(diff)
    let grad = T::where_(in_quad, diff / beta_t, sign);
    let dx = grad * dy;
    T::store(dx_ptr.add_offsets(offsets), dx, Some(in_bounds), &[], None, None);
}
