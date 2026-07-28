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

use teeny_core::dtype::Float;
use teeny_macros::kernel;
use teeny_triton::triton::{
    types::{AddOffsets, Comparison},
    *,
};

// ── LeakyReLU ────────────────────────────────────────────────────────────────

/// Forward: y = x if x > 0 else negative_slope * x
#[kernel(backward = LeakyReluBackward)]
pub fn leaky_relu_forward<T: Triton, D: Float, const BLOCK_SIZE: i32>(
    x_ptr: T::Pointer<D>,
    y_ptr: T::Pointer<D>,
    n_elements: i32,
    negative_slope: f32,
) where
    T::I32Tensor: types::Tensor<i32, 1>,
    T::I32Tensor: Comparison<i32, BoolTensor = T::BoolTensor>,
    T::Pointer<D>: AddOffsets<i32, 1, T::I32Tensor, Output = T::Tensor<T::Pointer<D>>>,
{
    let pid = T::program_id(Axis::X);
    let block_start = pid * BLOCK_SIZE;
    let offsets = T::arange(0, BLOCK_SIZE) + block_start;
    let in_bounds = offsets.lt(n_elements);

    let x      = T::load(x_ptr.add_offsets(offsets), Some(in_bounds), None, &[], None, None, None, false);
    let slope  = T::full(&[BLOCK_SIZE], D::from_f64(negative_slope as f64));
    let x_pos  = T::gt(x, T::zeros_like(x));
    let y      = T::where_(x_pos, x, slope * x);
    T::store(y_ptr.add_offsets(offsets), y, Some(in_bounds), &[], None, None);
}

/// Backward: dx = dy if x > 0 else negative_slope * dy
#[kernel]
pub fn leaky_relu_backward<T: Triton, D: Float, const BLOCK_SIZE: i32>(
    dy_ptr: T::Pointer<D>,
    x_ptr: T::Pointer<D>,
    dx_ptr: T::Pointer<D>,
    n_elements: i32,
    negative_slope: f32,
) where
    T::I32Tensor: types::Tensor<i32, 1>,
    T::I32Tensor: Comparison<i32, BoolTensor = T::BoolTensor>,
    T::Pointer<D>: AddOffsets<i32, 1, T::I32Tensor, Output = T::Tensor<T::Pointer<D>>>,
{
    let pid = T::program_id(Axis::X);
    let block_start = pid * BLOCK_SIZE;
    let offsets = T::arange(0, BLOCK_SIZE) + block_start;
    let in_bounds = offsets.lt(n_elements);

    let dy     = T::load(dy_ptr.add_offsets(offsets), Some(in_bounds), None, &[], None, None, None, false);
    let x      = T::load(x_ptr.add_offsets(offsets),  Some(in_bounds), None, &[], None, None, None, false);
    let slope  = T::full(&[BLOCK_SIZE], D::from_f64(negative_slope as f64));
    let x_pos  = T::gt(x, T::zeros_like(x));
    let dx     = T::where_(x_pos, dy, slope * dy);
    T::store(dx_ptr.add_offsets(offsets), dx, Some(in_bounds), &[], None, None);
}

// ── Threshold ────────────────────────────────────────────────────────────────

/// Forward: y = x if x > threshold else value
#[kernel(backward = ThresholdBackward)]
pub fn threshold_forward<T: Triton, D: Float, const BLOCK_SIZE: i32>(
    x_ptr: T::Pointer<D>,
    y_ptr: T::Pointer<D>,
    n_elements: i32,
    threshold: f32,
    value: f32,
) where
    T::I32Tensor: types::Tensor<i32, 1>,
    T::I32Tensor: Comparison<i32, BoolTensor = T::BoolTensor>,
    T::Pointer<D>: AddOffsets<i32, 1, T::I32Tensor, Output = T::Tensor<T::Pointer<D>>>,
{
    let pid = T::program_id(Axis::X);
    let block_start = pid * BLOCK_SIZE;
    let offsets = T::arange(0, BLOCK_SIZE) + block_start;
    let in_bounds = offsets.lt(n_elements);

    let x      = T::load(x_ptr.add_offsets(offsets), Some(in_bounds), None, &[], None, None, None, false);
    let thr    = T::full(&[BLOCK_SIZE], D::from_f64(threshold as f64));
    let val    = T::full(&[BLOCK_SIZE], D::from_f64(value as f64));
    let above  = T::gt(x, thr);
    let y      = T::where_(above, x, val);
    T::store(y_ptr.add_offsets(offsets), y, Some(in_bounds), &[], None, None);
}

/// Backward: dx = dy if x > threshold else 0
#[kernel]
pub fn threshold_backward<T: Triton, D: Float, const BLOCK_SIZE: i32>(
    dy_ptr: T::Pointer<D>,
    x_ptr: T::Pointer<D>,
    dx_ptr: T::Pointer<D>,
    n_elements: i32,
    threshold: f32,
) where
    T::I32Tensor: types::Tensor<i32, 1>,
    T::I32Tensor: Comparison<i32, BoolTensor = T::BoolTensor>,
    T::Pointer<D>: AddOffsets<i32, 1, T::I32Tensor, Output = T::Tensor<T::Pointer<D>>>,
{
    let pid = T::program_id(Axis::X);
    let block_start = pid * BLOCK_SIZE;
    let offsets = T::arange(0, BLOCK_SIZE) + block_start;
    let in_bounds = offsets.lt(n_elements);

    let dy     = T::load(dy_ptr.add_offsets(offsets), Some(in_bounds), None, &[], None, None, None, false);
    let x      = T::load(x_ptr.add_offsets(offsets),  Some(in_bounds), None, &[], None, None, None, false);
    let thr    = T::full(&[BLOCK_SIZE], D::from_f64(threshold as f64));
    let above  = T::gt(x, thr);
    let dx     = T::where_(above, dy, T::zeros_like(dy));
    T::store(dx_ptr.add_offsets(offsets), dx, Some(in_bounds), &[], None, None);
}

// ── Softsign ─────────────────────────────────────────────────────────────────

/// Forward: y = x / (1 + |x|)
#[kernel(backward = SoftsignBackward)]
pub fn softsign_forward<T: Triton, D: Float, const BLOCK_SIZE: i32>(
    x_ptr: T::Pointer<D>,
    y_ptr: T::Pointer<D>,
    n_elements: i32,
) where
    T::I32Tensor: types::Tensor<i32, 1>,
    T::I32Tensor: Comparison<i32, BoolTensor = T::BoolTensor>,
    T::Pointer<D>: AddOffsets<i32, 1, T::I32Tensor, Output = T::Tensor<T::Pointer<D>>>,
{
    let pid = T::program_id(Axis::X);
    let block_start = pid * BLOCK_SIZE;
    let offsets = T::arange(0, BLOCK_SIZE) + block_start;
    let in_bounds = offsets.lt(n_elements);

    let x   = T::load(x_ptr.add_offsets(offsets), Some(in_bounds), None, &[], None, None, None, false);
    let one = T::full(&[BLOCK_SIZE], D::from_f64(1.0));
    let d   = one + T::abs(x);
    let y   = x / d;
    T::store(y_ptr.add_offsets(offsets), y, Some(in_bounds), &[], None, None);
}

/// Backward: dx = dy / (1 + |x|)²
#[kernel]
pub fn softsign_backward<T: Triton, D: Float, const BLOCK_SIZE: i32>(
    dy_ptr: T::Pointer<D>,
    x_ptr: T::Pointer<D>,
    dx_ptr: T::Pointer<D>,
    n_elements: i32,
) where
    T::I32Tensor: types::Tensor<i32, 1>,
    T::I32Tensor: Comparison<i32, BoolTensor = T::BoolTensor>,
    T::Pointer<D>: AddOffsets<i32, 1, T::I32Tensor, Output = T::Tensor<T::Pointer<D>>>,
{
    let pid = T::program_id(Axis::X);
    let block_start = pid * BLOCK_SIZE;
    let offsets = T::arange(0, BLOCK_SIZE) + block_start;
    let in_bounds = offsets.lt(n_elements);

    let dy  = T::load(dy_ptr.add_offsets(offsets), Some(in_bounds), None, &[], None, None, None, false);
    let x   = T::load(x_ptr.add_offsets(offsets),  Some(in_bounds), None, &[], None, None, None, false);
    let one = T::full(&[BLOCK_SIZE], D::from_f64(1.0));
    let d   = one + T::abs(x);
    let dx  = dy / (d * d);
    T::store(dx_ptr.add_offsets(offsets), dx, Some(in_bounds), &[], None, None);
}

// ── Softshrink ───────────────────────────────────────────────────────────────

/// Forward: y = x - lambda if x > lambda, x + lambda if x < -lambda, else 0
#[kernel(backward = SoftshrinkBackward)]
pub fn softshrink_forward<T: Triton, D: Float, const BLOCK_SIZE: i32>(
    x_ptr: T::Pointer<D>,
    y_ptr: T::Pointer<D>,
    n_elements: i32,
    lambda: f32,
) where
    T::I32Tensor: types::Tensor<i32, 1>,
    T::I32Tensor: Comparison<i32, BoolTensor = T::BoolTensor>,
    T::Pointer<D>: AddOffsets<i32, 1, T::I32Tensor, Output = T::Tensor<T::Pointer<D>>>,
{
    let pid = T::program_id(Axis::X);
    let block_start = pid * BLOCK_SIZE;
    let offsets = T::arange(0, BLOCK_SIZE) + block_start;
    let in_bounds = offsets.lt(n_elements);

    let x        = T::load(x_ptr.add_offsets(offsets), Some(in_bounds), None, &[], None, None, None, false);
    let lam      = T::full(&[BLOCK_SIZE], D::from_f64(lambda as f64));
    let neg_lam  = T::full(&[BLOCK_SIZE], D::from_f64(-(lambda as f64)));
    let x_gt_lam = T::gt(x, lam);
    let x_lt_neg = T::lt(x, neg_lam);
    let y_upper  = x - lam;
    let y_lower  = x + lam;
    let y_mid    = T::where_(x_lt_neg, y_lower, T::zeros_like(x));
    let y        = T::where_(x_gt_lam, y_upper, y_mid);
    T::store(y_ptr.add_offsets(offsets), y, Some(in_bounds), &[], None, None);
}

/// Backward: dx = dy if |x| > lambda else 0
#[kernel]
pub fn softshrink_backward<T: Triton, D: Float, const BLOCK_SIZE: i32>(
    dy_ptr: T::Pointer<D>,
    x_ptr: T::Pointer<D>,
    dx_ptr: T::Pointer<D>,
    n_elements: i32,
    lambda: f32,
) where
    T::I32Tensor: types::Tensor<i32, 1>,
    T::I32Tensor: Comparison<i32, BoolTensor = T::BoolTensor>,
    T::Pointer<D>: AddOffsets<i32, 1, T::I32Tensor, Output = T::Tensor<T::Pointer<D>>>,
{
    let pid = T::program_id(Axis::X);
    let block_start = pid * BLOCK_SIZE;
    let offsets = T::arange(0, BLOCK_SIZE) + block_start;
    let in_bounds = offsets.lt(n_elements);

    let dy      = T::load(dy_ptr.add_offsets(offsets), Some(in_bounds), None, &[], None, None, None, false);
    let x       = T::load(x_ptr.add_offsets(offsets),  Some(in_bounds), None, &[], None, None, None, false);
    let lam     = T::full(&[BLOCK_SIZE], D::from_f64(lambda as f64));
    let outside = T::gt(T::abs(x), lam);
    let dx      = T::where_(outside, dy, T::zeros_like(dy));
    T::store(dx_ptr.add_offsets(offsets), dx, Some(in_bounds), &[], None, None);
}

// ── Softplus ─────────────────────────────────────────────────────────────────

/// Forward: y = (1/beta) * log(1 + exp(beta*x))
///   For beta*x > threshold: y ≈ x (numerically safe pass-through)
#[kernel(backward = SoftplusBackward)]
pub fn softplus_forward<T: Triton, D: Float, const BLOCK_SIZE: i32>(
    x_ptr: T::Pointer<D>,
    y_ptr: T::Pointer<D>,
    n_elements: i32,
    beta: f32,
    threshold: f32,
) where
    T::I32Tensor: types::Tensor<i32, 1>,
    T::I32Tensor: Comparison<i32, BoolTensor = T::BoolTensor>,
    T::Pointer<D>: AddOffsets<i32, 1, T::I32Tensor, Output = T::Tensor<T::Pointer<D>>>,
{
    let pid = T::program_id(Axis::X);
    let block_start = pid * BLOCK_SIZE;
    let offsets = T::arange(0, BLOCK_SIZE) + block_start;
    let in_bounds = offsets.lt(n_elements);

    let x         = T::load(x_ptr.add_offsets(offsets), Some(in_bounds), None, &[], None, None, None, false);
    let beta_t    = T::full(&[BLOCK_SIZE], D::from_f64(beta as f64));
    let inv_beta  = T::full(&[BLOCK_SIZE], D::from_f64(1.0 / beta as f64));
    let thr       = T::full(&[BLOCK_SIZE], D::from_f64(threshold as f64));
    let one       = T::full(&[BLOCK_SIZE], D::from_f64(1.0));
    let bx        = beta_t * x;
    let above_thr = T::gt(bx, thr);
    let y_safe    = inv_beta * T::log(one + T::exp(bx));
    let y         = T::where_(above_thr, x, y_safe);
    T::store(y_ptr.add_offsets(offsets), y, Some(in_bounds), &[], None, None);
}

/// Backward: dx = dy * sigmoid(beta*x)
///   For beta*x > threshold: dx ≈ dy
#[kernel]
pub fn softplus_backward<T: Triton, D: Float, const BLOCK_SIZE: i32>(
    dy_ptr: T::Pointer<D>,
    x_ptr: T::Pointer<D>,
    dx_ptr: T::Pointer<D>,
    n_elements: i32,
    beta: f32,
    threshold: f32,
) where
    T::I32Tensor: types::Tensor<i32, 1>,
    T::I32Tensor: Comparison<i32, BoolTensor = T::BoolTensor>,
    T::Pointer<D>: AddOffsets<i32, 1, T::I32Tensor, Output = T::Tensor<T::Pointer<D>>>,
{
    let pid = T::program_id(Axis::X);
    let block_start = pid * BLOCK_SIZE;
    let offsets = T::arange(0, BLOCK_SIZE) + block_start;
    let in_bounds = offsets.lt(n_elements);

    let dy        = T::load(dy_ptr.add_offsets(offsets), Some(in_bounds), None, &[], None, None, None, false);
    let x         = T::load(x_ptr.add_offsets(offsets),  Some(in_bounds), None, &[], None, None, None, false);
    let beta_t    = T::full(&[BLOCK_SIZE], D::from_f64(beta as f64));
    let neg_beta  = T::full(&[BLOCK_SIZE], D::from_f64(-(beta as f64)));
    let thr       = T::full(&[BLOCK_SIZE], D::from_f64(threshold as f64));
    let one       = T::full(&[BLOCK_SIZE], D::from_f64(1.0));
    let bx        = beta_t * x;
    let neg_bx    = neg_beta * x;
    let above_thr = T::gt(bx, thr);
    // sigmoid(bx) = 1 / (1 + exp(-bx))
    let dx_safe   = dy * (one / (one + T::exp(neg_bx)));
    let dx        = T::where_(above_thr, dy, dx_safe);
    T::store(dx_ptr.add_offsets(offsets), dx, Some(in_bounds), &[], None, None);
}


pub struct LeakyReluOp<D: Float> {
    pub forward: LeakyReluForward<D>,
    pub backward: LeakyReluBackward<D>,
}

pub struct ThresholdOp<D: Float> {
    pub forward: ThresholdForward<D>,
    pub backward: ThresholdBackward<D>,
}

pub struct SoftsignOp<D: Float> {
    pub forward: SoftsignForward<D>,
    pub backward: SoftsignBackward<D>,
}

pub struct SoftshrinkOp<D: Float> {
    pub forward: SoftshrinkForward<D>,
    pub backward: SoftshrinkBackward<D>,
}

pub struct SoftplusOp<D: Float> {
    pub forward: SoftplusForward<D>,
    pub backward: SoftplusBackward<D>,
}
