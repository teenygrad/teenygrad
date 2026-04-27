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

// ── Hardtanh ─────────────────────────────────────────────────────────────────

/// Forward: y = clamp(x, min_val, max_val)
#[kernel]
pub fn hardtanh_forward<T: Triton, const BLOCK_SIZE: i32>(
    x_ptr: T::Pointer<f32>,
    y_ptr: T::Pointer<f32>,
    n_elements: i32,
    min_val: f32,
    max_val: f32,
) where
    T::I32Tensor: types::Tensor<i32, 1>,
    T::I32Tensor: Comparison<i32, BoolTensor = T::BoolTensor>,
    T::Pointer<f32>: AddOffsets<i32, 1, T::I32Tensor, Output = T::Tensor<T::Pointer<f32>>>,
{
    let pid = T::program_id(Axis::X);
    let block_start = pid * BLOCK_SIZE;
    let offsets = T::arange(0, BLOCK_SIZE) + block_start;
    let in_bounds = offsets.lt(n_elements);

    let x = T::load(x_ptr.add_offsets(offsets), Some(in_bounds), None, &[], None, None, None, false);
    let lo = T::full(&[BLOCK_SIZE], min_val);
    let hi = T::full(&[BLOCK_SIZE], max_val);
    let y  = T::clamp(x, lo, hi);
    T::store(y_ptr.add_offsets(offsets), y, Some(in_bounds), &[], None, None);
}

/// Backward: dx = dy if min_val < x < max_val else 0
#[kernel]
pub fn hardtanh_backward<T: Triton, const BLOCK_SIZE: i32>(
    dy_ptr: T::Pointer<f32>,
    x_ptr: T::Pointer<f32>,
    dx_ptr: T::Pointer<f32>,
    n_elements: i32,
    min_val: f32,
    max_val: f32,
) where
    T::I32Tensor: types::Tensor<i32, 1>,
    T::I32Tensor: Comparison<i32, BoolTensor = T::BoolTensor>,
    T::Pointer<f32>: AddOffsets<i32, 1, T::I32Tensor, Output = T::Tensor<T::Pointer<f32>>>,
{
    let pid = T::program_id(Axis::X);
    let block_start = pid * BLOCK_SIZE;
    let offsets = T::arange(0, BLOCK_SIZE) + block_start;
    let in_bounds = offsets.lt(n_elements);

    let dy = T::load(dy_ptr.add_offsets(offsets), Some(in_bounds), None, &[], None, None, None, false);
    let x  = T::load(x_ptr.add_offsets(offsets),  Some(in_bounds), None, &[], None, None, None, false);

    // |x - midpoint| < half_range  ≡  min_val < x < max_val
    let lo      = T::full(&[BLOCK_SIZE], min_val);
    let hi      = T::full(&[BLOCK_SIZE], max_val);
    let in_range = T::gt(T::minimum(x - lo, hi - x), T::zeros_like(x));
    let dx = T::where_(in_range, dy, T::zeros_like(dy));
    T::store(dx_ptr.add_offsets(offsets), dx, Some(in_bounds), &[], None, None);
}

// ── ReLU6 ────────────────────────────────────────────────────────────────────

/// Forward: y = clamp(x, 0, 6)
#[kernel]
pub fn relu6_forward<T: Triton, const BLOCK_SIZE: i32>(
    x_ptr: T::Pointer<f32>,
    y_ptr: T::Pointer<f32>,
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

    let x  = T::load(x_ptr.add_offsets(offsets), Some(in_bounds), None, &[], None, None, None, false);
    let lo = T::zeros_like(x);
    let hi = T::full(&[BLOCK_SIZE], 6.0_f32);
    let y  = T::clamp(x, lo, hi);
    T::store(y_ptr.add_offsets(offsets), y, Some(in_bounds), &[], None, None);
}

/// Backward: dx = dy if 0 < x < 6 else 0
#[kernel]
pub fn relu6_backward<T: Triton, const BLOCK_SIZE: i32>(
    dy_ptr: T::Pointer<f32>,
    x_ptr: T::Pointer<f32>,
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

    let dy = T::load(dy_ptr.add_offsets(offsets), Some(in_bounds), None, &[], None, None, None, false);
    let x  = T::load(x_ptr.add_offsets(offsets),  Some(in_bounds), None, &[], None, None, None, false);

    let six      = T::full(&[BLOCK_SIZE], 6.0_f32);
    // min(x, 6-x) > 0  ≡  0 < x < 6
    let in_range = T::gt(T::minimum(x, six - x), T::zeros_like(x));
    let dx = T::where_(in_range, dy, T::zeros_like(dy));
    T::store(dx_ptr.add_offsets(offsets), dx, Some(in_bounds), &[], None, None);
}

// ── Hardsigmoid ──────────────────────────────────────────────────────────────

/// Forward: y = clamp((x + 3) / 6, 0, 1)
#[kernel]
pub fn hardsigmoid_forward<T: Triton, const BLOCK_SIZE: i32>(
    x_ptr: T::Pointer<f32>,
    y_ptr: T::Pointer<f32>,
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

    let x    = T::load(x_ptr.add_offsets(offsets), Some(in_bounds), None, &[], None, None, None, false);
    let three = T::full(&[BLOCK_SIZE], 3.0_f32);
    let six   = T::full(&[BLOCK_SIZE], 6.0_f32);
    let lo    = T::zeros_like(x);
    let hi    = T::full(&[BLOCK_SIZE], 1.0_f32);
    let y     = T::clamp((x + three) / six, lo, hi);
    T::store(y_ptr.add_offsets(offsets), y, Some(in_bounds), &[], None, None);
}

/// Backward: dx = dy/6 if |x| < 3 else 0
#[kernel]
pub fn hardsigmoid_backward<T: Triton, const BLOCK_SIZE: i32>(
    dy_ptr: T::Pointer<f32>,
    x_ptr: T::Pointer<f32>,
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

    let dy = T::load(dy_ptr.add_offsets(offsets), Some(in_bounds), None, &[], None, None, None, false);
    let x  = T::load(x_ptr.add_offsets(offsets),  Some(in_bounds), None, &[], None, None, None, false);

    let three    = T::full(&[BLOCK_SIZE], 3.0_f32);
    let in_range = T::lt(T::abs(x), three);          // |x| < 3
    let sixth    = T::full(&[BLOCK_SIZE], 1.0_f32 / 6.0_f32);
    let dx = T::where_(in_range, dy * sixth, T::zeros_like(dy));
    T::store(dx_ptr.add_offsets(offsets), dx, Some(in_bounds), &[], None, None);
}

// ── Hardswish ────────────────────────────────────────────────────────────────

/// Forward: y = x * clamp((x + 3) / 6, 0, 1)
#[kernel]
pub fn hardswish_forward<T: Triton, const BLOCK_SIZE: i32>(
    x_ptr: T::Pointer<f32>,
    y_ptr: T::Pointer<f32>,
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

    let x     = T::load(x_ptr.add_offsets(offsets), Some(in_bounds), None, &[], None, None, None, false);
    let three = T::full(&[BLOCK_SIZE], 3.0_f32);
    let six   = T::full(&[BLOCK_SIZE], 6.0_f32);
    let lo    = T::zeros_like(x);
    let hi    = T::full(&[BLOCK_SIZE], 1.0_f32);
    let hs    = T::clamp((x + three) / six, lo, hi);
    let y     = x * hs;
    T::store(y_ptr.add_offsets(offsets), y, Some(in_bounds), &[], None, None);
}

/// Backward:
///   dx = 0           if x <= -3
///   dx = dy          if x >= 3
///   dx = dy*(2x+3)/6 otherwise
#[kernel]
pub fn hardswish_backward<T: Triton, const BLOCK_SIZE: i32>(
    dy_ptr: T::Pointer<f32>,
    x_ptr: T::Pointer<f32>,
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

    let dy = T::load(dy_ptr.add_offsets(offsets), Some(in_bounds), None, &[], None, None, None, false);
    let x  = T::load(x_ptr.add_offsets(offsets),  Some(in_bounds), None, &[], None, None, None, false);

    let three      = T::full(&[BLOCK_SIZE], 3.0_f32);
    let neg_three  = T::full(&[BLOCK_SIZE], -3.0_f32);
    let six        = T::full(&[BLOCK_SIZE], 6.0_f32);
    let two        = T::full(&[BLOCK_SIZE], 2.0_f32);

    let x_le_neg3  = T::le(x, neg_three);
    let x_ge_3     = T::ge(x, three);
    let dx_mid     = dy * (two * x + three) / six;

    // Build from inner outward: start with mid, then override boundary regions.
    let dx_not_lo = T::where_(x_ge_3, dy, dx_mid);
    let dx        = T::where_(x_le_neg3, T::zeros_like(dy), dx_not_lo);
    T::store(dx_ptr.add_offsets(offsets), dx, Some(in_bounds), &[], None, None);
}

// ── Hardshrink ───────────────────────────────────────────────────────────────

/// Forward: y = x if |x| > lambda else 0
#[kernel]
pub fn hardshrink_forward<T: Triton, const BLOCK_SIZE: i32>(
    x_ptr: T::Pointer<f32>,
    y_ptr: T::Pointer<f32>,
    n_elements: i32,
    lambda: f32,
) where
    T::I32Tensor: types::Tensor<i32, 1>,
    T::I32Tensor: Comparison<i32, BoolTensor = T::BoolTensor>,
    T::Pointer<f32>: AddOffsets<i32, 1, T::I32Tensor, Output = T::Tensor<T::Pointer<f32>>>,
{
    let pid = T::program_id(Axis::X);
    let block_start = pid * BLOCK_SIZE;
    let offsets = T::arange(0, BLOCK_SIZE) + block_start;
    let in_bounds = offsets.lt(n_elements);

    let x       = T::load(x_ptr.add_offsets(offsets), Some(in_bounds), None, &[], None, None, None, false);
    let lam     = T::full(&[BLOCK_SIZE], lambda);
    let outside = T::gt(T::abs(x), lam);
    let y       = T::where_(outside, x, T::zeros_like(x));
    T::store(y_ptr.add_offsets(offsets), y, Some(in_bounds), &[], None, None);
}

/// Backward: dx = dy if |x| > lambda else 0
#[kernel]
pub fn hardshrink_backward<T: Triton, const BLOCK_SIZE: i32>(
    dy_ptr: T::Pointer<f32>,
    x_ptr: T::Pointer<f32>,
    dx_ptr: T::Pointer<f32>,
    n_elements: i32,
    lambda: f32,
) where
    T::I32Tensor: types::Tensor<i32, 1>,
    T::I32Tensor: Comparison<i32, BoolTensor = T::BoolTensor>,
    T::Pointer<f32>: AddOffsets<i32, 1, T::I32Tensor, Output = T::Tensor<T::Pointer<f32>>>,
{
    let pid = T::program_id(Axis::X);
    let block_start = pid * BLOCK_SIZE;
    let offsets = T::arange(0, BLOCK_SIZE) + block_start;
    let in_bounds = offsets.lt(n_elements);

    let dy      = T::load(dy_ptr.add_offsets(offsets), Some(in_bounds), None, &[], None, None, None, false);
    let x       = T::load(x_ptr.add_offsets(offsets),  Some(in_bounds), None, &[], None, None, None, false);
    let lam     = T::full(&[BLOCK_SIZE], lambda);
    let outside = T::gt(T::abs(x), lam);
    let dx      = T::where_(outside, dy, T::zeros_like(dy));
    T::store(dx_ptr.add_offsets(offsets), dx, Some(in_bounds), &[], None, None);
}


pub struct HardtanhOp {
    pub forward: HardtanhForward,
    pub backward: HardtanhBackward,
}

pub struct Relu6Op {
    pub forward: Relu6Forward,
    pub backward: Relu6Backward,
}

pub struct HardsigmoidOp {
    pub forward: HardsigmoidForward,
    pub backward: HardsigmoidBackward,
}

pub struct HardswishOp {
    pub forward: HardswishForward,
    pub backward: HardswishBackward,
}

pub struct HardshrinkOp {
    pub forward: HardshrinkForward,
    pub backward: HardshrinkBackward,
}
