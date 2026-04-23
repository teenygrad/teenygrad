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

// 1/sqrt(2)
const INV_SQRT2: f32 = 0.7071067811865476_f32;
// 1/sqrt(2*pi)
const INV_SQRT_2PI: f32 = 0.3989422804014327_f32;

// ── GELU ─────────────────────────────────────────────────────────────────────

/// Forward: y = x / (1 + exp(-2 * c * (x + a*x³)))
///   where c = sqrt(2/pi), a = 0.044715 — the tanh GELU approximation.
#[kernel]
pub fn gelu_forward<T: Triton, const BLOCK_SIZE: i32>(
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

    let x = T::load(x_ptr.add_offsets(offsets), Some(in_bounds), None, &[], None, None, None, false);

    let one         = T::full(&[BLOCK_SIZE], 1.0_f32);
    let neg2c       = T::full(&[BLOCK_SIZE], -2.0_f32 * 0.7978845608028654_f32);
    let coeff       = T::full(&[BLOCK_SIZE], 0.044715_f32);

    // tanh-GELU: y = x * 0.5 * (1 + tanh(c*(x + a*x³)))
    //              = x / (1 + exp(-2c*(x + a*x³)))
    let inner = x + coeff * x * x * x;
    let y = x / (one + T::exp(neg2c * inner));
    T::store(y_ptr.add_offsets(offsets), y, Some(in_bounds), &[], None, None);
}

/// Backward of the tanh-GELU approximation.
///   Let inner = x + a*x³, s = sigmoid(2c*inner), t = tanh(c*inner) = 2s-1
///   d/dx = 0.5*(1 + t) + x * 0.5 * sech²(c*inner) * c*(1+3a*x²)
#[kernel]
pub fn gelu_backward<T: Triton, const BLOCK_SIZE: i32>(
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

    let one         = T::full(&[BLOCK_SIZE], 1.0_f32);
    let half        = T::full(&[BLOCK_SIZE], 0.5_f32);
    let two         = T::full(&[BLOCK_SIZE], 2.0_f32);
    let three       = T::full(&[BLOCK_SIZE], 3.0_f32);
    let c           = T::full(&[BLOCK_SIZE], 0.7978845608028654_f32);
    let neg2c       = T::full(&[BLOCK_SIZE], -2.0_f32 * 0.7978845608028654_f32);
    let coeff       = T::full(&[BLOCK_SIZE], 0.044715_f32);

    let inner  = x + coeff * x * x * x;
    let s      = one / (one + T::exp(neg2c * inner));  // sigmoid(2c*inner)
    let t      = two * s - one;                         // tanh(c*inner)
    let sech2  = one - t * t;                           // 1 - tanh²
    let dinner = c * (one + three * coeff * x * x);
    let dx     = dy * (half * (one + t) + x * half * sech2 * dinner);
    T::store(dx_ptr.add_offsets(offsets), dx, Some(in_bounds), &[], None, None);
}

// ── Mish ─────────────────────────────────────────────────────────────────────

/// Forward: y = x * tanh(softplus(x)) = x * tanh(log(1 + exp(x)))
#[kernel]
pub fn mish_forward<T: Triton, const BLOCK_SIZE: i32>(
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
    let one  = T::full(&[BLOCK_SIZE], 1.0_f32);
    let two  = T::full(&[BLOCK_SIZE], 2.0_f32);
    let neg2 = T::full(&[BLOCK_SIZE], -2.0_f32);
    let sp   = T::log(one + T::exp(x));              // softplus(x)
    // tanh(sp) = 2*sigmoid(2*sp) - 1 = 2/(1+exp(-2*sp)) - 1
    let s2   = one / (one + T::exp(neg2 * sp));
    let t    = two * s2 - one;
    let y    = x * t;
    T::store(y_ptr.add_offsets(offsets), y, Some(in_bounds), &[], None, None);
}

/// Backward: dx = dy * (tanh(sp) + x * (1 - tanh²(sp)) * sigmoid(x))
///   where sp = softplus(x). Recomputes all intermediates from x.
#[kernel]
pub fn mish_backward<T: Triton, const BLOCK_SIZE: i32>(
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

    let dy   = T::load(dy_ptr.add_offsets(offsets), Some(in_bounds), None, &[], None, None, None, false);
    let x    = T::load(x_ptr.add_offsets(offsets),  Some(in_bounds), None, &[], None, None, None, false);
    let one  = T::full(&[BLOCK_SIZE], 1.0_f32);
    let two  = T::full(&[BLOCK_SIZE], 2.0_f32);
    let neg1 = T::full(&[BLOCK_SIZE], -1.0_f32);
    let neg2 = T::full(&[BLOCK_SIZE], -2.0_f32);
    let sp   = T::log(one + T::exp(x));
    let s2   = one / (one + T::exp(neg2 * sp));
    let t    = two * s2 - one;                        // tanh(sp)
    // sigmoid(x) = 1 / (1 + exp(-x))
    let s    = one / (one + T::exp(neg1 * x));
    // dx = t + x * (1 - t²) * s
    let dx   = dy * (t + x * (one - t * t) * s);
    T::store(dx_ptr.add_offsets(offsets), dx, Some(in_bounds), &[], None, None);
}
