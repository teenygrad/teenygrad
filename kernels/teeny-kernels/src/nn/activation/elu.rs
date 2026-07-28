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

// ── ELU ──────────────────────────────────────────────────────────────────────

/// Forward: y = x if x > 0 else alpha*(exp(x) - 1)
#[kernel(backward = EluBackward)]
pub fn elu_forward<T: Triton, D: Float, const BLOCK_SIZE: i32>(
    x_ptr: T::Pointer<D>,
    y_ptr: T::Pointer<D>,
    n_elements: i32,
    alpha: f32,
) where
    T::I32Tensor: types::Tensor<i32, 1>,
    T::I32Tensor: Comparison<i32, BoolTensor = T::BoolTensor>,
    T::Pointer<D>: AddOffsets<i32, 1, T::I32Tensor, Output = T::Tensor<T::Pointer<D>>>,
{
    let pid = T::program_id(Axis::X);
    let block_start = pid * BLOCK_SIZE;
    let offsets = T::arange(0, BLOCK_SIZE) + block_start;
    let in_bounds = offsets.lt(n_elements);

    let x       = T::load(x_ptr.add_offsets(offsets), Some(in_bounds), None, &[], None, None, None, false);
    let one     = T::full(&[BLOCK_SIZE], D::from_f64(1.0));
    let alpha_t = T::full(&[BLOCK_SIZE], D::from_f64(alpha as f64));
    let x_pos   = T::gt(x, T::zeros_like(x));
    let y       = T::where_(x_pos, x, alpha_t * (T::exp(x) - one));
    T::store(y_ptr.add_offsets(offsets), y, Some(in_bounds), &[], None, None);
}

/// Backward: dx = dy if x > 0 else dy * alpha * exp(x)
#[kernel]
pub fn elu_backward<T: Triton, D: Float, const BLOCK_SIZE: i32>(
    dy_ptr: T::Pointer<D>,
    x_ptr: T::Pointer<D>,
    dx_ptr: T::Pointer<D>,
    n_elements: i32,
    alpha: f32,
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
    let alpha_t = T::full(&[BLOCK_SIZE], D::from_f64(alpha as f64));
    let x_pos   = T::gt(x, T::zeros_like(x));
    let dx      = T::where_(x_pos, dy, dy * alpha_t * T::exp(x));
    T::store(dx_ptr.add_offsets(offsets), dx, Some(in_bounds), &[], None, None);
}

// ── SELU ─────────────────────────────────────────────────────────────────────

/// Forward: y = SCALE * (x if x > 0 else ALPHA*(exp(x) - 1))
#[kernel(backward = SeluBackward)]
pub fn selu_forward<T: Triton, D: Float, const BLOCK_SIZE: i32>(
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

    let x       = T::load(x_ptr.add_offsets(offsets), Some(in_bounds), None, &[], None, None, None, false);
    let one     = T::full(&[BLOCK_SIZE], D::from_f64(1.0));
    let scale   = T::full(&[BLOCK_SIZE], D::from_f64(1.0507009873554804));
    let alpha   = T::full(&[BLOCK_SIZE], D::from_f64(1.6732632423543772));
    let x_pos   = T::gt(x, T::zeros_like(x));
    let y       = scale * T::where_(x_pos, x, alpha * (T::exp(x) - one));
    T::store(y_ptr.add_offsets(offsets), y, Some(in_bounds), &[], None, None);
}

/// Backward: dx = SCALE*dy if x > 0 else dy * SCALE*ALPHA*exp(x)
#[kernel]
pub fn selu_backward<T: Triton, D: Float, const BLOCK_SIZE: i32>(
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

    let dy           = T::load(dy_ptr.add_offsets(offsets), Some(in_bounds), None, &[], None, None, None, false);
    let x            = T::load(x_ptr.add_offsets(offsets),  Some(in_bounds), None, &[], None, None, None, false);
    let scale        = T::full(&[BLOCK_SIZE], D::from_f64(1.0507009873554804));
    let scale_alpha  = T::full(&[BLOCK_SIZE], D::from_f64(1.7580993408473766));
    let x_pos        = T::gt(x, T::zeros_like(x));
    let dx           = T::where_(x_pos, dy * scale, dy * scale_alpha * T::exp(x));
    T::store(dx_ptr.add_offsets(offsets), dx, Some(in_bounds), &[], None, None);
}

// ── CELU ─────────────────────────────────────────────────────────────────────

/// Forward: y = max(0, x) + min(0, alpha*(exp(x/alpha) - 1))
#[kernel(backward = CeluBackward)]
pub fn celu_forward<T: Triton, D: Float, const BLOCK_SIZE: i32>(
    x_ptr: T::Pointer<D>,
    y_ptr: T::Pointer<D>,
    n_elements: i32,
    alpha: f32,
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
    let zero      = T::zeros_like(x);
    let one       = T::full(&[BLOCK_SIZE], D::from_f64(1.0));
    let alpha_t   = T::full(&[BLOCK_SIZE], D::from_f64(alpha as f64));
    let inv_alpha = T::full(&[BLOCK_SIZE], D::from_f64(1.0 / alpha as f64));
    let elu_neg   = alpha_t * (T::exp(x * inv_alpha) - one);
    let y         = T::maximum(zero, x) + T::minimum(zero, elu_neg);
    T::store(y_ptr.add_offsets(offsets), y, Some(in_bounds), &[], None, None);
}

/// Backward: dx = dy if x >= 0 else dy * exp(x/alpha)
#[kernel]
pub fn celu_backward<T: Triton, D: Float, const BLOCK_SIZE: i32>(
    dy_ptr: T::Pointer<D>,
    x_ptr: T::Pointer<D>,
    dx_ptr: T::Pointer<D>,
    n_elements: i32,
    alpha: f32,
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
    let inv_alpha = T::full(&[BLOCK_SIZE], D::from_f64(1.0 / alpha as f64));
    let x_ge_zero = T::ge(x, T::zeros_like(x));
    let dx        = T::where_(x_ge_zero, dy, dy * T::exp(x * inv_alpha));
    T::store(dx_ptr.add_offsets(offsets), dx, Some(in_bounds), &[], None, None);
}


pub struct EluOp<D: Float> {
    pub forward: EluForward<D>,
    pub backward: EluBackward<D>,
}

pub struct SeluOp<D: Float> {
    pub forward: SeluForward<D>,
    pub backward: SeluBackward<D>,
}

pub struct CeluOp<D: Float> {
    pub forward: CeluForward<D>,
    pub backward: CeluBackward<D>,
}
