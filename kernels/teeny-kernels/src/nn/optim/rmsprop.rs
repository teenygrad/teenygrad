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

/// RMSprop step (no momentum, not centred).
///
/// ```text
/// square_avg = alpha * square_avg + (1 - alpha) * g²
/// p          = p - lr * g / (sqrt(square_avg) + eps)
/// ```
///
/// Grid: `[ceil(n_elements / BLOCK_SIZE), 1, 1]`.
#[kernel]
pub fn rmsprop_step<T: Triton, const BLOCK_SIZE: i32>(
    params_ptr: T::Pointer<f32>,
    grad_ptr: T::Pointer<f32>,
    square_avg_ptr: T::Pointer<f32>,
    n_elements: i32,
    lr: f32,
    alpha: f32,
    eps: f32,
    weight_decay: f32,
) where
    T::I32Tensor: types::Tensor<i32, 1>,
    T::I32Tensor: Comparison<i32, BoolTensor = T::BoolTensor>,
    T::Pointer<f32>: AddOffsets<i32, 1, T::I32Tensor, Output = T::Tensor<T::Pointer<f32>>>,
{
    let pid = T::program_id(Axis::X);
    let block_start = pid * BLOCK_SIZE;
    let offsets = T::arange(0, BLOCK_SIZE) + block_start;
    let mask = offsets.lt(n_elements);

    let p          = T::load(params_ptr.add_offsets(offsets),    Some(mask), None, &[], None, None, None, false);
    let g          = T::load(grad_ptr.add_offsets(offsets),      Some(mask), None, &[], None, None, None, false);
    let square_avg = T::load(square_avg_ptr.add_offsets(offsets), Some(mask), None, &[], None, None, None, false);

    let lr_t       = T::full(&[BLOCK_SIZE], lr);
    let alpha_t    = T::full(&[BLOCK_SIZE], alpha);
    let one_m_alpha = T::full(&[BLOCK_SIZE], 1.0_f32 - alpha);
    let eps_t      = T::full(&[BLOCK_SIZE], eps);
    let wd_t       = T::full(&[BLOCK_SIZE], weight_decay);

    let g_eff          = g + wd_t * p;
    let square_avg_new = alpha_t * square_avg + one_m_alpha * g_eff * g_eff;
    let p_new          = p - lr_t * g_eff / (T::sqrt_rn(square_avg_new) + eps_t);

    T::store(params_ptr.add_offsets(offsets),    p_new,          Some(mask), &[], None, None);
    T::store(square_avg_ptr.add_offsets(offsets), square_avg_new, Some(mask), &[], None, None);
}

/// RMSprop step with momentum.
///
/// ```text
/// square_avg = alpha * square_avg + (1 - alpha) * g²
/// buf        = momentum * buf + g / sqrt(square_avg + eps)
/// p          = p - lr * buf
/// ```
///
/// Grid: `[ceil(n_elements / BLOCK_SIZE), 1, 1]`.
#[kernel]
pub fn rmsprop_momentum_step<T: Triton, const BLOCK_SIZE: i32>(
    params_ptr: T::Pointer<f32>,
    grad_ptr: T::Pointer<f32>,
    square_avg_ptr: T::Pointer<f32>,
    buf_ptr: T::Pointer<f32>,
    n_elements: i32,
    lr: f32,
    alpha: f32,
    eps: f32,
    weight_decay: f32,
    momentum: f32,
) where
    T::I32Tensor: types::Tensor<i32, 1>,
    T::I32Tensor: Comparison<i32, BoolTensor = T::BoolTensor>,
    T::Pointer<f32>: AddOffsets<i32, 1, T::I32Tensor, Output = T::Tensor<T::Pointer<f32>>>,
{
    let pid = T::program_id(Axis::X);
    let block_start = pid * BLOCK_SIZE;
    let offsets = T::arange(0, BLOCK_SIZE) + block_start;
    let mask = offsets.lt(n_elements);

    let p          = T::load(params_ptr.add_offsets(offsets),    Some(mask), None, &[], None, None, None, false);
    let g          = T::load(grad_ptr.add_offsets(offsets),      Some(mask), None, &[], None, None, None, false);
    let square_avg = T::load(square_avg_ptr.add_offsets(offsets), Some(mask), None, &[], None, None, None, false);
    let buf        = T::load(buf_ptr.add_offsets(offsets),        Some(mask), None, &[], None, None, None, false);

    let lr_t        = T::full(&[BLOCK_SIZE], lr);
    let alpha_t     = T::full(&[BLOCK_SIZE], alpha);
    let one_m_alpha = T::full(&[BLOCK_SIZE], 1.0_f32 - alpha);
    let eps_t       = T::full(&[BLOCK_SIZE], eps);
    let wd_t        = T::full(&[BLOCK_SIZE], weight_decay);
    let mu_t        = T::full(&[BLOCK_SIZE], momentum);

    let g_eff          = g + wd_t * p;
    let square_avg_new = alpha_t * square_avg + one_m_alpha * g_eff * g_eff;
    let buf_new        = mu_t * buf + g_eff / T::sqrt_rn(square_avg_new + eps_t);
    let p_new          = p - lr_t * buf_new;

    T::store(params_ptr.add_offsets(offsets),     p_new,          Some(mask), &[], None, None);
    T::store(square_avg_ptr.add_offsets(offsets),  square_avg_new, Some(mask), &[], None, None);
    T::store(buf_ptr.add_offsets(offsets),         buf_new,        Some(mask), &[], None, None);
}
