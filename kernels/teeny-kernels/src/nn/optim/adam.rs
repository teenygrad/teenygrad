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

/// Adam step.
///
/// ```text
/// exp_avg    = beta1 * exp_avg    + (1 - beta1) * g
/// exp_avg_sq = beta2 * exp_avg_sq + (1 - beta2) * g²
/// denom      = sqrt(exp_avg_sq) / bias_corr2_sqrt + eps
/// p          = p - step_size * exp_avg / denom
/// ```
///
/// Scalars precomputed on host:
/// - `step_size = lr / bias_correction1`   where `bias_correction1 = 1 - beta1^t`
/// - `bias_corr2_sqrt = sqrt(1 - beta2^t)`
///
/// Grid: `[ceil(n_elements / BLOCK_SIZE), 1, 1]`.
#[kernel]
pub fn adam_step<T: Triton, const BLOCK_SIZE: i32>(
    params_ptr: T::Pointer<f32>,
    grad_ptr: T::Pointer<f32>,
    exp_avg_ptr: T::Pointer<f32>,
    exp_avg_sq_ptr: T::Pointer<f32>,
    n_elements: i32,
    step_size: f32,
    bias_corr2_sqrt: f32,
    beta1: f32,
    beta2: f32,
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

    let p          = T::load(params_ptr.add_offsets(offsets),      Some(mask), None, &[], None, None, None, false);
    let g          = T::load(grad_ptr.add_offsets(offsets),        Some(mask), None, &[], None, None, None, false);
    let exp_avg    = T::load(exp_avg_ptr.add_offsets(offsets),     Some(mask), None, &[], None, None, None, false);
    let exp_avg_sq = T::load(exp_avg_sq_ptr.add_offsets(offsets),  Some(mask), None, &[], None, None, None, false);

    let beta1_t         = T::full(&[BLOCK_SIZE], beta1);
    let beta2_t         = T::full(&[BLOCK_SIZE], beta2);
    let one_m_beta1     = T::full(&[BLOCK_SIZE], 1.0_f32 - beta1);
    let one_m_beta2     = T::full(&[BLOCK_SIZE], 1.0_f32 - beta2);
    let eps_t           = T::full(&[BLOCK_SIZE], eps);
    let step_size_t     = T::full(&[BLOCK_SIZE], step_size);
    let bc2sqrt_t       = T::full(&[BLOCK_SIZE], bias_corr2_sqrt);
    let wd_t            = T::full(&[BLOCK_SIZE], weight_decay);

    let g_eff = g + wd_t * p;

    let exp_avg_new    = beta1_t * exp_avg    + one_m_beta1 * g_eff;
    let exp_avg_sq_new = beta2_t * exp_avg_sq + one_m_beta2 * g_eff * g_eff;

    let half = T::full(&[BLOCK_SIZE], 0.5_f32);
    let denom = T::exp(half * T::log(exp_avg_sq_new)) / bc2sqrt_t + eps_t;
    let p_new = p - step_size_t * exp_avg_new / denom;

    T::store(params_ptr.add_offsets(offsets),     p_new,          Some(mask), &[], None, None);
    T::store(exp_avg_ptr.add_offsets(offsets),     exp_avg_new,    Some(mask), &[], None, None);
    T::store(exp_avg_sq_ptr.add_offsets(offsets),  exp_avg_sq_new, Some(mask), &[], None, None);
}

/// AdamW step (decoupled weight decay).
///
/// ```text
/// p          = p * (1 - lr * weight_decay)
/// exp_avg    = beta1 * exp_avg    + (1 - beta1) * g
/// exp_avg_sq = beta2 * exp_avg_sq + (1 - beta2) * g²
/// denom      = sqrt(exp_avg_sq) / bias_corr2_sqrt + eps
/// p          = p - step_size * exp_avg / denom
/// ```
///
/// Grid: `[ceil(n_elements / BLOCK_SIZE), 1, 1]`.
#[kernel]
pub fn adamw_step<T: Triton, const BLOCK_SIZE: i32>(
    params_ptr: T::Pointer<f32>,
    grad_ptr: T::Pointer<f32>,
    exp_avg_ptr: T::Pointer<f32>,
    exp_avg_sq_ptr: T::Pointer<f32>,
    n_elements: i32,
    step_size: f32,
    bias_corr2_sqrt: f32,
    beta1: f32,
    beta2: f32,
    eps: f32,
    weight_decay: f32,
    lr: f32,
) where
    T::I32Tensor: types::Tensor<i32, 1>,
    T::I32Tensor: Comparison<i32, BoolTensor = T::BoolTensor>,
    T::Pointer<f32>: AddOffsets<i32, 1, T::I32Tensor, Output = T::Tensor<T::Pointer<f32>>>,
{
    let pid = T::program_id(Axis::X);
    let block_start = pid * BLOCK_SIZE;
    let offsets = T::arange(0, BLOCK_SIZE) + block_start;
    let mask = offsets.lt(n_elements);

    let p          = T::load(params_ptr.add_offsets(offsets),     Some(mask), None, &[], None, None, None, false);
    let g          = T::load(grad_ptr.add_offsets(offsets),       Some(mask), None, &[], None, None, None, false);
    let exp_avg    = T::load(exp_avg_ptr.add_offsets(offsets),    Some(mask), None, &[], None, None, None, false);
    let exp_avg_sq = T::load(exp_avg_sq_ptr.add_offsets(offsets), Some(mask), None, &[], None, None, None, false);

    let beta1_t     = T::full(&[BLOCK_SIZE], beta1);
    let beta2_t     = T::full(&[BLOCK_SIZE], beta2);
    let one_m_beta1 = T::full(&[BLOCK_SIZE], 1.0_f32 - beta1);
    let one_m_beta2 = T::full(&[BLOCK_SIZE], 1.0_f32 - beta2);
    let eps_t       = T::full(&[BLOCK_SIZE], eps);
    let step_size_t = T::full(&[BLOCK_SIZE], step_size);
    let bc2sqrt_t   = T::full(&[BLOCK_SIZE], bias_corr2_sqrt);
    let wd_decay    = T::full(&[BLOCK_SIZE], 1.0_f32 - lr * weight_decay);

    // Decoupled weight decay applied directly to params
    let p_decayed = p * wd_decay;

    let exp_avg_new    = beta1_t * exp_avg    + one_m_beta1 * g;
    let exp_avg_sq_new = beta2_t * exp_avg_sq + one_m_beta2 * g * g;

    let half = T::full(&[BLOCK_SIZE], 0.5_f32);
    let denom = T::exp(half * T::log(exp_avg_sq_new)) / bc2sqrt_t + eps_t;
    let p_new = p_decayed - step_size_t * exp_avg_new / denom;

    T::store(params_ptr.add_offsets(offsets),     p_new,          Some(mask), &[], None, None);
    T::store(exp_avg_ptr.add_offsets(offsets),    exp_avg_new,    Some(mask), &[], None, None);
    T::store(exp_avg_sq_ptr.add_offsets(offsets), exp_avg_sq_new, Some(mask), &[], None, None);
}
