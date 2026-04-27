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

/// Adamax step (infinity-norm variant of Adam).
///
/// ```text
/// exp_avg = beta1 * exp_avg + (1 - beta1) * g
/// exp_inf = max(beta2 * exp_inf, |g| + eps)
/// p       = p - clr * exp_avg / exp_inf
/// ```
///
/// `clr = lr / (1 - beta1^t)` is precomputed on the host.
///
/// Grid: `[ceil(n_elements / BLOCK_SIZE), 1, 1]`.
#[kernel]
pub fn adamax_step<T: Triton, const BLOCK_SIZE: i32>(
    params_ptr: T::Pointer<f32>,
    grad_ptr: T::Pointer<f32>,
    exp_avg_ptr: T::Pointer<f32>,
    exp_inf_ptr: T::Pointer<f32>,
    n_elements: i32,
    clr: f32,
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

    let p       = T::load(params_ptr.add_offsets(offsets),  Some(mask), None, &[], None, None, None, false);
    let g       = T::load(grad_ptr.add_offsets(offsets),    Some(mask), None, &[], None, None, None, false);
    let exp_avg = T::load(exp_avg_ptr.add_offsets(offsets), Some(mask), None, &[], None, None, None, false);
    let exp_inf = T::load(exp_inf_ptr.add_offsets(offsets), Some(mask), None, &[], None, None, None, false);

    let beta1_t     = T::full(&[BLOCK_SIZE], beta1);
    let one_m_beta1 = T::full(&[BLOCK_SIZE], 1.0_f32 - beta1);
    let beta2_t     = T::full(&[BLOCK_SIZE], beta2);
    let eps_t       = T::full(&[BLOCK_SIZE], eps);
    let clr_t       = T::full(&[BLOCK_SIZE], clr);
    let wd_t        = T::full(&[BLOCK_SIZE], weight_decay);

    let g_eff       = g + wd_t * p;
    let exp_avg_new = beta1_t * exp_avg + one_m_beta1 * g_eff;
    // exp_inf = max(beta2 * exp_inf, |g| + eps)
    let exp_inf_new = T::maximum(beta2_t * exp_inf, T::abs(g_eff) + eps_t);
    let p_new       = p - clr_t * exp_avg_new / exp_inf_new;

    T::store(params_ptr.add_offsets(offsets),  p_new,       Some(mask), &[], None, None);
    T::store(exp_avg_ptr.add_offsets(offsets), exp_avg_new, Some(mask), &[], None, None);
    T::store(exp_inf_ptr.add_offsets(offsets), exp_inf_new, Some(mask), &[], None, None);
}
