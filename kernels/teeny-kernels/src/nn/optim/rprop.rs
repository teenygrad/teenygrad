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

/// Rprop step (resilient backpropagation).
///
/// ```text
/// sign      = g * prev_g          (product of current and previous gradient)
/// step_size = step_size * eta_plus   if sign > 0
///           = step_size * eta_minus  if sign < 0
///           = step_size              otherwise
/// step_size = clamp(step_size, step_min, step_max)
/// g_masked  = 0                    if sign < 0 (gradient reversal: skip update)
///           = g                    otherwise
/// p        -= sign(g_masked) * step_size
/// prev_g    = g_masked
/// ```
///
/// Grid: `[ceil(n_elements / BLOCK_SIZE), 1, 1]`.
#[kernel]
pub fn rprop_step<T: Triton, const BLOCK_SIZE: i32>(
    params_ptr: T::Pointer<f32>,
    grad_ptr: T::Pointer<f32>,
    prev_grad_ptr: T::Pointer<f32>,
    step_size_ptr: T::Pointer<f32>,
    n_elements: i32,
    eta_plus: f32,
    eta_minus: f32,
    step_min: f32,
    step_max: f32,
) where
    T::I32Tensor: types::Tensor<i32, 1>,
    T::I32Tensor: Comparison<i32, BoolTensor = T::BoolTensor>,
    T::Pointer<f32>: AddOffsets<i32, 1, T::I32Tensor, Output = T::Tensor<T::Pointer<f32>>>,
{
    let pid = T::program_id(Axis::X);
    let block_start = pid * BLOCK_SIZE;
    let offsets = T::arange(0, BLOCK_SIZE) + block_start;
    let mask = offsets.lt(n_elements);

    let p         = T::load(params_ptr.add_offsets(offsets),    Some(mask), None, &[], None, None, None, false);
    let g         = T::load(grad_ptr.add_offsets(offsets),      Some(mask), None, &[], None, None, None, false);
    let prev_g    = T::load(prev_grad_ptr.add_offsets(offsets), Some(mask), None, &[], None, None, None, false);
    let step_size = T::load(step_size_ptr.add_offsets(offsets), Some(mask), None, &[], None, None, None, false);

    let zeros      = T::zeros::<f32>(&[BLOCK_SIZE]);
    let ones       = T::full(&[BLOCK_SIZE], 1.0_f32);
    let neg_ones   = T::full(&[BLOCK_SIZE], -1.0_f32);
    let eta_plus_t  = T::full(&[BLOCK_SIZE], eta_plus);
    let eta_minus_t = T::full(&[BLOCK_SIZE], eta_minus);
    let step_min_t  = T::full(&[BLOCK_SIZE], step_min);
    let step_max_t  = T::full(&[BLOCK_SIZE], step_max);

    let prod      = g * prev_g;
    let sign_pos  = T::gt(prod, zeros);   // sign > 0: same direction
    let sign_neg  = T::lt(prod, zeros);   // sign < 0: reversal

    // Scale step size based on gradient sign agreement
    let step_after_pos = T::where_(sign_pos, step_size * eta_plus_t, step_size);
    let step_scaled    = T::where_(sign_neg, step_after_pos * eta_minus_t, step_after_pos);
    let step_clamped   = T::clamp(step_scaled, step_min_t, step_max_t);

    // Mask gradient to zero when sign reversal (backtracking: skip update this step)
    let g_masked = T::where_(sign_neg, zeros, g);

    // Update: p -= sign(g_masked) * step_size
    let g_pos     = T::gt(g_masked, zeros);
    let g_neg     = T::lt(g_masked, zeros);
    let g_sign    = T::where_(g_pos, ones, T::where_(g_neg, neg_ones, zeros));
    let p_new     = p - g_sign * step_clamped;

    T::store(params_ptr.add_offsets(offsets),    p_new,       Some(mask), &[], None, None);
    T::store(step_size_ptr.add_offsets(offsets), step_clamped, Some(mask), &[], None, None);
    T::store(prev_grad_ptr.add_offsets(offsets), g_masked,    Some(mask), &[], None, None);
}
