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

/// ASGD step (Averaged SGD) — averaging-active phase (step > t0).
///
/// ```text
/// p  = p - lr * g
/// ax = ax + (p - ax) / d_ax        where d_ax = max(1, step - t0)
/// ```
///
/// `d_ax = max(1.0, step - t0)` is precomputed on the host.
///
/// Grid: `[ceil(n_elements / BLOCK_SIZE), 1, 1]`.
#[kernel]
pub fn asgd_step<T: Triton, const BLOCK_SIZE: i32>(
    params_ptr: T::Pointer<f32>,
    grad_ptr: T::Pointer<f32>,
    ax_ptr: T::Pointer<f32>,
    n_elements: i32,
    lr: f32,
    weight_decay: f32,
    d_ax: f32,
) where
    T::I32Tensor: types::Tensor<i32, 1>,
    T::I32Tensor: Comparison<i32, BoolTensor = T::BoolTensor>,
    T::Pointer<f32>: AddOffsets<i32, 1, T::I32Tensor, Output = T::Tensor<T::Pointer<f32>>>,
{
    let pid = T::program_id(Axis::X);
    let block_start = pid * BLOCK_SIZE;
    let offsets = T::arange(0, BLOCK_SIZE) + block_start;
    let mask = offsets.lt(n_elements);

    let p  = T::load(params_ptr.add_offsets(offsets), Some(mask), None, &[], None, None, None, false);
    let g  = T::load(grad_ptr.add_offsets(offsets),   Some(mask), None, &[], None, None, None, false);
    let ax = T::load(ax_ptr.add_offsets(offsets),     Some(mask), None, &[], None, None, None, false);

    let lr_t   = T::full(&[BLOCK_SIZE], lr);
    let wd_t   = T::full(&[BLOCK_SIZE], weight_decay);
    let d_ax_t = T::full(&[BLOCK_SIZE], d_ax);

    let g_eff  = g + wd_t * p;
    let p_new  = p - lr_t * g_eff;
    let ax_new = ax + (p_new - ax) / d_ax_t;

    T::store(params_ptr.add_offsets(offsets), p_new,  Some(mask), &[], None, None);
    T::store(ax_ptr.add_offsets(offsets),     ax_new, Some(mask), &[], None, None);
}
