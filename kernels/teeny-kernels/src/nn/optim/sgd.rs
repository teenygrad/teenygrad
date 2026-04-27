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

/// SGD step (no momentum).
///
/// ```text
/// p = p - lr * (g + weight_decay * p)
/// ```
///
/// Grid: `[ceil(n_elements / BLOCK_SIZE), 1, 1]`.
#[kernel]
pub fn sgd_step<T: Triton, const BLOCK_SIZE: i32>(
    params_ptr: T::Pointer<f32>,
    grad_ptr: T::Pointer<f32>,
    n_elements: i32,
    lr: f32,
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

    let p = T::load(params_ptr.add_offsets(offsets), Some(mask), None, &[], None, None, None, false);
    let g = T::load(grad_ptr.add_offsets(offsets),   Some(mask), None, &[], None, None, None, false);

    let lr_t = T::full(&[BLOCK_SIZE], lr);
    let wd_t = T::full(&[BLOCK_SIZE], weight_decay);

    let p_new = p - lr_t * (g + wd_t * p);
    T::store(params_ptr.add_offsets(offsets), p_new, Some(mask), &[], None, None);
}

/// SGD step with momentum (non-Nesterov).
///
/// ```text
/// buf = momentum * buf + (1 - dampening) * (g + weight_decay * p)
/// p   = p - lr * buf
/// ```
///
/// Grid: `[ceil(n_elements / BLOCK_SIZE), 1, 1]`.
#[kernel]
pub fn sgd_momentum_step<T: Triton, const BLOCK_SIZE: i32>(
    params_ptr: T::Pointer<f32>,
    grad_ptr: T::Pointer<f32>,
    buf_ptr: T::Pointer<f32>,
    n_elements: i32,
    lr: f32,
    momentum: f32,
    dampening: f32,
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

    let p   = T::load(params_ptr.add_offsets(offsets), Some(mask), None, &[], None, None, None, false);
    let g   = T::load(grad_ptr.add_offsets(offsets),   Some(mask), None, &[], None, None, None, false);
    let buf = T::load(buf_ptr.add_offsets(offsets),    Some(mask), None, &[], None, None, None, false);

    let lr_t   = T::full(&[BLOCK_SIZE], lr);
    let mu_t   = T::full(&[BLOCK_SIZE], momentum);
    let damp_t = T::full(&[BLOCK_SIZE], 1.0_f32 - dampening);
    let wd_t   = T::full(&[BLOCK_SIZE], weight_decay);

    let g_eff   = g + wd_t * p;
    let buf_new = mu_t * buf + damp_t * g_eff;
    let p_new   = p - lr_t * buf_new;

    T::store(params_ptr.add_offsets(offsets), p_new,   Some(mask), &[], None, None);
    T::store(buf_ptr.add_offsets(offsets),    buf_new, Some(mask), &[], None, None);
}

/// SGD step with Nesterov momentum.
///
/// ```text
/// buf   = momentum * buf + (1 - dampening) * (g + weight_decay * p)
/// g_nes = (g + weight_decay * p) + momentum * buf
/// p     = p - lr * g_nes
/// ```
///
/// Grid: `[ceil(n_elements / BLOCK_SIZE), 1, 1]`.
#[kernel]
pub fn sgd_nesterov_step<T: Triton, const BLOCK_SIZE: i32>(
    params_ptr: T::Pointer<f32>,
    grad_ptr: T::Pointer<f32>,
    buf_ptr: T::Pointer<f32>,
    n_elements: i32,
    lr: f32,
    momentum: f32,
    dampening: f32,
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

    let p   = T::load(params_ptr.add_offsets(offsets), Some(mask), None, &[], None, None, None, false);
    let g   = T::load(grad_ptr.add_offsets(offsets),   Some(mask), None, &[], None, None, None, false);
    let buf = T::load(buf_ptr.add_offsets(offsets),    Some(mask), None, &[], None, None, None, false);

    let lr_t   = T::full(&[BLOCK_SIZE], lr);
    let mu_t   = T::full(&[BLOCK_SIZE], momentum);
    let damp_t = T::full(&[BLOCK_SIZE], 1.0_f32 - dampening);
    let wd_t   = T::full(&[BLOCK_SIZE], weight_decay);

    let g_eff   = g + wd_t * p;
    let buf_new = mu_t * buf + damp_t * g_eff;
    let g_nes   = g_eff + mu_t * buf_new;
    let p_new   = p - lr_t * g_nes;

    T::store(params_ptr.add_offsets(offsets), p_new,   Some(mask), &[], None, None);
    T::store(buf_ptr.add_offsets(offsets),    buf_new, Some(mask), &[], None, None);
}
