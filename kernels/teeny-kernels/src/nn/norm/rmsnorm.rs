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

//! RMSNorm Triton kernels.
//!
//! RMSNorm normalises each row by its root-mean-square (no mean subtraction):
//!   rms[m]    = sqrt( (1/N) * Σ_n x[m,n]² + eps )
//!   y[m,n]    = x[m,n] / rms[m] * γ[n]
//!
//! Grid: `[M]` — one CTA per row. Layout identical to LayerNorm.

#![allow(non_snake_case)]

use teeny_core::dtype::Float;
use teeny_macros::kernel;
use teeny_triton::triton::{
    types::{AddOffsets, Comparison},
    *,
};

// ─── Forward ─────────────────────────────────────────────────────────────────

/// RMSNorm forward pass.
///
/// Grid: `[M]` — one CTA per row.
#[kernel]
pub fn rms_norm_forward<T: Triton, D: Float, const BLOCK_N: i32>(
    x_ptr: T::Pointer<D>,
    y_ptr: T::Pointer<D>,
    weight_ptr: T::Pointer<D>,
    rrms_ptr: T::Pointer<D>,
    M: i32,
    N: i32,
    eps: f32,
) where
    T::I32Tensor: types::Tensor<i32, 1>,
    T::I32Tensor: Comparison<i32, BoolTensor = T::BoolTensor>,
    T::Pointer<D>: AddOffsets<i32, 1, T::I32Tensor, Output = T::Tensor<T::Pointer<D>>>,
{
    let row = T::program_id(Axis::X);
    let row_start = row * N;
    let row_idx = T::arange(0, 1) + row;

    let zeros = T::zeros::<D>(&[BLOCK_N]);
    let zero_1 = T::zeros::<D>(&[1]);
    let n_inv = T::cast::<f32, D>(T::full::<f32>(&[1], 1.0f32 / (N as f32)), None, false);

    // ── Pass 1: accumulate Σ x² ──────────────────────────────────────────────
    let mut sq_sum = zero_1;
    let mut n_start: i32 = 0;
    while n_start < N {
        let col_offs = T::arange(0, BLOCK_N) + n_start;
        let mask = col_offs.lt(N);
        let x_tile = T::load(
            x_ptr.add_offsets(col_offs + row_start),
            Some(mask),
            Some(zeros),
            &[],
            None,
            None,
            None,
            false,
        );
        sq_sum = sq_sum + T::sum(x_tile * x_tile, None, true);
        n_start += BLOCK_N;
    }
    let eps_t = T::cast::<f32, D>(T::full::<f32>(&[1], eps), None, false);
    let rrms_1 = T::rsqrt(sq_sum * n_inv + eps_t);
    let rrms = T::broadcast_to(rrms_1, &[BLOCK_N]);

    T::store(rrms_ptr.add_offsets(row_idx), rrms_1, None, &[], None, None);

    // ── Pass 2: normalise ─────────────────────────────────────────────────────
    n_start = 0;
    while n_start < N {
        let col_offs = T::arange(0, BLOCK_N) + n_start;
        let mask = col_offs.lt(N);
        let x_tile = T::load(
            x_ptr.add_offsets(col_offs + row_start),
            Some(mask),
            Some(zeros),
            &[],
            None,
            None,
            None,
            false,
        );
        let gamma = T::load(weight_ptr.add_offsets(col_offs), Some(mask), Some(zeros), &[], None, None, None, false);
        let y_tile = x_tile * rrms * gamma;
        T::store(y_ptr.add_offsets(col_offs + row_start), y_tile, Some(mask), &[], None, None);
        n_start += BLOCK_N;
    }
}

// ─── Backward ────────────────────────────────────────────────────────────────

/// RMSNorm backward pass.
///
/// ```text
/// dx[m,n] = rrms[m] * γ[n] * (dy[m,n] - x[m,n] * rrms[m]² * Σ_n dy[m,n]*γ[n]*x[m,n] / N)
/// dweight[n] = Σ_m dy[m,n] * x[m,n] * rrms[m]
/// ```
///
/// Grid: `[M]` — one CTA per row.
#[cfg(feature = "training")]
#[kernel]
pub fn rms_norm_backward<T: Triton, D: Float, const BLOCK_N: i32>(
    dy_ptr: T::Pointer<D>,
    x_ptr: T::Pointer<D>,
    dx_ptr: T::Pointer<D>,
    weight_ptr: T::Pointer<D>,
    dweight_ptr: T::Pointer<D>,
    rrms_ptr: T::Pointer<D>,
    M: i32,
    N: i32,
) where
    T::I32Tensor: types::Tensor<i32, 1>,
    T::I32Tensor: Comparison<i32, BoolTensor = T::BoolTensor>,
    T::Pointer<D>: AddOffsets<i32, 1, T::I32Tensor, Output = T::Tensor<T::Pointer<D>>>,
{
    let row = T::program_id(Axis::X);
    let row_start = row * N;
    let row_idx = T::arange(0, 1) + row;

    let zeros = T::zeros::<D>(&[BLOCK_N]);
    let zero_1 = T::zeros::<D>(&[1]);
    let n_inv = T::cast::<f32, D>(T::full::<f32>(&[1], 1.0f32 / (N as f32)), None, false);

    let rrms_1 = T::load(rrms_ptr.add_offsets(row_idx), None, None, &[], None, None, None, false);
    let rrms = T::broadcast_to(rrms_1, &[BLOCK_N]);

    // ── Pass 1: Σ dy * γ * x ─────────────────────────────────────────────────
    let mut dot = zero_1;
    let mut n_start: i32 = 0;
    while n_start < N {
        let col_offs = T::arange(0, BLOCK_N) + n_start;
        let mask = col_offs.lt(N);
        let x_tile = T::load(
            x_ptr.add_offsets(col_offs + row_start),
            Some(mask),
            Some(zeros),
            &[],
            None,
            None,
            None,
            false,
        );
        let dy_tile = T::load(
            dy_ptr.add_offsets(col_offs + row_start),
            Some(mask),
            Some(zeros),
            &[],
            None,
            None,
            None,
            false,
        );
        let gamma = T::load(weight_ptr.add_offsets(col_offs), Some(mask), Some(zeros), &[], None, None, None, false);
        dot = dot + T::sum(dy_tile * gamma * x_tile, None, true);
        n_start += BLOCK_N;
    }
    let rrms_sq = T::broadcast_to(rrms_1 * rrms_1, &[BLOCK_N]);
    let scale = T::broadcast_to(dot * n_inv, &[BLOCK_N]);

    // ── Pass 2: dx and dweight ────────────────────────────────────────────────
    n_start = 0;
    while n_start < N {
        let col_offs = T::arange(0, BLOCK_N) + n_start;
        let mask = col_offs.lt(N);
        let x_tile = T::load(
            x_ptr.add_offsets(col_offs + row_start),
            Some(mask),
            Some(zeros),
            &[],
            None,
            None,
            None,
            false,
        );
        let dy_tile = T::load(
            dy_ptr.add_offsets(col_offs + row_start),
            Some(mask),
            Some(zeros),
            &[],
            None,
            None,
            None,
            false,
        );
        let gamma = T::load(weight_ptr.add_offsets(col_offs), Some(mask), Some(zeros), &[], None, None, None, false);
        let dw_old = T::load(dweight_ptr.add_offsets(col_offs), Some(mask), Some(zeros), &[], None, None, None, false);

        let dx_tile = rrms * gamma * (dy_tile - x_tile * rrms_sq * scale);
        T::store(dx_ptr.add_offsets(col_offs + row_start), dx_tile, Some(mask), &[], None, None);
        T::store(dweight_ptr.add_offsets(col_offs), dw_old + dy_tile * x_tile * rrms, Some(mask), &[], None, None);
        n_start += BLOCK_N;
    }
}
