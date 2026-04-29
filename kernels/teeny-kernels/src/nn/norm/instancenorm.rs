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

//! InstanceNorm Triton kernels.
//!
//! InstanceNorm normalises over the spatial dimensions (L) independently per
//! sample (n) and per channel (c):
//!
//!   y[n,c,l] = (x[n,c,l] - mean[n,c]) / sqrt(var[n,c] + eps) * γ[c] + β[c]
//!
//! Input shape: `[N, C, L]` — N batch, C channels, L spatial elements.
//! Grid: `[N * C]` — one CTA per (sample, channel) pair.
//! The CTA index encodes the pair as `cta = n * C + c`.

#![allow(non_snake_case)]

use teeny_core::dtype::Float;
use teeny_macros::kernel;
use teeny_triton::triton::{
    types::{AddOffsets, Comparison},
    *,
};

// ─── Inference ───────────────────────────────────────────────────────────────

/// InstanceNorm forward (inference — no running stats).
///
/// Grid: `[N * C]` — one CTA per (sample, channel).
#[kernel]
pub fn instance_norm_forward_inference<T: Triton, D: Float, const BLOCK_L: i32>(
    x_ptr: T::Pointer<D>,
    y_ptr: T::Pointer<D>,
    weight_ptr: T::Pointer<D>,
    bias_ptr: T::Pointer<D>,
    N: i32,
    C: i32,
    L: i32,
    eps: f32,
) where
    T::I32Tensor: types::Tensor<i32, 1>,
    T::I32Tensor: Comparison<i32, BoolTensor = T::BoolTensor>,
    T::Pointer<D>: AddOffsets<i32, 1, T::I32Tensor, Output = T::Tensor<T::Pointer<D>>>,
{
    let pid = T::program_id(Axis::X);
    let n = pid / C;
    let c = pid - n * C;
    let row_start = (n * C + c) * L;

    let c_idx = T::arange(0, 1) + c;
    let zeros = T::zeros::<D>(&[BLOCK_L]);
    let zero_1 = T::zeros::<D>(&[1]);
    let l_inv = T::cast::<f32, D>(T::full::<f32>(&[1], 1.0f32 / (L as f32)), None, false);

    // ── Pass 1: mean ─────────────────────────────────────────────────────────
    let mut sum = zero_1;
    let mut l_start: i32 = 0;
    while l_start < L {
        let col_offs = T::arange(0, BLOCK_L) + l_start;
        let mask = col_offs.lt(L);
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
        sum = sum + T::sum(x_tile, None, true);
        l_start += BLOCK_L;
    }
    let mean_1 = sum * l_inv;
    let mean = T::broadcast_to(mean_1, &[BLOCK_L]);

    // ── Pass 2: variance ─────────────────────────────────────────────────────
    let mut var_sum = zero_1;
    l_start = 0;
    while l_start < L {
        let col_offs = T::arange(0, BLOCK_L) + l_start;
        let mask = col_offs.lt(L);
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
        // Mask the diff so out-of-bounds positions don't contribute mean^2 to variance.
        let diff = T::where_::<D>(mask, x_tile - mean, zeros);
        var_sum = var_sum + T::sum(diff * diff, None, true);
        l_start += BLOCK_L;
    }
    let eps_t = T::cast::<f32, D>(T::full::<f32>(&[1], eps), None, false);
    let rstd = T::broadcast_to(T::rsqrt(var_sum * l_inv + eps_t), &[BLOCK_L]);

    let gamma = T::broadcast_to(
        T::load(weight_ptr.add_offsets(c_idx), None, None, &[], None, None, None, false),
        &[BLOCK_L],
    );
    let beta = T::broadcast_to(
        T::load(bias_ptr.add_offsets(c_idx), None, None, &[], None, None, None, false),
        &[BLOCK_L],
    );

    // ── Pass 3: normalise ─────────────────────────────────────────────────────
    l_start = 0;
    while l_start < L {
        let col_offs = T::arange(0, BLOCK_L) + l_start;
        let mask = col_offs.lt(L);
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
        let y_tile = (x_tile - mean) * rstd * gamma + beta;
        T::store(y_ptr.add_offsets(col_offs + row_start), y_tile, Some(mask), &[], None, None);
        l_start += BLOCK_L;
    }
}

// ─── Training forward ─────────────────────────────────────────────────────────

/// InstanceNorm training forward — saves per-(n,c) mean and rstd.
///
/// Grid: `[N * C]` — one CTA per (sample, channel).
#[cfg(feature = "training")]
#[kernel]
pub fn instance_norm_forward<T: Triton, D: Float, const BLOCK_L: i32>(
    x_ptr: T::Pointer<D>,
    y_ptr: T::Pointer<D>,
    weight_ptr: T::Pointer<D>,
    bias_ptr: T::Pointer<D>,
    mean_ptr: T::Pointer<D>,
    rstd_ptr: T::Pointer<D>,
    N: i32,
    C: i32,
    L: i32,
    eps: f32,
) where
    T::I32Tensor: types::Tensor<i32, 1>,
    T::I32Tensor: Comparison<i32, BoolTensor = T::BoolTensor>,
    T::Pointer<D>: AddOffsets<i32, 1, T::I32Tensor, Output = T::Tensor<T::Pointer<D>>>,
{
    let pid = T::program_id(Axis::X);
    let n = pid / C;
    let c = pid - n * C;
    let row_start = (n * C + c) * L;
    let stat_idx = T::arange(0, 1) + pid;
    let c_idx = T::arange(0, 1) + c;

    let zeros = T::zeros::<D>(&[BLOCK_L]);
    let zero_1 = T::zeros::<D>(&[1]);
    let l_inv = T::cast::<f32, D>(T::full::<f32>(&[1], 1.0f32 / (L as f32)), None, false);

    // ── Pass 1: mean ─────────────────────────────────────────────────────────
    let mut sum = zero_1;
    let mut l_start: i32 = 0;
    while l_start < L {
        let col_offs = T::arange(0, BLOCK_L) + l_start;
        let mask = col_offs.lt(L);
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
        sum = sum + T::sum(x_tile, None, true);
        l_start += BLOCK_L;
    }
    let mean_1 = sum * l_inv;
    let mean = T::broadcast_to(mean_1, &[BLOCK_L]);

    // ── Pass 2: variance ─────────────────────────────────────────────────────
    let mut var_sum = zero_1;
    l_start = 0;
    while l_start < L {
        let col_offs = T::arange(0, BLOCK_L) + l_start;
        let mask = col_offs.lt(L);
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
        // Mask the diff so out-of-bounds positions don't contribute mean^2 to variance.
        let diff = T::where_::<D>(mask, x_tile - mean, zeros);
        var_sum = var_sum + T::sum(diff * diff, None, true);
        l_start += BLOCK_L;
    }
    let eps_t = T::cast::<f32, D>(T::full::<f32>(&[1], eps), None, false);
    let rstd_1 = T::rsqrt(var_sum * l_inv + eps_t);
    let rstd = T::broadcast_to(rstd_1, &[BLOCK_L]);

    T::store(mean_ptr.add_offsets(stat_idx), mean_1, None, &[], None, None);
    T::store(rstd_ptr.add_offsets(stat_idx), rstd_1, None, &[], None, None);

    let gamma = T::broadcast_to(
        T::load(weight_ptr.add_offsets(c_idx), None, None, &[], None, None, None, false),
        &[BLOCK_L],
    );
    let beta = T::broadcast_to(
        T::load(bias_ptr.add_offsets(c_idx), None, None, &[], None, None, None, false),
        &[BLOCK_L],
    );

    // ── Pass 3: normalise ─────────────────────────────────────────────────────
    l_start = 0;
    while l_start < L {
        let col_offs = T::arange(0, BLOCK_L) + l_start;
        let mask = col_offs.lt(L);
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
        let y_tile = (x_tile - mean) * rstd * gamma + beta;
        T::store(y_ptr.add_offsets(col_offs + row_start), y_tile, Some(mask), &[], None, None);
        l_start += BLOCK_L;
    }
}

// ─── Training backward ───────────────────────────────────────────────────────

/// InstanceNorm backward pass.
///
/// Grid: `[N * C]` — one CTA per (sample, channel).
#[cfg(feature = "training")]
#[kernel]
pub fn instance_norm_backward<T: Triton, D: Float, const BLOCK_L: i32>(
    dy_ptr: T::Pointer<D>,
    x_ptr: T::Pointer<D>,
    dx_ptr: T::Pointer<D>,
    weight_ptr: T::Pointer<D>,
    dweight_ptr: T::Pointer<D>,
    dbias_ptr: T::Pointer<D>,
    mean_ptr: T::Pointer<D>,
    rstd_ptr: T::Pointer<D>,
    N: i32,
    C: i32,
    L: i32,
) where
    T::I32Tensor: types::Tensor<i32, 1>,
    T::I32Tensor: Comparison<i32, BoolTensor = T::BoolTensor>,
    T::Pointer<D>: AddOffsets<i32, 1, T::I32Tensor, Output = T::Tensor<T::Pointer<D>>>,
{
    let pid = T::program_id(Axis::X);
    let n = pid / C;
    let c = pid - n * C;
    let row_start = (n * C + c) * L;
    let stat_idx = T::arange(0, 1) + pid;
    let c_idx = T::arange(0, 1) + c;

    let zeros = T::zeros::<D>(&[BLOCK_L]);
    let zero_1 = T::zeros::<D>(&[1]);
    let l_inv = T::cast::<f32, D>(T::full::<f32>(&[1], 1.0f32 / (L as f32)), None, false);

    let rstd_1 = T::load(rstd_ptr.add_offsets(stat_idx), None, None, &[], None, None, None, false);
    let mean_1 = T::load(mean_ptr.add_offsets(stat_idx), None, None, &[], None, None, None, false);
    let rstd = T::broadcast_to(rstd_1, &[BLOCK_L]);
    let mean = T::broadcast_to(mean_1, &[BLOCK_L]);

    let gamma = T::broadcast_to(
        T::load(weight_ptr.add_offsets(c_idx), None, None, &[], None, None, None, false),
        &[BLOCK_L],
    );

    // ── Pass 1: accumulate row dot products ───────────────────────────────────
    let mut sum_dy_gamma = zero_1;
    let mut sum_dy_gamma_xhat = zero_1;
    let mut l_start: i32 = 0;
    while l_start < L {
        let col_offs = T::arange(0, BLOCK_L) + l_start;
        let mask = col_offs.lt(L);
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
        let xhat = (x_tile - mean) * rstd;
        sum_dy_gamma = sum_dy_gamma + T::sum(dy_tile * gamma, None, true);
        sum_dy_gamma_xhat = sum_dy_gamma_xhat + T::sum(dy_tile * gamma * xhat, None, true);
        l_start += BLOCK_L;
    }
    let c1 = T::broadcast_to(sum_dy_gamma * l_inv, &[BLOCK_L]);
    let c2 = T::broadcast_to(sum_dy_gamma_xhat * l_inv, &[BLOCK_L]);

    // ── Pass 2: dx and dweight / dbias ───────────────────────────────────────
    l_start = 0;
    while l_start < L {
        let col_offs = T::arange(0, BLOCK_L) + l_start;
        let mask = col_offs.lt(L);
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
        let dw_old = T::load(dweight_ptr.add_offsets(c_idx), None, None, &[], None, None, None, false);
        let db_old = T::load(dbias_ptr.add_offsets(c_idx), None, None, &[], None, None, None, false);

        let xhat = (x_tile - mean) * rstd;
        let dx_tile = rstd * gamma * (dy_tile - c1 - xhat * c2);

        T::store(dx_ptr.add_offsets(col_offs + row_start), dx_tile, Some(mask), &[], None, None);
        T::store(dweight_ptr.add_offsets(c_idx), dw_old + T::sum(dy_tile * xhat, None, true), None, &[], None, None);
        T::store(dbias_ptr.add_offsets(c_idx), db_old + T::sum(dy_tile, None, true), None, &[], None, None);
        l_start += BLOCK_L;
    }
}
