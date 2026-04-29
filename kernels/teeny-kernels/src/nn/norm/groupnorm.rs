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

//! GroupNorm Triton kernels.
//!
//! GroupNorm partitions C channels into G groups and normalises over each
//! group independently, per sample:
//!
//!   y[n,c,l] = (x[n,c,l] - mean[n,g]) / sqrt(var[n,g] + eps) * γ[c] + β[c]
//!
//! where g = c / (C / G) is the group index.
//!
//! Input shape: `[N, C, L]`. Grid: `[N * G]` — one CTA per (sample, group).
//! Each CTA covers channels `[g*(C/G), (g+1)*(C/G))` × all L elements,
//! so the normalised tile has `(C/G) * L` elements.
//!
//! BLOCK_NL must be >= (C/G) * L and a power of two.

#![allow(non_snake_case)]

use teeny_core::dtype::Float;
use teeny_macros::kernel;
use teeny_triton::triton::{
    types::{AddOffsets, Comparison},
    *,
};

// ─── Inference ───────────────────────────────────────────────────────────────

/// GroupNorm forward (inference).
///
/// Grid: `[N * G]` — one CTA per (sample, group).
#[kernel]
pub fn group_norm_forward_inference<T: Triton, D: Float, const BLOCK_NL: i32>(
    x_ptr: T::Pointer<D>,
    y_ptr: T::Pointer<D>,
    weight_ptr: T::Pointer<D>,
    bias_ptr: T::Pointer<D>,
    N: i32,
    C: i32,
    L: i32,
    G: i32,
    eps: f32,
) where
    T::I32Tensor: types::Tensor<i32, 1>,
    T::I32Tensor: Comparison<i32, BoolTensor = T::BoolTensor>,
    T::Pointer<D>: AddOffsets<i32, 1, T::I32Tensor, Output = T::Tensor<T::Pointer<D>>>,
{
    let pid = T::program_id(Axis::X);
    let n = pid / G;
    let g = pid - n * G;
    let channels_per_group = C / G;

    // Flat element count for this (n, g) group: channels_per_group * L.
    let group_size = channels_per_group * L;
    // Offset to the first element of this group in the flat [N, C, L] layout.
    let group_start = n * C * L + g * channels_per_group * L;

    let zeros = T::zeros::<D>(&[BLOCK_NL]);
    let zero_1 = T::zeros::<D>(&[1]);
    let gs_inv = T::cast::<f32, D>(T::full::<f32>(&[1], 1.0f32 / (group_size as f32)), None, false);

    // ── Pass 1: mean ─────────────────────────────────────────────────────────
    let mut sum = zero_1;
    let mut t: i32 = 0;
    while t < group_size {
        let offs = T::arange(0, BLOCK_NL) + t;
        let mask = offs.lt(group_size);
        let x_tile = T::load(
            x_ptr.add_offsets(offs + group_start),
            Some(mask),
            Some(zeros),
            &[],
            None,
            None,
            None,
            false,
        );
        sum = sum + T::sum(x_tile, None, true);
        t += BLOCK_NL;
    }
    let mean_1 = sum * gs_inv;
    let mean = T::broadcast_to(mean_1, &[BLOCK_NL]);

    // ── Pass 2: variance ─────────────────────────────────────────────────────
    let mut var_sum = zero_1;
    t = 0;
    while t < group_size {
        let offs = T::arange(0, BLOCK_NL) + t;
        let mask = offs.lt(group_size);
        let x_tile = T::load(
            x_ptr.add_offsets(offs + group_start),
            Some(mask),
            Some(zeros),
            &[],
            None,
            None,
            None,
            false,
        );
        // Mask the diff so out-of-bounds positions (x_tile=0, mean=true_mean)
        // don't contribute (0-mean)^2 = mean^2 to the variance sum.
        let diff = T::where_::<D>(mask, x_tile - mean, zeros);
        var_sum = var_sum + T::sum(diff * diff, None, true);
        t += BLOCK_NL;
    }
    let eps_t = T::cast::<f32, D>(T::full::<f32>(&[1], eps), None, false);
    let rstd = T::broadcast_to(T::rsqrt(var_sum * gs_inv + eps_t), &[BLOCK_NL]);

    // ── Pass 3: normalise with per-channel affine ─────────────────────────────
    // Element at group-flat index (t+i) maps to channel g*cpg + (t+i)/L.
    t = 0;
    while t < group_size {
        let offs = T::arange(0, BLOCK_NL) + t;
        let mask = offs.lt(group_size);
        let chan_offs = (T::arange(0, BLOCK_NL) + t) / L + g * channels_per_group;
        let x_tile = T::load(
            x_ptr.add_offsets(offs + group_start),
            Some(mask),
            Some(zeros),
            &[],
            None,
            None,
            None,
            false,
        );
        let gamma = T::load(weight_ptr.add_offsets(chan_offs), Some(mask), Some(zeros), &[], None, None, None, false);
        let beta = T::load(bias_ptr.add_offsets(chan_offs), Some(mask), Some(zeros), &[], None, None, None, false);
        let y_tile = (x_tile - mean) * rstd * gamma + beta;
        T::store(y_ptr.add_offsets(offs + group_start), y_tile, Some(mask), &[], None, None);
        t += BLOCK_NL;
    }
}

// ─── Training forward ─────────────────────────────────────────────────────────

/// GroupNorm training forward — saves per-(n,g) mean and rstd.
///
/// Grid: `[N * G]` — one CTA per (sample, group).
#[cfg(feature = "training")]
#[kernel]
pub fn group_norm_forward<T: Triton, D: Float, const BLOCK_NL: i32>(
    x_ptr: T::Pointer<D>,
    y_ptr: T::Pointer<D>,
    weight_ptr: T::Pointer<D>,
    bias_ptr: T::Pointer<D>,
    mean_ptr: T::Pointer<D>,
    rstd_ptr: T::Pointer<D>,
    N: i32,
    C: i32,
    L: i32,
    G: i32,
    eps: f32,
) where
    T::I32Tensor: types::Tensor<i32, 1>,
    T::I32Tensor: Comparison<i32, BoolTensor = T::BoolTensor>,
    T::Pointer<D>: AddOffsets<i32, 1, T::I32Tensor, Output = T::Tensor<T::Pointer<D>>>,
{
    let pid = T::program_id(Axis::X);
    let n = pid / G;
    let g = pid - n * G;
    let channels_per_group = C / G;
    let group_size = channels_per_group * L;
    let group_start = n * C * L + g * channels_per_group * L;
    let stat_idx = T::arange(0, 1) + pid;

    let zeros = T::zeros::<D>(&[BLOCK_NL]);
    let zero_1 = T::zeros::<D>(&[1]);
    let gs_inv = T::cast::<f32, D>(T::full::<f32>(&[1], 1.0f32 / (group_size as f32)), None, false);

    // ── Pass 1: mean ─────────────────────────────────────────────────────────
    let mut sum = zero_1;
    let mut t: i32 = 0;
    while t < group_size {
        let offs = T::arange(0, BLOCK_NL) + t;
        let mask = offs.lt(group_size);
        let x_tile = T::load(
            x_ptr.add_offsets(offs + group_start),
            Some(mask),
            Some(zeros),
            &[],
            None,
            None,
            None,
            false,
        );
        sum = sum + T::sum(x_tile, None, true);
        t += BLOCK_NL;
    }
    let mean_1 = sum * gs_inv;
    let mean = T::broadcast_to(mean_1, &[BLOCK_NL]);

    // ── Pass 2: variance ─────────────────────────────────────────────────────
    let mut var_sum = zero_1;
    t = 0;
    while t < group_size {
        let offs = T::arange(0, BLOCK_NL) + t;
        let mask = offs.lt(group_size);
        let x_tile = T::load(
            x_ptr.add_offsets(offs + group_start),
            Some(mask),
            Some(zeros),
            &[],
            None,
            None,
            None,
            false,
        );
        // Mask the diff so out-of-bounds positions (x_tile=0, mean=true_mean)
        // don't contribute (0-mean)^2 = mean^2 to the variance sum.
        let diff = T::where_::<D>(mask, x_tile - mean, zeros);
        var_sum = var_sum + T::sum(diff * diff, None, true);
        t += BLOCK_NL;
    }
    let eps_t = T::cast::<f32, D>(T::full::<f32>(&[1], eps), None, false);
    let rstd_1 = T::rsqrt(var_sum * gs_inv + eps_t);
    let rstd = T::broadcast_to(rstd_1, &[BLOCK_NL]);

    T::store(mean_ptr.add_offsets(stat_idx), mean_1, None, &[], None, None);
    T::store(rstd_ptr.add_offsets(stat_idx), rstd_1, None, &[], None, None);

    // ── Pass 3: normalise ─────────────────────────────────────────────────────
    t = 0;
    while t < group_size {
        let offs = T::arange(0, BLOCK_NL) + t;
        let mask = offs.lt(group_size);
        let chan_offs = (T::arange(0, BLOCK_NL) + t) / L + g * channels_per_group;
        let x_tile = T::load(
            x_ptr.add_offsets(offs + group_start),
            Some(mask),
            Some(zeros),
            &[],
            None,
            None,
            None,
            false,
        );
        let gamma = T::load(weight_ptr.add_offsets(chan_offs), Some(mask), Some(zeros), &[], None, None, None, false);
        let beta = T::load(bias_ptr.add_offsets(chan_offs), Some(mask), Some(zeros), &[], None, None, None, false);
        let y_tile = (x_tile - mean) * rstd * gamma + beta;
        T::store(y_ptr.add_offsets(offs + group_start), y_tile, Some(mask), &[], None, None);
        t += BLOCK_NL;
    }
}

// ─── Training backward ───────────────────────────────────────────────────────

/// GroupNorm backward pass.
///
/// Grid: `[N * G]` — one CTA per (sample, group).
#[cfg(feature = "training")]
#[kernel]
pub fn group_norm_backward<T: Triton, D: Float, const BLOCK_NL: i32>(
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
    G: i32,
) where
    T::I32Tensor: types::Tensor<i32, 1>,
    T::I32Tensor: Comparison<i32, BoolTensor = T::BoolTensor>,
    T::Pointer<D>: AddOffsets<i32, 1, T::I32Tensor, Output = T::Tensor<T::Pointer<D>>>,
{
    let pid = T::program_id(Axis::X);
    let n = pid / G;
    let g = pid - n * G;
    let channels_per_group = C / G;
    let group_size = channels_per_group * L;
    let group_start = n * C * L + g * channels_per_group * L;
    let stat_idx = T::arange(0, 1) + pid;

    let zeros = T::zeros::<D>(&[BLOCK_NL]);
    let zero_1 = T::zeros::<D>(&[1]);
    let gs_inv = T::cast::<f32, D>(T::full::<f32>(&[1], 1.0f32 / (group_size as f32)), None, false);

    let rstd_1 = T::load(rstd_ptr.add_offsets(stat_idx), None, None, &[], None, None, None, false);
    let mean_1 = T::load(mean_ptr.add_offsets(stat_idx), None, None, &[], None, None, None, false);
    let rstd = T::broadcast_to(rstd_1, &[BLOCK_NL]);
    let mean = T::broadcast_to(mean_1, &[BLOCK_NL]);

    // ── Pass 1: row-level dot products ────────────────────────────────────────
    let mut sum_dy_gamma = zero_1;
    let mut sum_dy_gamma_xhat = zero_1;
    let mut t: i32 = 0;
    while t < group_size {
        let offs = T::arange(0, BLOCK_NL) + t;
        let mask = offs.lt(group_size);
        let chan_offs = (T::arange(0, BLOCK_NL) + t) / L + g * channels_per_group;
        let x_tile = T::load(
            x_ptr.add_offsets(offs + group_start),
            Some(mask),
            Some(zeros),
            &[],
            None,
            None,
            None,
            false,
        );
        let dy_tile = T::load(
            dy_ptr.add_offsets(offs + group_start),
            Some(mask),
            Some(zeros),
            &[],
            None,
            None,
            None,
            false,
        );
        let gamma = T::load(weight_ptr.add_offsets(chan_offs), Some(mask), Some(zeros), &[], None, None, None, false);
        let xhat = (x_tile - mean) * rstd;
        sum_dy_gamma = sum_dy_gamma + T::sum(dy_tile * gamma, None, true);
        sum_dy_gamma_xhat = sum_dy_gamma_xhat + T::sum(dy_tile * gamma * xhat, None, true);
        t += BLOCK_NL;
    }
    let c1 = T::broadcast_to(sum_dy_gamma * gs_inv, &[BLOCK_NL]);
    let c2 = T::broadcast_to(sum_dy_gamma_xhat * gs_inv, &[BLOCK_NL]);

    // ── Pass 2: dx and dweight / dbias ───────────────────────────────────────
    t = 0;
    while t < group_size {
        let offs = T::arange(0, BLOCK_NL) + t;
        let mask = offs.lt(group_size);
        let chan_offs = (T::arange(0, BLOCK_NL) + t) / L + g * channels_per_group;
        let x_tile = T::load(
            x_ptr.add_offsets(offs + group_start),
            Some(mask),
            Some(zeros),
            &[],
            None,
            None,
            None,
            false,
        );
        let dy_tile = T::load(
            dy_ptr.add_offsets(offs + group_start),
            Some(mask),
            Some(zeros),
            &[],
            None,
            None,
            None,
            false,
        );
        let gamma = T::load(weight_ptr.add_offsets(chan_offs), Some(mask), Some(zeros), &[], None, None, None, false);
        let dw_old = T::load(dweight_ptr.add_offsets(chan_offs), Some(mask), Some(zeros), &[], None, None, None, false);
        let db_old = T::load(dbias_ptr.add_offsets(chan_offs), Some(mask), Some(zeros), &[], None, None, None, false);

        let xhat = (x_tile - mean) * rstd;
        let dx_tile = rstd * gamma * (dy_tile - c1 - xhat * c2);

        T::store(dx_ptr.add_offsets(offs + group_start), dx_tile, Some(mask), &[], None, None);
        T::store(dweight_ptr.add_offsets(chan_offs), dw_old + dy_tile * xhat, Some(mask), &[], None, None);
        T::store(dbias_ptr.add_offsets(chan_offs), db_old + dy_tile, Some(mask), &[], None, None);
        t += BLOCK_NL;
    }
}
