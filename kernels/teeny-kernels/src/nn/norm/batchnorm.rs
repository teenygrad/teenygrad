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

//! BatchNorm1d Triton kernels.
//!
//! Layout: input `x` is `[N, C]` row-major. Element `x[n, c]` lives at flat
//! offset `n * C + c`.
//!
//! Parallelism: **one CTA per channel**. Each CTA iterates over all N batch
//! elements in `BLOCK_N`-wide tiles. This avoids cross-CTA synchronisation
//! entirely — C channels execute concurrently across SMs.
//!
//! Training requires two sequential kernel launches separated by a host sync:
//!   1. `batch_norm_stats_forward`   — computes per-channel mean + rstd, updates
//!      running stats.
//!   2. `batch_norm_normalize_forward` — normalises x using the saved stats.
//!
//! Inference uses a single kernel that reads the frozen running statistics.

#![allow(non_snake_case)]

use teeny_core::dtype::Float;
use teeny_macros::kernel;
use teeny_triton::triton::{
    types::{AddOffsets, Comparison},
    *,
};

// ─── Inference: single kernel, frozen running statistics ─────────────────────

/// Normalises input `x` using the frozen `running_mean` / `running_var`.
///
/// Grid: `[C]` — one CTA per channel.
#[kernel]
pub fn batch_norm_forward_inference<T: Triton, D: Float, const BLOCK_N: i32>(
    x_ptr: T::Pointer<D>,
    y_ptr: T::Pointer<D>,
    weight_ptr: T::Pointer<D>,
    bias_ptr: T::Pointer<D>,
    running_mean_ptr: T::Pointer<D>,
    running_var_ptr: T::Pointer<D>,
    N: i32,
    C: i32,
    eps: f32,
) where
    T::I32Tensor: types::Tensor<i32, 1>,
    T::I32Tensor: Comparison<i32, BoolTensor = T::BoolTensor>,
    T::Pointer<D>: AddOffsets<i32, 1, T::I32Tensor, Output = T::Tensor<T::Pointer<D>>>,
{
    let c = T::program_id(Axis::X);
    let c_idx = T::arange(0, 1) + c;

    // Load per-channel scalars (shape [1]) and broadcast to [BLOCK_N].
    let mean = T::broadcast_to(
        T::load(running_mean_ptr.add_offsets(c_idx), None, None, &[], None, None, None, false),
        &[BLOCK_N],
    );
    let var = T::load(running_var_ptr.add_offsets(c_idx), None, None, &[], None, None, None, false);
    let rstd = T::broadcast_to(T::rsqrt(var + T::cast::<f32, D>(T::full::<f32>(&[1], eps), None, false)), &[BLOCK_N]);
    let gamma = T::broadcast_to(
        T::load(weight_ptr.add_offsets(c_idx), None, None, &[], None, None, None, false),
        &[BLOCK_N],
    );
    let beta = T::broadcast_to(
        T::load(bias_ptr.add_offsets(c_idx), None, None, &[], None, None, None, false),
        &[BLOCK_N],
    );

    // Normalise all N elements for this channel.
    let zeros = T::zeros::<D>(&[BLOCK_N]);
    let mut n_start: i32 = 0;
    while n_start < N {
        let offsets_n = T::arange(0, BLOCK_N) + n_start;
        let mask = offsets_n.lt(N);
        let elem_offsets = offsets_n * C + c;

        let x_tile = T::load(x_ptr.add_offsets(elem_offsets), Some(mask), Some(zeros), &[], None, None, None, false);
        let y_tile = gamma * (x_tile - mean) * rstd + beta;

        T::store(y_ptr.add_offsets(elem_offsets), y_tile, Some(mask), &[], None, None);

        n_start += BLOCK_N;
    }
}

// ─── Training: kernel 1 — compute per-channel statistics ─────────────────────

/// Computes per-channel mean and rstd from the current mini-batch, saves them
/// for the normalisation kernel and the backward pass, and updates the running
/// statistics with exponential moving average.
///
/// Grid: `[C]` — one CTA per channel.
///
/// **Must complete (host sync) before `batch_norm_normalize_forward` is launched.**
#[cfg(feature = "training")]
#[kernel]
pub fn batch_norm_stats_forward<T: Triton, D: Float, const BLOCK_N: i32>(
    x_ptr: T::Pointer<D>,
    mean_ptr: T::Pointer<D>,
    rstd_ptr: T::Pointer<D>,
    running_mean_ptr: T::Pointer<D>,
    running_var_ptr: T::Pointer<D>,
    N: i32,
    C: i32,
    eps: f32,
    momentum: f32,
) where
    T::I32Tensor: types::Tensor<i32, 1>,
    T::I32Tensor: Comparison<i32, BoolTensor = T::BoolTensor>,
    T::Pointer<D>: AddOffsets<i32, 1, T::I32Tensor, Output = T::Tensor<T::Pointer<D>>>,
{
    let c = T::program_id(Axis::X);

    // Accumulate sum(x) and sum(x²) over all N elements for this channel.
    let zeros_1 = T::zeros::<D>(&[1]);
    let zeros_blk = T::zeros::<D>(&[BLOCK_N]);
    let mut sum = zeros_1;
    let mut sum_sq = zeros_1;
    let mut n_start: i32 = 0;

    while n_start < N {
        let offsets_n = T::arange(0, BLOCK_N) + n_start;
        let mask = offsets_n.lt(N);
        let elem_offsets = offsets_n * C + c;

        let x_tile = T::load(x_ptr.add_offsets(elem_offsets), Some(mask), Some(zeros_blk), &[], None, None, None, false);
        sum = sum + T::sum(x_tile, None, true);
        sum_sq = sum_sq + T::sum(x_tile * x_tile, None, true);

        n_start += BLOCK_N;
    }

    // Derive mean, biased variance, and rstd (all shape [1]).
    let n_inv = T::cast::<f32, D>(T::full::<f32>(&[1], 1.0f32 / (N as f32)), None, false);
    let mean_1 = sum * n_inv;
    let var_1 = sum_sq * n_inv - mean_1 * mean_1;
    let rstd_1 = T::rsqrt(var_1 + T::cast::<f32, D>(T::full::<f32>(&[1], eps), None, false));

    // Save for the normalisation and backward kernels.
    let c_idx = T::arange(0, 1) + c;
    T::store(mean_ptr.add_offsets(c_idx), mean_1, None, &[], None, None);
    T::store(rstd_ptr.add_offsets(c_idx), rstd_1, None, &[], None, None);

    // Exponential moving average: running = (1 - m) * running + m * batch.
    let m = T::cast::<f32, D>(T::full::<f32>(&[1], momentum), None, false);
    let one_m = T::cast::<f32, D>(T::full::<f32>(&[1], 1.0f32 - momentum), None, false);
    let running_mean_old = T::load(running_mean_ptr.add_offsets(c_idx), None, None, &[], None, None, None, false);
    let running_var_old = T::load(running_var_ptr.add_offsets(c_idx), None, None, &[], None, None, None, false);

    T::store(running_mean_ptr.add_offsets(c_idx), one_m * running_mean_old + m * mean_1, None, &[], None, None);
    T::store(running_var_ptr.add_offsets(c_idx), one_m * running_var_old + m * var_1, None, &[], None, None);
}

// ─── Training: kernel 2 — normalise using saved statistics ───────────────────

/// Normalises x using the mean and rstd produced by `batch_norm_stats_forward`.
///
/// Grid: `[C]` — one CTA per channel.
#[cfg(feature = "training")]
#[kernel]
pub fn batch_norm_normalize_forward<T: Triton, D: Float, const BLOCK_N: i32>(
    x_ptr: T::Pointer<D>,
    y_ptr: T::Pointer<D>,
    weight_ptr: T::Pointer<D>,
    bias_ptr: T::Pointer<D>,
    mean_ptr: T::Pointer<D>,
    rstd_ptr: T::Pointer<D>,
    N: i32,
    C: i32,
) where
    T::I32Tensor: types::Tensor<i32, 1>,
    T::I32Tensor: Comparison<i32, BoolTensor = T::BoolTensor>,
    T::Pointer<D>: AddOffsets<i32, 1, T::I32Tensor, Output = T::Tensor<T::Pointer<D>>>,
{
    let c = T::program_id(Axis::X);
    let c_idx = T::arange(0, 1) + c;

    // Load per-channel scalars and broadcast to [BLOCK_N].
    let mean = T::broadcast_to(
        T::load(mean_ptr.add_offsets(c_idx), None, None, &[], None, None, None, false),
        &[BLOCK_N],
    );
    let rstd = T::broadcast_to(
        T::load(rstd_ptr.add_offsets(c_idx), None, None, &[], None, None, None, false),
        &[BLOCK_N],
    );
    let gamma = T::broadcast_to(
        T::load(weight_ptr.add_offsets(c_idx), None, None, &[], None, None, None, false),
        &[BLOCK_N],
    );
    let beta = T::broadcast_to(
        T::load(bias_ptr.add_offsets(c_idx), None, None, &[], None, None, None, false),
        &[BLOCK_N],
    );

    let zeros = T::zeros::<D>(&[BLOCK_N]);
    let mut n_start: i32 = 0;
    while n_start < N {
        let offsets_n = T::arange(0, BLOCK_N) + n_start;
        let mask = offsets_n.lt(N);
        let elem_offsets = offsets_n * C + c;

        let x_tile = T::load(x_ptr.add_offsets(elem_offsets), Some(mask), Some(zeros), &[], None, None, None, false);
        let y_tile = gamma * (x_tile - mean) * rstd + beta;

        T::store(y_ptr.add_offsets(elem_offsets), y_tile, Some(mask), &[], None, None);

        n_start += BLOCK_N;
    }
}

// ─── Training: backward pass ──────────────────────────────────────────────────

/// Computes gradients for BatchNorm.
///
/// Given saved `mean` and `rstd` from the forward pass:
/// ```text
/// xhat      = (x - mean) * rstd
/// dbias[c]  = Σ_n dy[n,c]
/// dweight[c]= Σ_n dy[n,c] * xhat[n,c]
/// dx[n,c]   = weight[c] * rstd[c] * (dy[n,c]
///               - dbias[c] / N
///               - xhat[n,c] * dweight[c] / N)
/// ```
///
/// Uses two sequential passes over N within the same CTA to avoid storing
/// the full xhat tensor.
///
/// Grid: `[C]` — one CTA per channel.
#[cfg(feature = "training")]
#[kernel]
pub fn batch_norm_backward<T: Triton, D: Float, const BLOCK_N: i32>(
    dy_ptr: T::Pointer<D>,
    x_ptr: T::Pointer<D>,
    dx_ptr: T::Pointer<D>,
    weight_ptr: T::Pointer<D>,
    mean_ptr: T::Pointer<D>,
    rstd_ptr: T::Pointer<D>,
    dweight_ptr: T::Pointer<D>,
    dbias_ptr: T::Pointer<D>,
    N: i32,
    C: i32,
) where
    T::I32Tensor: types::Tensor<i32, 1>,
    T::I32Tensor: Comparison<i32, BoolTensor = T::BoolTensor>,
    T::Pointer<D>: AddOffsets<i32, 1, T::I32Tensor, Output = T::Tensor<T::Pointer<D>>>,
{
    let c = T::program_id(Axis::X);
    let c_idx = T::arange(0, 1) + c;

    // Load per-channel scalars; broadcast to [BLOCK_N] for element-wise ops.
    let mean = T::broadcast_to(
        T::load(mean_ptr.add_offsets(c_idx), None, None, &[], None, None, None, false),
        &[BLOCK_N],
    );
    let rstd = T::broadcast_to(
        T::load(rstd_ptr.add_offsets(c_idx), None, None, &[], None, None, None, false),
        &[BLOCK_N],
    );
    let weight = T::broadcast_to(
        T::load(weight_ptr.add_offsets(c_idx), None, None, &[], None, None, None, false),
        &[BLOCK_N],
    );

    // Pass 1: accumulate dbias (= Σ dy) and dweight (= Σ dy * xhat).
    let zeros_1 = T::zeros::<D>(&[1]);
    let zeros_blk = T::zeros::<D>(&[BLOCK_N]);
    let mut sum_dy = zeros_1;
    let mut sum_dy_xhat = zeros_1;
    let mut n_start: i32 = 0;

    while n_start < N {
        let offsets_n = T::arange(0, BLOCK_N) + n_start;
        let mask = offsets_n.lt(N);
        let elem_offsets = offsets_n * C + c;

        let x_tile = T::load(x_ptr.add_offsets(elem_offsets), Some(mask), Some(zeros_blk), &[], None, None, None, false);
        let dy_tile = T::load(dy_ptr.add_offsets(elem_offsets), Some(mask), Some(zeros_blk), &[], None, None, None, false);
        let xhat = (x_tile - mean) * rstd;

        sum_dy = sum_dy + T::sum(dy_tile, None, true);
        sum_dy_xhat = sum_dy_xhat + T::sum(dy_tile * xhat, None, true);

        n_start += BLOCK_N;
    }

    // Save dweight and dbias (shape [1] → stored as scalars).
    T::store(dweight_ptr.add_offsets(c_idx), sum_dy_xhat, None, &[], None, None);
    T::store(dbias_ptr.add_offsets(c_idx), sum_dy, None, &[], None, None);

    // Broadcast reduction results and 1/N for pass 2.
    let n_inv = T::broadcast_to(T::cast::<f32, D>(T::full::<f32>(&[1], 1.0f32 / (N as f32)), None, false), &[BLOCK_N]);
    let sum_dy_bcast = T::broadcast_to(sum_dy, &[BLOCK_N]);
    let sum_dy_xhat_bcast = T::broadcast_to(sum_dy_xhat, &[BLOCK_N]);

    // Pass 2: compute dx.
    n_start = 0;
    while n_start < N {
        let offsets_n = T::arange(0, BLOCK_N) + n_start;
        let mask = offsets_n.lt(N);
        let elem_offsets = offsets_n * C + c;

        let x_tile = T::load(x_ptr.add_offsets(elem_offsets), Some(mask), Some(zeros_blk), &[], None, None, None, false);
        let dy_tile = T::load(dy_ptr.add_offsets(elem_offsets), Some(mask), Some(zeros_blk), &[], None, None, None, false);
        let xhat = (x_tile - mean) * rstd;

        let dx_tile = weight * rstd * (dy_tile - sum_dy_bcast * n_inv - xhat * sum_dy_xhat_bcast * n_inv);

        T::store(dx_ptr.add_offsets(elem_offsets), dx_tile, Some(mask), &[], None, None);

        n_start += BLOCK_N;
    }
}
