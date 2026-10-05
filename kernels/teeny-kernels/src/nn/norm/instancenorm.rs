/*
 * Copyright (c) 2026 teenygrad (https://teenygrad.org).
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
use teeny_macros::{kernel, tiled_kernel};
use teeny_triton::triton::{
    types::{AddOffsets, Comparison},
    *,
};

// ─── Inference ───────────────────────────────────────────────────────────────

/// InstanceNorm forward (inference — no running stats).
///
/// Grid: `[N * C]` — one CTA per (sample, channel).
#[tiled_kernel]
#[tile_loop(trip_count = [L, BLOCK_L])]
#[tile_carry(sum = [1], var_sum = [1])]
pub fn instance_norm_forward_inference<T: Triton, D: Float, const BLOCK_L: i32>(
    #[tile(name = "N", extent = _N)]
    #[tile(extent = C)]
    #[tile(extent = L, reduce)]
    x_ptr: In<T::Pointer<D>>,
    #[tile(name = "N", extent = _N)]
    #[tile(extent = C)]
    #[tile(extent = L, reduce)]
    y_ptr: Out<T::Pointer<D>>,
    #[tile(extent = C)] weight_ptr: In<T::Pointer<D>>,
    #[tile(extent = C)] bias_ptr: In<T::Pointer<D>>,
    _N: i32,
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
        T::load(
            weight_ptr.add_offsets(c_idx),
            None,
            None,
            &[],
            None,
            None,
            None,
            false,
        ),
        &[BLOCK_L],
    );
    let beta = T::broadcast_to(
        T::load(
            bias_ptr.add_offsets(c_idx),
            None,
            None,
            &[],
            None,
            None,
            None,
            false,
        ),
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
        T::store(
            y_ptr.add_offsets(col_offs + row_start),
            y_tile,
            Some(mask),
            &[],
            None,
            None,
        );
        l_start += BLOCK_L;
    }
}

// ─── Training forward ─────────────────────────────────────────────────────────

/// InstanceNorm training forward — saves per-(n,c) mean and rstd.
///
/// Grid: `[N * C]` — one CTA per (sample, channel).
#[cfg(feature = "training")]
#[tiled_kernel]
#[tile_loop(trip_count = [L, BLOCK_L])]
#[tile_carry(sum = [1], var_sum = [1])]
// teenygrad-3rk6.2 stage 3: the same two-pass shape as layer_norm, over L
// instead of N, with one more gridded axis. The statistic is per
// (sample, channel) rather than per row.
#[tile_reduce_pass(
    over = L,
    read = x,
    into = sum,
    acc = T::sum(x, None, true),
    finish = sum * T::cast::<f32, D>(
        T::full::<f32>(&[1], 1.0f32 / (L as f32)), None, false
    ),
    store = mean
)]
#[tile_reduce_pass(
    over = L,
    read = x,
    into = var_sum,
    acc = T::sum(
        (x - T::broadcast_to(sum, &[BLOCK_L])) * (x - T::broadcast_to(sum, &[BLOCK_L])),
        None,
        true
    ),
    finish = T::rsqrt(
        var_sum * T::cast::<f32, D>(
            T::full::<f32>(&[1], 1.0f32 / (L as f32)), None, false
        ) + T::cast::<f32, D>(T::full::<f32>(&[1], eps), None, false)
    ),
    // Masked lanes load as the mean so their diff is zero. With the default
    // zeros fill each would contribute `mean^2` -- the bug layer_norm's
    // conversion hit (teenygrad-3rk6.2).
    fill = T::broadcast_to(sum, &[BLOCK_L]),
    store = rstd
)]
pub fn instance_norm_forward<T: Triton, D: Float, const BLOCK_L: i32>(
    #[tile(name = "N", block = 1, extent = _N)]
    #[tile(block = 1, extent = C)]
    #[tile(block = BLOCK_L, extent = L, reduce, walk)]
    x: In<Tile<T, D>>,
    #[tile(name = "N", block = 1, extent = _N)]
    #[tile(block = 1, extent = C)]
    #[tile(block = BLOCK_L, extent = L, walk)]
    y: Out<Tile<T, D>>,
    // Raw pointers, as conv2d_bias leaves its bias. These are the read-once
    // broadcast operand of teenygrad-3rk6.1: indexed by the gridded C alone,
    // they would need a blocked axis for the prelude to shape their load, and
    // with two gridded block-1 axes that load comes out rank 2 -- while this
    // kernel's map pass works in rank-1 blocks. `broadcast_to` cannot bridge
    // ranks, so they are loaded in the body instead.
    weight_ptr: In<T::Pointer<D>>,
    bias_ptr: In<T::Pointer<D>>,
    #[tile(name = "N", block = 1, extent = _N)]
    #[tile(block = 1, extent = C)]
    mean: Out<Tile<T, D>>,
    #[tile(name = "N", block = 1, extent = _N)]
    #[tile(block = 1, extent = C)]
    rstd: Out<Tile<T, D>>,
    _N: i32,
    C: i32,
    L: i32,
    eps: f32,
) where
    T::I32Tensor: types::Tensor<i32, 1>,
    T::I32Tensor: Comparison<i32, BoolTensor = T::BoolTensor>,
    T::Pointer<D>: AddOffsets<i32, 1, T::I32Tensor, Output = T::Tensor<T::Pointer<D>>>,
{
    // One iteration of the map pass. `tile_c` is the prelude's channel index.
    let c_idx = T::arange(0, 1) + tile_c;
    let gamma = T::broadcast_to(
        T::load(
            weight_ptr.add_offsets(c_idx),
            None,
            None,
            &[],
            None,
            None,
            None,
            false,
        ),
        &[BLOCK_L],
    );
    let beta = T::broadcast_to(
        T::load(
            bias_ptr.add_offsets(c_idx),
            None,
            None,
            &[],
            None,
            None,
            None,
            false,
        ),
        &[BLOCK_L],
    );
    (x - T::broadcast_to(sum, &[BLOCK_L])) * T::broadcast_to(var_sum, &[BLOCK_L]) * gamma
        + beta
}

// ─── Training backward ───────────────────────────────────────────────────────

/// InstanceNorm backward pass.
///
/// Grid: `[N * C]` — one CTA per (sample, channel).
#[cfg(feature = "training")]
#[kernel]
pub fn instance_norm_backward<T: Triton, D: Float, const BLOCK_L: i32>(
    dy_ptr: In<T::Pointer<D>>,
    x_ptr: In<T::Pointer<D>>,
    dx_ptr: Out<T::Pointer<D>>,
    weight_ptr: In<T::Pointer<D>>,
    dweight_ptr: InOut<T::Pointer<D>>,
    dbias_ptr: InOut<T::Pointer<D>>,
    mean_ptr: In<T::Pointer<D>>,
    rstd_ptr: In<T::Pointer<D>>,
    _N: i32,
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

    let rstd_1 = T::load(
        rstd_ptr.add_offsets(stat_idx),
        None,
        None,
        &[],
        None,
        None,
        None,
        false,
    );
    let mean_1 = T::load(
        mean_ptr.add_offsets(stat_idx),
        None,
        None,
        &[],
        None,
        None,
        None,
        false,
    );
    let rstd = T::broadcast_to(rstd_1, &[BLOCK_L]);
    let mean = T::broadcast_to(mean_1, &[BLOCK_L]);

    let gamma = T::broadcast_to(
        T::load(
            weight_ptr.add_offsets(c_idx),
            None,
            None,
            &[],
            None,
            None,
            None,
            false,
        ),
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
        let dw_old = T::load(
            dweight_ptr.add_offsets(c_idx),
            None,
            None,
            &[],
            None,
            None,
            None,
            false,
        );
        let db_old = T::load(
            dbias_ptr.add_offsets(c_idx),
            None,
            None,
            &[],
            None,
            None,
            None,
            false,
        );

        let xhat = (x_tile - mean) * rstd;
        let dx_tile = rstd * gamma * (dy_tile - c1 - xhat * c2);

        T::store(
            dx_ptr.add_offsets(col_offs + row_start),
            dx_tile,
            Some(mask),
            &[],
            None,
            None,
        );
        T::store(
            dweight_ptr.add_offsets(c_idx),
            dw_old + T::sum(dy_tile * xhat, None, true),
            None,
            &[],
            None,
            None,
        );
        T::store(
            dbias_ptr.add_offsets(c_idx),
            db_old + T::sum(dy_tile, None, true),
            None,
            &[],
            None,
            None,
        );
        l_start += BLOCK_L;
    }
}
