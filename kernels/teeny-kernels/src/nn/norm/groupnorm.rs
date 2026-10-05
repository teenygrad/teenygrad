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
use teeny_macros::{kernel, tiled_kernel};
use teeny_triton::triton::{
    types::{AddOffsets, Comparison},
    *,
};

// ─── Inference ───────────────────────────────────────────────────────────────

/// GroupNorm forward (inference).
///
/// Grid: `[N * G]` — one CTA per (sample, group).
// Declares what is true and no more (teenygrad-1tl.8). Two things about this
// kernel are not expressible today, and guessing at them would be worse than
// leaving them out:
//
// 1. NO `reduce` FLAG. The reduction runs over `group_size = (C / G) * L` --
//    jointly across this group's channels and the spatial axis -- but
//    `TensorTileSpec::reduction_axis` is a single index and cannot say "these
//    two dims together". Marking L alone would understate it.
// 2. NO `divide_by` ON C. The grid is `[N * G]`, so a CTA owns `C / G`
//    channels, not all of C. `divide_by` exists for exactly this
//    (`channels_per_group`), but it is a number fixed at spec-construction
//    time while `G` here is a runtime `i32` parameter, and the generated
//    `tile_spec()` returns `&'static` data. So C falls back to its full
//    extent: conservative and correct, just less precise, which matches the
//    "never smaller than needed" philosophy used elsewhere.
//
// The loop below IS declarable: its carries are scalars and its trip count
// factors are all named parameters.
#[tiled_kernel]
#[tile_loop(trip_count = [C, G, L, BLOCK_NL])]
#[tile_carry(sum = [1], var_sum = [1])]
pub fn group_norm_forward_inference<T: Triton, D: Float, const BLOCK_NL: i32>(
    #[tile(name = "N", extent = _N)]
    #[tile(extent = C)]
    #[tile(extent = L)]
    x_ptr: In<T::Pointer<D>>,
    #[tile(name = "N", extent = _N)]
    #[tile(extent = C)]
    #[tile(extent = L)]
    y_ptr: Out<T::Pointer<D>>,
    #[tile(extent = C)] weight_ptr: In<T::Pointer<D>>,
    #[tile(extent = C)] bias_ptr: In<T::Pointer<D>>,
    _N: i32,
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
    let gs_inv = T::cast::<f32, D>(
        T::full::<f32>(&[1], 1.0f32 / (group_size as f32)),
        None,
        false,
    );

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
        let gamma = T::load(
            weight_ptr.add_offsets(chan_offs),
            Some(mask),
            Some(zeros),
            &[],
            None,
            None,
            None,
            false,
        );
        let beta = T::load(
            bias_ptr.add_offsets(chan_offs),
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
            y_ptr.add_offsets(offs + group_start),
            y_tile,
            Some(mask),
            &[],
            None,
            None,
        );
        t += BLOCK_NL;
    }
}

// ─── Training forward ─────────────────────────────────────────────────────────

/// GroupNorm training forward — saves per-(n,g) mean and rstd.
///
/// Grid: `[N * G]` — one CTA per (sample, group).
#[cfg(feature = "training")]
// Declares what is true and no more (teenygrad-1tl.8). Two things about this
// kernel are not expressible today, and guessing at them would be worse than
// leaving them out:
//
// 1. NO `reduce` FLAG. The reduction runs over `group_size = (C / G) * L` --
//    jointly across this group's channels and the spatial axis -- but
//    `TensorTileSpec::reduction_axis` is a single index and cannot say "these
//    two dims together". Marking L alone would understate it.
// 2. NO `divide_by` ON C. The grid is `[N * G]`, so a CTA owns `C / G`
//    channels, not all of C. `divide_by` exists for exactly this
//    (`channels_per_group`), but it is a number fixed at spec-construction
//    time while `G` here is a runtime `i32` parameter, and the generated
//    `tile_spec()` returns `&'static` data. So C falls back to its full
//    extent: conservative and correct, just less precise, which matches the
//    "never smaller than needed" philosophy used elsewhere.
//
// The loop below IS declarable: its carries are scalars and its trip count
// factors are all named parameters.
#[tiled_kernel]
#[tile_loop(trip_count = [C, G, L, BLOCK_NL])]
#[tile_carry(sum = [1], var_sum = [1])]
// teenygrad-3rk6.2: two reduction passes over the group span, then the map.
// The statistic is per (sample, group).
#[tile_reduce_pass(
    over = group_size,
    read = x,
    into = sum,
    acc = T::sum(x, None, true),
    finish = sum * T::cast::<f32, D>(
        T::full::<f32>(&[1], 1.0f32 / (group_size as f32)), None, false
    ),
    store = mean
)]
#[tile_reduce_pass(
    over = group_size,
    read = x,
    into = var_sum,
    acc = T::sum(
        (x - T::broadcast_to(sum, &[BLOCK_NL])) * (x - T::broadcast_to(sum, &[BLOCK_NL])),
        None,
        true
    ),
    finish = T::rsqrt(
        var_sum * T::cast::<f32, D>(
            T::full::<f32>(&[1], 1.0f32 / (group_size as f32)), None, false
        ) + T::cast::<f32, D>(T::full::<f32>(&[1], eps), None, false)
    ),
    fill = T::broadcast_to(sum, &[BLOCK_NL]),
    store = rstd
)]
pub fn group_norm_forward<T: Triton, D: Float, const BLOCK_NL: i32>(
    // Declared as [N, G, group_size] rather than [N, C, L]: the same memory,
    // decomposed so the walked span is ONE axis. A group's elements are
    // contiguous -- `G * group_size == C * L` -- and the old form needed a
    // flattened walk over `(C / G) * L`, which no declaration could name
    // (teenygrad-3rk6.2).
    #[tile(name = "N", block = 1, extent = _N)]
    #[tile(name = "G", block = 1, extent = G)]
    #[tile(block = BLOCK_NL, extent = group_size, reduce, walk)]
    x: In<Tile<T, D>>,
    #[tile(name = "N", block = 1, extent = _N)]
    #[tile(name = "G", block = 1, extent = G)]
    #[tile(block = BLOCK_NL, extent = group_size, walk)]
    y: Out<Tile<T, D>>,
    // Raw pointers, and not for teenygrad-3rk6.1's reason: these are a
    // COMPUTED gather. A lane's channel is `tile_walk / L + tile_g * (C / G)`,
    // so gamma varies WITHIN the walked span and no declared axis offset
    // reaches it. The body loads them from the exposed walk coordinate.
    weight_ptr: In<T::Pointer<D>>,
    bias_ptr: In<T::Pointer<D>>,
    #[tile(name = "N", block = 1, extent = _N)]
    #[tile(name = "G", block = 1, extent = G)]
    mean: Out<Tile<T, D>>,
    #[tile(name = "N", block = 1, extent = _N)]
    #[tile(name = "G", block = 1, extent = G)]
    rstd: Out<Tile<T, D>>,
    _N: i32,
    C: i32,
    L: i32,
    G: i32,
    // `(C / G) * L`, passed rather than derived: `extent = ..` names a
    // parameter, so a walked axis the caller computes must be one.
    group_size: i32,
    eps: f32,
) where
    T::I32Tensor: types::Tensor<i32, 1>,
    T::I32Tensor: Comparison<i32, BoolTensor = T::BoolTensor>,
    T::Pointer<D>: AddOffsets<i32, 1, T::I32Tensor, Output = T::Tensor<T::Pointer<D>>>,
{
    // One iteration of the map pass. `tile_walk` is this block's coordinate
    // along the walked span and `tile_g` the group index, so a lane's channel
    // is `tile_walk / L + tile_g * (C / G)` -- the gather the declarations
    // cannot express.
    let chan = tile_walk / L + tile_g * (C / G);
    let gamma = T::load(
        weight_ptr.add_offsets(chan),
        Some(tile_walk_mask),
        Some(T::zeros::<D>(&[BLOCK_NL])),
        &[],
        None,
        None,
        None,
        false,
    );
    let beta = T::load(
        bias_ptr.add_offsets(chan),
        Some(tile_walk_mask),
        Some(T::zeros::<D>(&[BLOCK_NL])),
        &[],
        None,
        None,
        None,
        false,
    );
    (x - T::broadcast_to(sum, &[BLOCK_NL])) * T::broadcast_to(var_sum, &[BLOCK_NL]) * gamma
        + beta
}

// ─── Training backward ───────────────────────────────────────────────────────

/// GroupNorm backward pass.
///
/// Grid: `[N * G]` — one CTA per (sample, group).
#[cfg(feature = "training")]
#[kernel]
pub fn group_norm_backward<T: Triton, D: Float, const BLOCK_NL: i32>(
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
    let gs_inv = T::cast::<f32, D>(
        T::full::<f32>(&[1], 1.0f32 / (group_size as f32)),
        None,
        false,
    );

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
        let gamma = T::load(
            weight_ptr.add_offsets(chan_offs),
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
        let gamma = T::load(
            weight_ptr.add_offsets(chan_offs),
            Some(mask),
            Some(zeros),
            &[],
            None,
            None,
            None,
            false,
        );
        let dw_old = T::load(
            dweight_ptr.add_offsets(chan_offs),
            Some(mask),
            Some(zeros),
            &[],
            None,
            None,
            None,
            false,
        );
        let db_old = T::load(
            dbias_ptr.add_offsets(chan_offs),
            Some(mask),
            Some(zeros),
            &[],
            None,
            None,
            None,
            false,
        );

        let xhat = (x_tile - mean) * rstd;
        let dx_tile = rstd * gamma * (dy_tile - c1 - xhat * c2);

        T::store(
            dx_ptr.add_offsets(offs + group_start),
            dx_tile,
            Some(mask),
            &[],
            None,
            None,
        );
        T::store(
            dweight_ptr.add_offsets(chan_offs),
            dw_old + dy_tile * xhat,
            Some(mask),
            &[],
            None,
            None,
        );
        T::store(
            dbias_ptr.add_offsets(chan_offs),
            db_old + dy_tile,
            Some(mask),
            &[],
            None,
            None,
        );
        t += BLOCK_NL;
    }
}
