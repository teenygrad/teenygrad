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

//! RMSNorm Triton kernels.
//!
//! RMSNorm normalises each row by its root-mean-square (no mean subtraction):
//!   rms[m]    = sqrt( (1/N) * Σ_n x[m,n]² + eps )
//!   y[m,n]    = x[m,n] / rms[m] * γ[n]
//!
//! Grid: `[M]` — one CTA per row. Layout identical to LayerNorm.

#![allow(non_snake_case)]

use teeny_core::dtype::Float;
use teeny_macros::{kernel, tiled_kernel};
use teeny_triton::triton::{
    types::{AddOffsets, Comparison},
    *,
};

// ─── Forward ─────────────────────────────────────────────────────────────────

/// RMSNorm forward pass.
///
/// Grid: `[M]` — one CTA per row.
#[tiled_kernel]
#[tile_loop(trip_count = [N, BLOCK_N])]
#[tile_carry(sq_sum = [1])]
// teenygrad-3rk6.2: the first multi-pass kernel. One declared reduction pass
// walks the row summing squares; the body below is the MAP pass, which the
// macro walks again and whose trailing expression it stores.
//
// `n_inv` and `eps` are inlined into `finish` rather than bound in a preamble:
// the passes run before any author code, so there is no preamble to bind them
// in. That is the cost of the design and it is confined to one expression.
#[tile_reduce_pass(
    over = N,
    read = x,
    into = sq_sum,
    acc = T::sum(x * x, None, true),
    finish = T::rsqrt(
        sq_sum * T::cast::<f32, D>(T::full::<f32>(&[1], 1.0f32 / (N as f32)), None, false)
            + T::cast::<f32, D>(T::full::<f32>(&[1], eps), None, false)
    ),
    store = rrms
)]
pub fn rms_norm_forward<T: Triton, D: Float, const BLOCK_N: i32>(
    #[tile(name = "M", block = 1, extent = _M)]
    // `reduce` AND `walk`: they say different things. `reduce` is what the axis
    // MEANS -- the pass collapses it to the per-row statistic `rrms`, and the
    // spec's `reduction_axis` must keep saying so for propagation. `walk` is
    // HOW it is read: in blocks by a pass, rather than whole into one tile.
    // Same split as `#[tile(.. window(..))]` saying what an axis is while
    // `#[tile_loop_tile]` says how to read it (teenygrad-3rk6.2).
    #[tile(block = BLOCK_N, extent = N, reduce, walk)]
    x: In<Tile<T, D>>,
    #[tile(name = "M", block = 1, extent = _M)]
    #[tile(block = BLOCK_N, extent = N, walk)]
    y: Out<Tile<T, D>>,
    #[tile(block = BLOCK_N, extent = N, walk)] weight: In<Tile<T, D>>,
    #[tile(name = "M", block = 1, extent = _M)] rrms: Out<Tile<T, D>>,
    _M: i32,
    N: i32,
    eps: f32,
) where
    T::I32Tensor: types::Tensor<i32, 1>,
    T::I32Tensor: Comparison<i32, BoolTensor = T::BoolTensor>,
    T::Pointer<D>: AddOffsets<i32, 1, T::I32Tensor, Output = T::Tensor<T::Pointer<D>>>,
{
    // One iteration of the map pass. `sq_sum` is the finished statistic, a
    // `[1]` tile, broadcast to the block.
    x * T::broadcast_to(sq_sum, &[BLOCK_N]) * weight
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
    dy_ptr: In<T::Pointer<D>>,
    x_ptr: In<T::Pointer<D>>,
    dx_ptr: Out<T::Pointer<D>>,
    weight_ptr: In<T::Pointer<D>>,
    dweight_ptr: InOut<T::Pointer<D>>,
    rrms_ptr: In<T::Pointer<D>>,
    _M: i32,
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

    let rrms_1 = T::load(
        rrms_ptr.add_offsets(row_idx),
        None,
        None,
        &[],
        None,
        None,
        None,
        false,
    );
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
        let gamma = T::load(
            weight_ptr.add_offsets(col_offs),
            Some(mask),
            Some(zeros),
            &[],
            None,
            None,
            None,
            false,
        );
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
        let gamma = T::load(
            weight_ptr.add_offsets(col_offs),
            Some(mask),
            Some(zeros),
            &[],
            None,
            None,
            None,
            false,
        );
        let dw_old = T::load(
            dweight_ptr.add_offsets(col_offs),
            Some(mask),
            Some(zeros),
            &[],
            None,
            None,
            None,
            false,
        );

        let dx_tile = rrms * gamma * (dy_tile - x_tile * rrms_sq * scale);
        T::store(
            dx_ptr.add_offsets(col_offs + row_start),
            dx_tile,
            Some(mask),
            &[],
            None,
            None,
        );
        T::store(
            dweight_ptr.add_offsets(col_offs),
            dw_old + dy_tile * x_tile * rrms,
            Some(mask),
            &[],
            None,
            None,
        );
        n_start += BLOCK_N;
    }
}
