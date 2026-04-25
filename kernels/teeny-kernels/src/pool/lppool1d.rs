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

use teeny_core::dtype::Float;
use teeny_macros::kernel;
use teeny_triton::triton::{
    types::{AddOffsets, Comparison, Tensor},
    *,
};

/// 1-D Lp-norm pooling forward pass.
///
/// `y = (Σ |x_i|^p)^(1/p)` over the kernel window.
///
/// `pow(|x|, p)` is computed as `exp(p * log(max(|x|, ε)))` to avoid
/// `log(0)`. `p` is a runtime float parameter.
///
/// Grid: `pid = (b * C + c) * num_ol_tiles + ol_tile`
///
/// **Constraints**: no padding; `OL = (L - KL) / STRIDE + 1`.
#[kernel]
pub fn lppool1d_forward<
    T: Triton,
    D: Float,
    const KL: i32,
    const STRIDE: i32,
    const BLOCK_OL: i32,
>(
    input_ptr: T::Pointer<D>,
    output_ptr: T::Pointer<D>,
    B: i32,
    C: i32,
    L: i32,
    OL: i32,
    p: f32,
) where
    T::I32Tensor: Tensor<i32, 1>,
    T::I32Tensor: Comparison<i32, BoolTensor = T::BoolTensor>,
    T::Pointer<D>: AddOffsets<i32, 1, T::I32Tensor, Output = T::Tensor<T::Pointer<D>>>,
{
    let pid = T::program_id(Axis::X);
    let num_ol_tiles = T::cdiv(OL, BLOCK_OL);

    let ol_tile = pid % num_ol_tiles;
    let bc = pid / num_ol_tiles;
    let c = bc % C;
    let b = bc / C;

    let ol_start = ol_tile * BLOCK_OL;
    let ol_range = T::arange(0, BLOCK_OL) + ol_start;
    let ol_mask = ol_range.lt(OL);

    let in_bc_base = (b * C + c) * L;
    let out_bc_base = (b * C + c) * OL;

    let p_vec = T::full::<f32>(&[BLOCK_OL], p);
    let inv_p_vec = T::full::<f32>(&[BLOCK_OL], 1.0_f32 / p);
    let eps_vec = T::full::<f32>(&[BLOCK_OL], 1e-12_f32);

    let mut acc = T::zeros::<f32>(&[BLOCK_OL]);

    let loop_bound = KL;
    for kl in 0..loop_bound {
        let il_range = ol_range * STRIDE + kl;
        let in_offsets = il_range + in_bc_base;
        let tile = T::load(
            input_ptr.add_offsets(in_offsets),
            Some(ol_mask),
            Some(T::zeros::<D>(&[BLOCK_OL])),
            &[],
            None,
            None,
            None,
            false,
        );
        let tile_f32 = T::cast::<D, f32>(tile, None, false);
        let abs_tile = T::abs(tile_f32);
        let safe_abs = T::maximum(abs_tile, eps_vec);
        // |x|^p = exp(p * log(|x|))
        let pow_tile = T::exp(p_vec * T::log(safe_abs));
        acc = acc + pow_tile;
    }

    // sum^(1/p) = exp(log(sum) / p)
    let safe_acc = T::maximum(acc, eps_vec);
    let result_f32 = T::exp(T::log(safe_acc) * inv_p_vec);
    let result = T::cast::<f32, D>(result_f32, None, false);

    let out_offsets = ol_range + out_bc_base;
    T::store(
        output_ptr.add_offsets(out_offsets),
        result,
        Some(ol_mask),
        &[],
        None,
        None,
    );
}

/// 1-D Lp-norm pooling backward pass.
///
/// `dx_i = dy * sign(x_i) * (|x_i| / max(y, ε))^(p-1) / max(y, ε)`.
///
/// Requires both the original input `x` and the forward output `y`.
/// `dx` must be zero-initialised before launch.
#[kernel]
pub fn lppool1d_backward<
    T: Triton,
    D: Float,
    const KL: i32,
    const STRIDE: i32,
    const BLOCK_OL: i32,
>(
    dy_ptr: T::Pointer<D>,
    x_ptr: T::Pointer<D>,
    y_ptr: T::Pointer<D>,
    dx_ptr: T::Pointer<D>,
    B: i32,
    C: i32,
    L: i32,
    OL: i32,
    p: f32,
) where
    T::I32Tensor: Tensor<i32, 1>,
    T::I32Tensor: Comparison<i32, BoolTensor = T::BoolTensor>,
    T::Pointer<D>: AddOffsets<i32, 1, T::I32Tensor, Output = T::Tensor<T::Pointer<D>>>,
{
    let pid = T::program_id(Axis::X);
    let num_ol_tiles = T::cdiv(OL, BLOCK_OL);

    let ol_tile = pid % num_ol_tiles;
    let bc = pid / num_ol_tiles;
    let c = bc % C;
    let b = bc / C;

    let ol_start = ol_tile * BLOCK_OL;
    let ol_range = T::arange(0, BLOCK_OL) + ol_start;
    let ol_mask = ol_range.lt(OL);

    let in_bc_base = (b * C + c) * L;
    let out_bc_base = (b * C + c) * OL;

    let pm1_vec = T::full::<f32>(&[BLOCK_OL], p - 1.0_f32);
    let eps_vec = T::full::<f32>(&[BLOCK_OL], 1e-12_f32);
    let zeros_f32 = T::zeros::<f32>(&[BLOCK_OL]);

    let dy_offsets = ol_range + out_bc_base;
    let dy_tile = T::load(
        dy_ptr.add_offsets(dy_offsets),
        Some(ol_mask),
        Some(T::zeros::<D>(&[BLOCK_OL])),
        &[],
        None,
        None,
        None,
        false,
    );
    let dy_f32 = T::cast::<D, f32>(dy_tile, None, false);

    let y_tile = T::load(
        y_ptr.add_offsets(dy_offsets),
        Some(ol_mask),
        Some(T::zeros::<D>(&[BLOCK_OL])),
        &[],
        None,
        None,
        None,
        false,
    );
    let y_f32 = T::cast::<D, f32>(y_tile, None, false);
    let safe_y = T::maximum(y_f32, eps_vec);

    let loop_bound = KL;
    for kl in 0..loop_bound {
        let il_range = ol_range * STRIDE + kl;
        let in_offsets = il_range + in_bc_base;
        let x_tile = T::load(
            x_ptr.add_offsets(in_offsets),
            Some(ol_mask),
            Some(T::zeros::<D>(&[BLOCK_OL])),
            &[],
            None,
            None,
            None,
            false,
        );
        let x_f32 = T::cast::<D, f32>(x_tile, None, false);
        let abs_x = T::abs(x_f32);
        let safe_abs = T::maximum(abs_x, eps_vec);

        // sign(x): 1.0 if x > 0, -1.0 if x < 0, 0.0 if x == 0.
        let pos = T::where_(T::gt(x_f32, zeros_f32), T::full(&[BLOCK_OL], 1.0_f32), zeros_f32);
        let neg = T::where_(T::gt(zeros_f32, x_f32), T::full(&[BLOCK_OL], 1.0_f32), zeros_f32);
        let sign_x = pos - neg;

        // (|x| / y)^(p-1) = exp((p-1) * log(|x| / y))
        let ratio = safe_abs / safe_y;
        let safe_ratio = T::maximum(ratio, eps_vec);
        let pow_ratio = T::exp(pm1_vec * T::log(safe_ratio));

        let dx_f32 = dy_f32 * sign_x * pow_ratio;
        let dx_tile = T::cast::<f32, D>(dx_f32, None, false);

        T::atomic_add(
            dx_ptr.add_offsets(in_offsets),
            dx_tile,
            Some(ol_mask),
            None,
            None,
        );
    }
}
