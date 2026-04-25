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

/// 2-D Lp-norm pooling forward pass.
///
/// Grid: `pid = ((b * C + c) * OH + oh) * num_ow_tiles + ow_tile`
///
/// **Constraints**: no padding; `OH = (H - KH) / STRIDE_H + 1`, `OW = (W - KW) / STRIDE_W + 1`.
#[kernel]
pub fn lppool2d_forward<
    T: Triton,
    D: Float,
    const KH: i32,
    const KW: i32,
    const STRIDE_H: i32,
    const STRIDE_W: i32,
    const BLOCK_OW: i32,
>(
    input_ptr: T::Pointer<D>,
    output_ptr: T::Pointer<D>,
    B: i32,
    C: i32,
    H: i32,
    W: i32,
    OH: i32,
    OW: i32,
    p: f32,
) where
    T::I32Tensor: Tensor<i32, 1>,
    T::I32Tensor: Comparison<i32, BoolTensor = T::BoolTensor>,
    T::Pointer<D>: AddOffsets<i32, 1, T::I32Tensor, Output = T::Tensor<T::Pointer<D>>>,
{
    let pid = T::program_id(Axis::X);
    let num_ow_tiles = T::cdiv(OW, BLOCK_OW);

    let ow_tile = pid % num_ow_tiles;
    let bco = pid / num_ow_tiles;
    let oh = bco % OH;
    let bc = bco / OH;
    let c = bc % C;
    let b = bc / C;

    let ow_start = ow_tile * BLOCK_OW;
    let ow_range = T::arange(0, BLOCK_OW) + ow_start;
    let ow_mask = ow_range.lt(OW);

    let in_bc_base = (b * C + c) * H * W;
    let out_bc_base = (b * C + c) * OH * OW;

    let p_vec = T::full::<f32>(&[BLOCK_OW], p);
    let inv_p_vec = T::full::<f32>(&[BLOCK_OW], 1.0_f32 / p);
    let eps_vec = T::full::<f32>(&[BLOCK_OW], 1e-12_f32);

    let mut acc = T::zeros::<f32>(&[BLOCK_OW]);

    let loop_bound = KH * KW;
    for idx in 0..loop_bound {
        let kw = idx % KW;
        let kh = idx / KW;
        let ih = oh * STRIDE_H + kh;
        let iw_range = ow_range * STRIDE_W + kw;
        let in_offsets = iw_range + (in_bc_base + ih * W);
        let tile = T::load(
            input_ptr.add_offsets(in_offsets),
            Some(ow_mask),
            Some(T::zeros::<D>(&[BLOCK_OW])),
            &[],
            None,
            None,
            None,
            false,
        );
        let tile_f32 = T::cast::<D, f32>(tile, None, false);
        let abs_tile = T::abs(tile_f32);
        let safe_abs = T::maximum(abs_tile, eps_vec);
        let pow_tile = T::exp(p_vec * T::log(safe_abs));
        acc = acc + pow_tile;
    }

    let safe_acc = T::maximum(acc, eps_vec);
    let result_f32 = T::exp(T::log(safe_acc) * inv_p_vec);
    let result = T::cast::<f32, D>(result_f32, None, false);

    let out_offsets = ow_range + (out_bc_base + oh * OW);
    T::store(
        output_ptr.add_offsets(out_offsets),
        result,
        Some(ow_mask),
        &[],
        None,
        None,
    );
}

/// 2-D Lp-norm pooling backward pass.
///
/// `dx_i = dy * sign(x_i) * (|x_i| / max(y, ε))^(p-1) / max(y, ε)`.
///
/// `dx` must be zero-initialised before launch.
#[kernel]
pub fn lppool2d_backward<
    T: Triton,
    D: Float,
    const KH: i32,
    const KW: i32,
    const STRIDE_H: i32,
    const STRIDE_W: i32,
    const BLOCK_OW: i32,
>(
    dy_ptr: T::Pointer<D>,
    x_ptr: T::Pointer<D>,
    y_ptr: T::Pointer<D>,
    dx_ptr: T::Pointer<D>,
    B: i32,
    C: i32,
    H: i32,
    W: i32,
    OH: i32,
    OW: i32,
    p: f32,
) where
    T::I32Tensor: Tensor<i32, 1>,
    T::I32Tensor: Comparison<i32, BoolTensor = T::BoolTensor>,
    T::Pointer<D>: AddOffsets<i32, 1, T::I32Tensor, Output = T::Tensor<T::Pointer<D>>>,
{
    let pid = T::program_id(Axis::X);
    let num_ow_tiles = T::cdiv(OW, BLOCK_OW);

    let ow_tile = pid % num_ow_tiles;
    let bco = pid / num_ow_tiles;
    let oh = bco % OH;
    let bc = bco / OH;
    let c = bc % C;
    let b = bc / C;

    let ow_start = ow_tile * BLOCK_OW;
    let ow_range = T::arange(0, BLOCK_OW) + ow_start;
    let ow_mask = ow_range.lt(OW);

    let in_bc_base = (b * C + c) * H * W;
    let out_bc_base = (b * C + c) * OH * OW;

    let pm1_vec = T::full::<f32>(&[BLOCK_OW], p - 1.0_f32);
    let eps_vec = T::full::<f32>(&[BLOCK_OW], 1e-12_f32);
    let zeros_f32 = T::zeros::<f32>(&[BLOCK_OW]);

    let out_offsets = ow_range + (out_bc_base + oh * OW);
    let dy_tile = T::load(
        dy_ptr.add_offsets(out_offsets),
        Some(ow_mask),
        Some(T::zeros::<D>(&[BLOCK_OW])),
        &[],
        None,
        None,
        None,
        false,
    );
    let dy_f32 = T::cast::<D, f32>(dy_tile, None, false);

    let y_tile = T::load(
        y_ptr.add_offsets(out_offsets),
        Some(ow_mask),
        Some(T::zeros::<D>(&[BLOCK_OW])),
        &[],
        None,
        None,
        None,
        false,
    );
    let y_f32 = T::cast::<D, f32>(y_tile, None, false);
    let safe_y = T::maximum(y_f32, eps_vec);

    let loop_bound = KH * KW;
    for idx in 0..loop_bound {
        let kw = idx % KW;
        let kh = idx / KW;
        let ih = oh * STRIDE_H + kh;
        let iw_range = ow_range * STRIDE_W + kw;
        let in_offsets = iw_range + (in_bc_base + ih * W);
        let x_tile = T::load(
            x_ptr.add_offsets(in_offsets),
            Some(ow_mask),
            Some(T::zeros::<D>(&[BLOCK_OW])),
            &[],
            None,
            None,
            None,
            false,
        );
        let x_f32 = T::cast::<D, f32>(x_tile, None, false);
        let abs_x = T::abs(x_f32);
        let safe_abs = T::maximum(abs_x, eps_vec);

        let pos = T::where_(T::gt(x_f32, zeros_f32), T::full(&[BLOCK_OW], 1.0_f32), zeros_f32);
        let neg = T::where_(T::gt(zeros_f32, x_f32), T::full(&[BLOCK_OW], 1.0_f32), zeros_f32);
        let sign_x = pos - neg;

        let ratio = safe_abs / safe_y;
        let safe_ratio = T::maximum(ratio, eps_vec);
        let pow_ratio = T::exp(pm1_vec * T::log(safe_ratio));

        let dx_f32 = dy_f32 * sign_x * pow_ratio;
        let dx_tile = T::cast::<f32, D>(dx_f32, None, false);

        T::atomic_add(
            dx_ptr.add_offsets(in_offsets),
            dx_tile,
            Some(ow_mask),
            None,
            None,
        );
    }
}
