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

/// 3-D convolution forward pass.
///
/// Grid: one CTA per (b, c_out, od, oh, ow-tile):
///   `pid = (((b * C_OUT + c_out) * OD + od) * OH + oh) * num_ow_tiles + ow_tile`
///
/// Each CTA computes a BLOCK_OW-wide strip of output width positions by
/// iterating over all `C_IN * KD * KH * KW` combinations.
///
/// **Constraints**: no padding;
/// `OD = (D - KD) / STRIDE_D + 1`, `OH = (H - KH) / STRIDE_H + 1`,
/// `OW = (W - KW) / STRIDE_W + 1`.
#[kernel]
pub fn conv3d_forward<
    T: Triton,
    D: Float,
    const KD: i32,
    const KH: i32,
    const KW: i32,
    const STRIDE_D: i32,
    const STRIDE_H: i32,
    const STRIDE_W: i32,
    const BLOCK_OW: i32,
>(
    x_ptr: T::Pointer<D>,
    w_ptr: T::Pointer<D>,
    y_ptr: T::Pointer<D>,
    B: i32,
    C_IN: i32,
    C_OUT: i32,
    Dv: i32,
    H: i32,
    W: i32,
    OD: i32,
    OH: i32,
    OW: i32,
) where
    T::I32Tensor: Tensor<i32, 1>,
    T::I32Tensor: Comparison<i32, BoolTensor = T::BoolTensor>,
    T::Pointer<D>: AddOffsets<i32, 1, T::I32Tensor, Output = T::Tensor<T::Pointer<D>>>,
{
    let pid = T::program_id(Axis::X);
    let num_ow_tiles = T::cdiv(OW, BLOCK_OW);

    // Decode flat pid → (b, c_out, od, oh, ow_tile).
    let ow_tile = pid % num_ow_tiles;
    let rest = pid / num_ow_tiles;
    let oh = rest % OH;
    let rest2 = rest / OH;
    let od = rest2 % OD;
    let bco = rest2 / OD;
    let c_out = bco % C_OUT;
    let b = bco / C_OUT;

    let ow_start = ow_tile * BLOCK_OW;
    let ow_range = T::arange(0, BLOCK_OW) + ow_start;
    let ow_mask = ow_range.lt(OW);

    let out_base = ((b * C_OUT + c_out) * OD * OH * OW) + od * OH * OW + oh * OW;

    let mut acc = T::zeros::<D>(&[BLOCK_OW]);

    let loop_bound = C_IN * KD * KH * KW;
    for idx in 0..loop_bound {
        let kw = idx % KW;
        let tmp = idx / KW;
        let kh = tmp % KH;
        let tmp2 = tmp / KH;
        let kd = tmp2 % KD;
        let c_in = tmp2 / KD;

        let id = od * STRIDE_D + kd;
        let ih = oh * STRIDE_H + kh;
        let iw_range = ow_range * STRIDE_W + kw;
        let x_offsets =
            iw_range + ((b * C_IN + c_in) * Dv * H * W + id * H * W + ih * W);
        let x_tile = T::load(
            x_ptr.add_offsets(x_offsets),
            Some(ow_mask),
            Some(T::zeros::<D>(&[BLOCK_OW])),
            &[],
            None,
            None,
            None,
            false,
        );

        let w_idx = (((c_out * C_IN + c_in) * KD + kd) * KH + kh) * KW + kw;
        let w_off = T::arange(0, 1) + w_idx;
        let w_1 = T::load(
            w_ptr.add_offsets(w_off),
            None,
            None,
            &[],
            None,
            None,
            None,
            false,
        );
        let w_tile = T::broadcast_to(w_1, &[BLOCK_OW]);

        acc = acc + x_tile * w_tile;
    }

    let out_offsets = ow_range + out_base;
    T::store(
        y_ptr.add_offsets(out_offsets),
        acc,
        Some(ow_mask),
        &[],
        None,
        None,
    );
}

/// 3-D convolution backward pass — gradient with respect to input (`dx`).
///
/// Grid: `pid = (((b * C_OUT + c_out) * OD + od) * OH + oh) * num_ow_tiles + ow_tile`
///
/// Scatters gradient via `atomic_add` to handle overlapping receptive fields.
#[kernel]
pub fn conv3d_backward_dx<
    T: Triton,
    D: Float,
    const KD: i32,
    const KH: i32,
    const KW: i32,
    const STRIDE_D: i32,
    const STRIDE_H: i32,
    const STRIDE_W: i32,
    const BLOCK_OW: i32,
>(
    dy_ptr: T::Pointer<D>,
    w_ptr: T::Pointer<D>,
    dx_ptr: T::Pointer<D>,
    B: i32,
    C_IN: i32,
    C_OUT: i32,
    Dv: i32,
    H: i32,
    W: i32,
    OD: i32,
    OH: i32,
    OW: i32,
) where
    T::I32Tensor: Tensor<i32, 1>,
    T::I32Tensor: Comparison<i32, BoolTensor = T::BoolTensor>,
    T::Pointer<D>: AddOffsets<i32, 1, T::I32Tensor, Output = T::Tensor<T::Pointer<D>>>,
{
    let pid = T::program_id(Axis::X);
    let num_ow_tiles = T::cdiv(OW, BLOCK_OW);

    let ow_tile = pid % num_ow_tiles;
    let rest = pid / num_ow_tiles;
    let oh = rest % OH;
    let rest2 = rest / OH;
    let od = rest2 % OD;
    let bco = rest2 / OD;
    let c_out = bco % C_OUT;
    let b = bco / C_OUT;

    let ow_start = ow_tile * BLOCK_OW;
    let ow_range = T::arange(0, BLOCK_OW) + ow_start;
    let ow_mask = ow_range.lt(OW);

    let dy_offsets =
        ow_range + ((b * C_OUT + c_out) * OD * OH * OW + od * OH * OW + oh * OW);
    let dy_tile = T::load(
        dy_ptr.add_offsets(dy_offsets),
        Some(ow_mask),
        Some(T::zeros::<D>(&[BLOCK_OW])),
        &[],
        None,
        None,
        None,
        false,
    );

    let loop_bound = C_IN * KD * KH * KW;
    for idx in 0..loop_bound {
        let kw = idx % KW;
        let tmp = idx / KW;
        let kh = tmp % KH;
        let tmp2 = tmp / KH;
        let kd = tmp2 % KD;
        let c_in = tmp2 / KD;

        let w_idx = (((c_out * C_IN + c_in) * KD + kd) * KH + kh) * KW + kw;
        let w_off = T::arange(0, 1) + w_idx;
        let w_1 = T::load(
            w_ptr.add_offsets(w_off),
            None,
            None,
            &[],
            None,
            None,
            None,
            false,
        );
        let w_tile = T::broadcast_to(w_1, &[BLOCK_OW]);

        let grad_tile = dy_tile * w_tile;

        let id = od * STRIDE_D + kd;
        let ih = oh * STRIDE_H + kh;
        let iw_range = ow_range * STRIDE_W + kw;
        let dx_offsets =
            iw_range + ((b * C_IN + c_in) * Dv * H * W + id * H * W + ih * W);
        T::atomic_add(
            dx_ptr.add_offsets(dx_offsets),
            grad_tile,
            Some(ow_mask),
            None,
            None,
        );
    }
}

/// 3-D convolution backward pass — gradient with respect to weights (`dw`).
///
/// Grid: `pid = (((b * C_OUT + c_out) * OD + od) * OH + oh) * num_ow_tiles + ow_tile`
///
/// `dw` must be zero-initialised before launch.
#[kernel]
pub fn conv3d_backward_dw<
    T: Triton,
    D: Float,
    const KD: i32,
    const KH: i32,
    const KW: i32,
    const STRIDE_D: i32,
    const STRIDE_H: i32,
    const STRIDE_W: i32,
    const BLOCK_OW: i32,
>(
    dy_ptr: T::Pointer<D>,
    x_ptr: T::Pointer<D>,
    dw_ptr: T::Pointer<D>,
    B: i32,
    C_IN: i32,
    C_OUT: i32,
    Dv: i32,
    H: i32,
    W: i32,
    OD: i32,
    OH: i32,
    OW: i32,
) where
    T::I32Tensor: Tensor<i32, 1>,
    T::I32Tensor: Comparison<i32, BoolTensor = T::BoolTensor>,
    T::Pointer<D>: AddOffsets<i32, 1, T::I32Tensor, Output = T::Tensor<T::Pointer<D>>>,
{
    let pid = T::program_id(Axis::X);
    let num_ow_tiles = T::cdiv(OW, BLOCK_OW);

    let ow_tile = pid % num_ow_tiles;
    let rest = pid / num_ow_tiles;
    let oh = rest % OH;
    let rest2 = rest / OH;
    let od = rest2 % OD;
    let bco = rest2 / OD;
    let c_out = bco % C_OUT;
    let b = bco / C_OUT;

    let ow_start = ow_tile * BLOCK_OW;
    let ow_range = T::arange(0, BLOCK_OW) + ow_start;
    let ow_mask = ow_range.lt(OW);

    let dy_offsets =
        ow_range + ((b * C_OUT + c_out) * OD * OH * OW + od * OH * OW + oh * OW);
    let dy_tile = T::load(
        dy_ptr.add_offsets(dy_offsets),
        Some(ow_mask),
        Some(T::zeros::<D>(&[BLOCK_OW])),
        &[],
        None,
        None,
        None,
        false,
    );

    let loop_bound = C_IN * KD * KH * KW;
    for idx in 0..loop_bound {
        let kw = idx % KW;
        let tmp = idx / KW;
        let kh = tmp % KH;
        let tmp2 = tmp / KH;
        let kd = tmp2 % KD;
        let c_in = tmp2 / KD;

        let id = od * STRIDE_D + kd;
        let ih = oh * STRIDE_H + kh;
        let iw_range = ow_range * STRIDE_W + kw;
        let x_offsets =
            iw_range + ((b * C_IN + c_in) * Dv * H * W + id * H * W + ih * W);
        let x_tile = T::load(
            x_ptr.add_offsets(x_offsets),
            Some(ow_mask),
            Some(T::zeros::<D>(&[BLOCK_OW])),
            &[],
            None,
            None,
            None,
            false,
        );

        let partial = T::sum(dy_tile * x_tile, Some(0), false);
        let partial_1 = T::expand_dims(partial, 0);

        let w_idx = (((c_out * C_IN + c_in) * KD + kd) * KH + kh) * KW + kw;
        let dw_off = T::arange(0, 1) + w_idx;
        T::atomic_add(dw_ptr.add_offsets(dw_off), partial_1, None, None, None);
    }
}
