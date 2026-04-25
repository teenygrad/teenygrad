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

use teeny_core::dtype::Num;
use teeny_macros::kernel;
use teeny_triton::triton::{
    types::{AddOffsets, Comparison, Tensor},
    *,
};

/// 2-D convolution forward pass.
///
/// Grid: one CTA per (b, c_out, oh, ow-tile):
///   `pid = ((b * C_OUT + c_out) * OH + oh) * num_ow_tiles + ow_tile`
///
/// Each CTA computes a BLOCK_OW-wide strip of output columns by iterating over
/// all `C_IN * KH * KW` combinations in a flat loop.  The weight for each
/// `(c_in, kh, kw)` is loaded as a [1] tensor and broadcast to `[BLOCK_OW]`.
///
/// **Constraints**: no padding; `OH = (H - KH) / STRIDE_H + 1`, `OW = (W - KW) / STRIDE_W + 1`.
#[kernel]
pub fn conv2d_forward<
    T: Triton,
    D: Num,
    const KH: i32,
    const KW: i32,
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
    H: i32,
    W: i32,
    OH: i32,
    OW: i32,
) where
    T::I32Tensor: Tensor<i32, 1>,
    T::I32Tensor: Comparison<i32, BoolTensor = T::BoolTensor>,
    T::Pointer<D>: AddOffsets<i32, 1, T::I32Tensor, Output = T::Tensor<T::Pointer<D>>>,
{
    let pid = T::program_id(Axis::X);
    let num_ow_tiles = T::cdiv(OW, BLOCK_OW);

    // Decode flat pid → (b, c_out, oh, ow_tile).
    let ow_tile = pid % num_ow_tiles;
    let bco = pid / num_ow_tiles;
    let oh = bco % OH;
    let bc = bco / OH;
    let c_out = bc % C_OUT;
    let b = bc / C_OUT;

    let ow_start = ow_tile * BLOCK_OW;
    let ow_range = T::arange(0, BLOCK_OW) + ow_start;
    let ow_mask = ow_range.lt(OW);

    let out_bc_base = (b * C_OUT + c_out) * OH * OW;

    let mut acc = T::zeros::<D>(&[BLOCK_OW]);

    // Flat loop over all C_IN * KH * KW combinations.
    let loop_bound = C_IN * KH * KW;
    for idx in 0..loop_bound {
        let kw = idx % KW;
        let kh_cin = idx / KW;
        let kh = kh_cin % KH;
        let c_in = kh_cin / KH;

        // Load BLOCK_OW input elements.
        let ih = oh * STRIDE_H + kh;
        let iw_range = ow_range * STRIDE_W + kw;
        let x_offsets = iw_range + ((b * C_IN + c_in) * H * W + ih * W);
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

        // Load scalar weight and broadcast to [BLOCK_OW].
        let w_idx = ((c_out * C_IN + c_in) * KH + kh) * KW + kw;
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

    let out_offsets = ow_range + (out_bc_base + oh * OW);
    T::store(
        y_ptr.add_offsets(out_offsets),
        acc,
        Some(ow_mask),
        &[],
        None,
        None,
    );
}

/// 2-D convolution backward pass — gradient with respect to input (`dx`).
///
/// Uses the same grid as the forward pass (over output positions) and scatters
/// the gradient back to the input via `atomic_add`, which correctly handles
/// overlapping receptive fields when `STRIDE < kernel size`.
///
/// Grid: `pid = ((b * C_OUT + c_out) * OH + oh) * num_ow_tiles + ow_tile`
///
/// **Constraints**: same as `conv2d_forward`.
#[kernel]
pub fn conv2d_backward_dx<
    T: Triton,
    D: Num,
    const KH: i32,
    const KW: i32,
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
    H: i32,
    W: i32,
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
    let bco = pid / num_ow_tiles;
    let oh = bco % OH;
    let bc = bco / OH;
    let c_out = bc % C_OUT;
    let b = bc / C_OUT;

    let ow_start = ow_tile * BLOCK_OW;
    let ow_range = T::arange(0, BLOCK_OW) + ow_start;
    let ow_mask = ow_range.lt(OW);

    // Load the upstream gradient tile for this output row.
    let dy_offsets = ow_range + ((b * C_OUT + c_out) * OH * OW + oh * OW);
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

    // Scatter gradient to input via flat loop over C_IN * KH * KW.
    let loop_bound = C_IN * KH * KW;
    for idx in 0..loop_bound {
        let kw = idx % KW;
        let kh_cin = idx / KW;
        let kh = kh_cin % KH;
        let c_in = kh_cin / KH;

        let w_idx = ((c_out * C_IN + c_in) * KH + kh) * KW + kw;
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

        let ih = oh * STRIDE_H + kh;
        let iw_range = ow_range * STRIDE_W + kw;
        let dx_offsets = iw_range + ((b * C_IN + c_in) * H * W + ih * W);
        T::atomic_add(
            dx_ptr.add_offsets(dx_offsets),
            grad_tile,
            Some(ow_mask),
            None,
            None,
        );
    }
}

/// 2-D convolution backward pass — gradient with respect to weights (`dw`).
///
/// Uses the same grid as the forward pass.  Each CTA at `(b, c_out, oh, ow_tile)`
/// accumulates partial sums into `dw` via `atomic_add` (one per `(c_in, kh, kw)`
/// combination).
///
/// Grid: `pid = ((b * C_OUT + c_out) * OH + oh) * num_ow_tiles + ow_tile`
///
/// **Constraints**: `dw` must be zero-initialised before launch; same spatial
/// constraints as `conv2d_forward`.
#[kernel]
pub fn conv2d_backward_dw<
    T: Triton,
    D: Num,
    const KH: i32,
    const KW: i32,
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
    H: i32,
    W: i32,
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
    let bco = pid / num_ow_tiles;
    let oh = bco % OH;
    let bc = bco / OH;
    let c_out = bc % C_OUT;
    let b = bc / C_OUT;

    let ow_start = ow_tile * BLOCK_OW;
    let ow_range = T::arange(0, BLOCK_OW) + ow_start;
    let ow_mask = ow_range.lt(OW);

    // Load dy tile for this output row.
    let dy_offsets = ow_range + ((b * C_OUT + c_out) * OH * OW + oh * OW);
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

    let ih_base = oh * STRIDE_H;

    // Flat loop over C_IN * KH * KW to compute partial weight gradients.
    let loop_bound = C_IN * KH * KW;
    for idx in 0..loop_bound {
        let kw = idx % KW;
        let kh_cin = idx / KW;
        let kh = kh_cin % KH;
        let c_in = kh_cin / KH;

        let ih = ih_base + kh;
        let iw_range = ow_range * STRIDE_W + kw;
        let x_offsets = iw_range + ((b * C_IN + c_in) * H * W + ih * W);
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

        // Partial sum for this weight element: scalar dot product over the ow tile.
        let partial = T::sum(dy_tile * x_tile, Some(0), false);
        let partial_1 = T::expand_dims(partial, 0);

        let w_idx = ((c_out * C_IN + c_in) * KH + kh) * KW + kw;
        let dw_off = T::arange(0, 1) + w_idx;
        T::atomic_add(dw_ptr.add_offsets(dw_off), partial_1, None, None, None);
    }
}

pub struct Conv2dOp<'a, T: Num> {
    pub forward: Conv2dForward<T>,
    pub backward_dx: Conv2dBackwardDx<T>,
    pub backward_dw: Conv2dBackwardDw<T>,
    _marker: core::marker::PhantomData<&'a ()>,
}
