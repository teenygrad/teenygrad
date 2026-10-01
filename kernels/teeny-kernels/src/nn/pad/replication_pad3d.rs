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

#![allow(non_snake_case)]

use core::ops::{BitAnd, BitOr};

use teeny_core::dtype::Num;
use teeny_macros::{kernel, tiled_kernel};
use teeny_triton::triton::{
    types::{AddOffsets, Comparison, Tensor},
    *,
};

/// 3-D replication padding forward pass.
///
/// Grid: `pid = (((b*C+c)*OD+od)*OH+oh)*num_ow_tiles + ow_tile`
#[tiled_kernel]
// teenygrad-1tl.5. Padding is a window of `stride = 1, kernel = 1`: each
// output element reads exactly one input element, at an origin shifted by the
// leading pad, so `(block - 1) * 1 + 1 = block` -- the tile's own width. Only
// the innermost axis is blocked, so the others carry the literal block 1,
// their `pid` index being a scalar.
//
// Exact for an interior tile, which is what the receptive field describes. A
// tile overlapping the pad region reads *fewer* distinct input elements, and
// the four families differ in where the out-of-range lanes land -- masked off
// (constant), mirrored (reflection), clamped (replication) or wrapped
// (circular). That changes which elements are read, not how many, so `block`
// stays a correct upper bound on the footprint.
pub fn replication_pad3d_forward<
    T: Triton,
    D: Num,
    const PD1: i32,
    const PD2: i32,
    const PH1: i32,
    const PH2: i32,
    const PW1: i32,
    const PW2: i32,
    const BLOCK_OW: i32,
>(
    #[tile(name = "B", extent = _B)]
    #[tile(extent = C)]
    #[tile(
        extent = Dv,
        window(stride = 1, pad = PD1, kernel = 1, output = OD)
    )]
    #[tile(
        extent = H,
        window(stride = 1, pad = PH1, kernel = 1, output = OH)
    )]
    #[tile(
        block = BLOCK_OW,
        extent = W,
        window(stride = 1, pad = PW1, kernel = 1, output = OW)
    )]
    input_ptr: In<T::Pointer<D>>,
    #[tile(name = "B", extent = _B)]
    #[tile(extent = C)]
    #[tile(extent = OD)]
    #[tile(extent = OH)]
    #[tile(block = BLOCK_OW, extent = OW)]
    output_ptr: Out<T::Pointer<D>>,
    _B: i32,
    C: i32,
    Dv: i32,
    H: i32,
    W: i32,
    OD: i32,
    OH: i32,
    OW: i32,
) where
    T::I32Tensor: Tensor<i32, 1>,
    T::I32Tensor: Comparison<i32, BoolTensor = T::BoolTensor>,
    T::BoolTensor: BitAnd<Output = T::BoolTensor>,
    T::BoolTensor: BitOr<Output = T::BoolTensor>,
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
    let c = bco % C;
    let b = bco / C;

    let ow_start = ow_tile * BLOCK_OW;
    let ow_range = T::arange(0, BLOCK_OW) + ow_start;
    let ow_mask = ow_range.lt(OW);

    let id_raw = od - PD1;
    let id = if id_raw < 0 {
        0
    } else if id_raw >= Dv {
        Dv - 1
    } else {
        id_raw
    };

    let ih_raw = oh - PH1;
    let ih = if ih_raw < 0 {
        0
    } else if ih_raw >= H {
        H - 1
    } else {
        ih_raw
    };

    let in_bc_base = ((b * C + c) * Dv + id) * H * W + ih * W;
    let out_bc_base = (((b * C + c) * OD + od) * OH + oh) * OW;

    let iw_raw = ow_range - PW1;
    let cond_left = iw_raw.lt(0);
    let cond_right = iw_raw.ge(W);
    let in_bounds = iw_raw.ge(0) & iw_raw.lt(W);

    #[allow(clippy::erasing_op)]
    let zero_iw = iw_raw * 0;
    #[allow(clippy::erasing_op)]
    let wm1_iw = iw_raw * 0 + (W - 1);

    let zeros = T::zeros::<D>(&[BLOCK_OW]);
    let val_center = T::load(
        input_ptr.add_offsets(iw_raw + in_bc_base),
        Some(ow_mask & in_bounds),
        Some(zeros),
        &[],
        None,
        None,
        None,
        false,
    );
    let val_left = T::load(
        input_ptr.add_offsets(zero_iw + in_bc_base),
        Some(ow_mask & cond_left),
        Some(zeros),
        &[],
        None,
        None,
        None,
        false,
    );
    let val_right = T::load(
        input_ptr.add_offsets(wm1_iw + in_bc_base),
        Some(ow_mask & cond_right),
        Some(zeros),
        &[],
        None,
        None,
        None,
        false,
    );

    let result = T::where_(
        cond_left,
        val_left,
        T::where_(cond_right, val_right, val_center),
    );

    let out_offsets = ow_range + out_bc_base;
    T::store(
        output_ptr.add_offsets(out_offsets),
        result,
        Some(ow_mask),
        &[],
        None,
        None,
    );
}

/// 3-D replication padding backward pass.
#[kernel]
pub fn replication_pad3d_backward<
    T: Triton,
    D: Num,
    const PD1: i32,
    const PD2: i32,
    const PH1: i32,
    const PH2: i32,
    const PW1: i32,
    const PW2: i32,
    const BLOCK_OW: i32,
>(
    dy_ptr: In<T::Pointer<D>>,
    dx_ptr: Out<T::Pointer<D>>,
    _B: i32,
    C: i32,
    Dv: i32,
    H: i32,
    W: i32,
    OD: i32,
    OH: i32,
    OW: i32,
) where
    T::I32Tensor: Tensor<i32, 1>,
    T::I32Tensor: Comparison<i32, BoolTensor = T::BoolTensor>,
    T::BoolTensor: BitAnd<Output = T::BoolTensor>,
    T::BoolTensor: BitOr<Output = T::BoolTensor>,
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
    let c = bco % C;
    let b = bco / C;

    let ow_start = ow_tile * BLOCK_OW;
    let ow_range = T::arange(0, BLOCK_OW) + ow_start;
    let ow_mask = ow_range.lt(OW);

    let id_raw = od - PD1;
    let id = if id_raw < 0 {
        0
    } else if id_raw >= Dv {
        Dv - 1
    } else {
        id_raw
    };

    let ih_raw = oh - PH1;
    let ih = if ih_raw < 0 {
        0
    } else if ih_raw >= H {
        H - 1
    } else {
        ih_raw
    };

    let dy_bc_base = (((b * C + c) * OD + od) * OH + oh) * OW;
    let dx_bc_base = ((b * C + c) * Dv + id) * H * W + ih * W;

    let dy_offsets = ow_range + dy_bc_base;
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

    let iw_raw = ow_range - PW1;
    let cond_left = iw_raw.lt(0);
    let cond_right = iw_raw.ge(W);
    let in_bounds = iw_raw.ge(0) & iw_raw.lt(W);

    #[allow(clippy::erasing_op)]
    let zero_iw = iw_raw * 0;
    #[allow(clippy::erasing_op)]
    let wm1_iw = iw_raw * 0 + (W - 1);

    T::atomic_add(
        dx_ptr.add_offsets(iw_raw + dx_bc_base),
        dy_tile,
        Some(ow_mask & in_bounds),
        None,
        None,
    );
    T::atomic_add(
        dx_ptr.add_offsets(zero_iw + dx_bc_base),
        dy_tile,
        Some(ow_mask & cond_left),
        None,
        None,
    );
    T::atomic_add(
        dx_ptr.add_offsets(wm1_iw + dx_bc_base),
        dy_tile,
        Some(ow_mask & cond_right),
        None,
        None,
    );
}

pub struct ReplicationPad3dOp<'a, T: Num> {
    pub forward: ReplicationPad3dForward<T>,
    pub backward: ReplicationPad3dBackward<T>,
    _marker: core::marker::PhantomData<&'a ()>,
}
