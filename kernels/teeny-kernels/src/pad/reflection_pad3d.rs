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

use core::ops::{BitAnd, BitOr};

use teeny_core::dtype::Num;
use teeny_macros::kernel;
use teeny_triton::triton::{
    types::{AddOffsets, Comparison, Tensor},
    *,
};

/// 3-D reflection padding forward pass.
///
/// Grid: `pid = (((b*C+c)*OD+od)*OH+oh)*num_ow_tiles + ow_tile`
#[kernel]
pub fn reflection_pad3d_forward<
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
    input_ptr: T::Pointer<D>,
    output_ptr: T::Pointer<D>,
    B: i32,
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
    let id = if id_raw < 0 { -id_raw } else if id_raw >= Dv { 2 * (Dv - 1) - id_raw } else { id_raw };

    let ih_raw = oh - PH1;
    let ih = if ih_raw < 0 { -ih_raw } else if ih_raw >= H { 2 * (H - 1) - ih_raw } else { ih_raw };

    let in_bc_base = ((b * C + c) * Dv + id) * H * W + ih * W;
    let out_bc_base = (((b * C + c) * OD + od) * OH + oh) * OW;

    let iw_raw = ow_range - PW1;
    let left_cond = iw_raw.lt(0);
    let right_cond = iw_raw.ge(W);
    let in_bounds = iw_raw.ge(0) & iw_raw.lt(W);

    let iw_left = iw_raw * (-1);
    let iw_right = iw_raw * (-1) + (2 * (W - 1));

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
        input_ptr.add_offsets(iw_left + in_bc_base),
        Some(ow_mask & left_cond),
        Some(zeros),
        &[],
        None,
        None,
        None,
        false,
    );
    let val_right = T::load(
        input_ptr.add_offsets(iw_right + in_bc_base),
        Some(ow_mask & right_cond),
        Some(zeros),
        &[],
        None,
        None,
        None,
        false,
    );

    let result = T::where_(left_cond, val_left, T::where_(right_cond, val_right, val_center));

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

/// 3-D reflection padding backward pass.
#[kernel]
pub fn reflection_pad3d_backward<
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
    dy_ptr: T::Pointer<D>,
    dx_ptr: T::Pointer<D>,
    B: i32,
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
    let id = if id_raw < 0 { -id_raw } else if id_raw >= Dv { 2 * (Dv - 1) - id_raw } else { id_raw };

    let ih_raw = oh - PH1;
    let ih = if ih_raw < 0 { -ih_raw } else if ih_raw >= H { 2 * (H - 1) - ih_raw } else { ih_raw };

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
    let left_cond = iw_raw.lt(0);
    let right_cond = iw_raw.ge(W);
    let in_bounds = iw_raw.ge(0) & iw_raw.lt(W);

    let iw_left = iw_raw * (-1);
    let iw_right = iw_raw * (-1) + (2 * (W - 1));

    T::atomic_add(
        dx_ptr.add_offsets(iw_raw + dx_bc_base),
        dy_tile,
        Some(ow_mask & in_bounds),
        None,
        None,
    );
    T::atomic_add(
        dx_ptr.add_offsets(iw_left + dx_bc_base),
        dy_tile,
        Some(ow_mask & left_cond),
        None,
        None,
    );
    T::atomic_add(
        dx_ptr.add_offsets(iw_right + dx_bc_base),
        dy_tile,
        Some(ow_mask & right_cond),
        None,
        None,
    );
}

pub struct ReflectionPad3dOp<'a, T: Num> {
    pub forward: ReflectionPad3dForward<T>,
    pub backward: ReflectionPad3dBackward<T>,
    _marker: core::marker::PhantomData<&'a ()>,
}
