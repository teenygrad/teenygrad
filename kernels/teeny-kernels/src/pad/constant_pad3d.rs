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

use teeny_core::dtype::Float;
use teeny_macros::kernel;
use teeny_triton::triton::{
    types::{AddOffsets, Comparison, Tensor},
    *,
};

/// 3-D constant padding forward pass.
///
/// Grid: `pid = (((b*C+c)*OD+od)*OH+oh)*num_ow_tiles + ow_tile`
///
/// `OD = PD1+D+PD2`, `OH = PH1+H+PH2`, `OW = PW1+W+PW2`.
#[kernel]
pub fn constant_pad3d_forward<
    T: Triton,
    D: Float,
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
    value: f32,
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

    let id = od - PD1;
    let ih = oh - PH1;
    let iw_range = ow_range - PW1;

    let value_vec = T::cast::<f32, D>(T::full::<f32>(&[BLOCK_OW], value), None, false);

    // Convert scalar id, ih into uniform tensors for branchless depth/height bounds checks.
    // `ow_range * 0` zeroes the range; adding the scalar makes all lanes equal to it.
    // in_bc_base uses the raw (possibly out-of-range) id/ih, but combined_mask is
    // all-false when either is out of bounds, so no actual OOB access occurs.
    let id_t = ow_range * 0 + id;
    let ih_t = ow_range * 0 + ih;
    let d_in_bounds = id_t.ge(0) & id_t.lt(Dv);
    let h_in_bounds = ih_t.ge(0) & ih_t.lt(H);
    let w_in_bounds = iw_range.ge(0) & iw_range.lt(W);

    let in_bc_base = ((b * C + c) * Dv + id) * H * W + ih * W;
    let out_bc_base = (((b * C + c) * OD + od) * OH + oh) * OW;

    let combined_mask = ow_mask & d_in_bounds & h_in_bounds & w_in_bounds;

    let tile = T::load(
        input_ptr.add_offsets(iw_range + in_bc_base),
        Some(combined_mask),
        Some(value_vec),
        &[],
        None,
        None,
        None,
        false,
    );
    let result = T::where_(combined_mask, tile, value_vec);

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

/// 3-D constant padding backward pass.
#[kernel]
pub fn constant_pad3d_backward<
    T: Triton,
    D: Float,
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

    let id = od - PD1;
    let ih = oh - PH1;
    let iw_range = ow_range - PW1;

    // Convert scalar id, ih into uniform tensors for branchless depth/height bounds checks.
    let id_t = ow_range * 0 + id;
    let ih_t = ow_range * 0 + ih;
    let d_in_bounds = id_t.ge(0) & id_t.lt(Dv);
    let h_in_bounds = ih_t.ge(0) & ih_t.lt(H);
    let w_in_bounds = iw_range.ge(0) & iw_range.lt(W);

    let dy_bc_base = (((b * C + c) * OD + od) * OH + oh) * OW;
    let dx_bc_base = ((b * C + c) * Dv + id) * H * W + ih * W;

    let store_mask = ow_mask & d_in_bounds & h_in_bounds & w_in_bounds;

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

    let dx_offsets = iw_range + dx_bc_base;
    T::store(
        dx_ptr.add_offsets(dx_offsets),
        dy_tile,
        Some(store_mask),
        &[],
        None,
        None,
    );
}

pub struct ConstantPad3dOp<'a, T: Float> {
    pub forward: ConstantPad3dForward<T>,
    pub backward: ConstantPad3dBackward<T>,
    _marker: core::marker::PhantomData<&'a ()>,
}
