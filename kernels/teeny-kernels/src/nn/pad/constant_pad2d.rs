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

/// 2-D constant padding forward pass.
///
/// Grid: `pid = ((b * C + c) * OH + oh) * num_ow_tiles + ow_tile`
///
/// `OH = PT + H + PB`, `OW = PL + W + PR`.
/// Positions outside input region are filled with `value`.
#[kernel]
pub fn constant_pad2d_forward<
    T: Triton,
    D: Num,
    const PT: i32,
    const PB: i32,
    const PL: i32,
    const PR: i32,
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
    let bc = rest / OH;
    let c = bc % C;
    let b = bc / C;

    let ow_start = ow_tile * BLOCK_OW;
    let ow_range = T::arange(0, BLOCK_OW) + ow_start;
    let ow_mask = ow_range.lt(OW);

    let ih = oh - PT;
    let iw_range = ow_range - PL;

    let value_vec = T::cast::<f32, D>(T::full::<f32>(&[BLOCK_OW], value), None, false);

    // Convert scalar ih into a uniform tensor for branchless height bounds check.
    // `ow_range * 0` zeroes the range; adding ih makes all lanes equal to ih.
    let ih_t = ow_range * 0 + ih;
    let h_in_bounds = ih_t.ge(0) & ih_t.lt(H);
    let w_in_bounds = iw_range.ge(0) & iw_range.lt(W);

    // in_bc_base uses the raw (possibly out-of-range) ih, but combined_mask
    // is all-false when ih is out of bounds, so no actual OOB access occurs.
    let in_bc_base = (b * C + c) * H * W + ih * W;
    let out_bc_base = ((b * C + c) * OH + oh) * OW;

    let combined_mask = ow_mask & h_in_bounds & w_in_bounds;

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

/// 2-D constant padding backward pass.
#[kernel]
pub fn constant_pad2d_backward<
    T: Triton,
    D: Num,
    const PT: i32,
    const PB: i32,
    const PL: i32,
    const PR: i32,
    const BLOCK_OW: i32,
>(
    dy_ptr: T::Pointer<D>,
    dx_ptr: T::Pointer<D>,
    B: i32,
    C: i32,
    H: i32,
    W: i32,
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
    let bc = rest / OH;
    let c = bc % C;
    let b = bc / C;

    let ow_start = ow_tile * BLOCK_OW;
    let ow_range = T::arange(0, BLOCK_OW) + ow_start;
    let ow_mask = ow_range.lt(OW);

    let ih = oh - PT;
    let iw_range = ow_range - PL;

    // Convert scalar ih into a uniform tensor for branchless height bounds check.
    let ih_t = ow_range * 0 + ih;
    let h_in_bounds = ih_t.ge(0) & ih_t.lt(H);
    let w_in_bounds = iw_range.ge(0) & iw_range.lt(W);

    let dy_bc_base = ((b * C + c) * OH + oh) * OW;
    let dx_bc_base = (b * C + c) * H * W + ih * W;

    let store_mask = ow_mask & h_in_bounds & w_in_bounds;

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

pub struct ConstantPad2dOp<'a, T: Num> {
    pub forward: ConstantPad2dForward<T>,
    pub backward: ConstantPad2dBackward<T>,
    _marker: core::marker::PhantomData<&'a ()>,
}
