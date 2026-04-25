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

/// 2-D circular padding forward pass.
///
/// Grid: `pid = ((b*C+c)*OH+oh) * num_ow_tiles + ow_tile`
#[kernel]
pub fn circular_pad2d_forward<
    T: Triton,
    D: Float,
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

    // Wrap height index (branchless: works when |ih_raw| < H)
    let ih_raw = oh - PT;
    let ih = (ih_raw + H) % H;

    let in_bc_base = (b * C + c) * H * W + ih * W;
    let out_bc_base = ((b * C + c) * OH + oh) * OW;

    let iw_raw = ow_range - PL;
    let cond_left = iw_raw.lt(0);
    let cond_right = iw_raw.ge(W);
    let in_bounds = iw_raw.ge(0) & iw_raw.lt(W);

    let iw_wrap_left = iw_raw + W;
    let iw_wrap_right = iw_raw - W;

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
        input_ptr.add_offsets(iw_wrap_left + in_bc_base),
        Some(ow_mask & cond_left),
        Some(zeros),
        &[],
        None,
        None,
        None,
        false,
    );
    let val_right = T::load(
        input_ptr.add_offsets(iw_wrap_right + in_bc_base),
        Some(ow_mask & cond_right),
        Some(zeros),
        &[],
        None,
        None,
        None,
        false,
    );

    let result = T::where_(cond_left, val_left, T::where_(cond_right, val_right, val_center));

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

/// 2-D circular padding backward pass.
#[kernel]
pub fn circular_pad2d_backward<
    T: Triton,
    D: Float,
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

    let ih_raw = oh - PT;
    let ih = (ih_raw + H) % H;

    let dy_bc_base = ((b * C + c) * OH + oh) * OW;
    let dx_bc_base = (b * C + c) * H * W + ih * W;

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

    let iw_raw = ow_range - PL;
    let cond_left = iw_raw.lt(0);
    let cond_right = iw_raw.ge(W);
    let in_bounds = iw_raw.ge(0) & iw_raw.lt(W);

    let iw_wrap_left = iw_raw + W;
    let iw_wrap_right = iw_raw - W;

    T::atomic_add(
        dx_ptr.add_offsets(iw_raw + dx_bc_base),
        dy_tile,
        Some(ow_mask & in_bounds),
        None,
        None,
    );
    T::atomic_add(
        dx_ptr.add_offsets(iw_wrap_left + dx_bc_base),
        dy_tile,
        Some(ow_mask & cond_left),
        None,
        None,
    );
    T::atomic_add(
        dx_ptr.add_offsets(iw_wrap_right + dx_bc_base),
        dy_tile,
        Some(ow_mask & cond_right),
        None,
        None,
    );
}
