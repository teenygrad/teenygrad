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

/// 1-D constant padding forward pass.
///
/// Grid: `pid = (b * C + c) * num_ol_tiles + ol_tile`
///
/// Output length `OL = PAD_LEFT + L + PAD_RIGHT`.
/// Positions outside `[PAD_LEFT, PAD_LEFT + L)` are filled with `value`.
#[kernel]
pub fn constant_pad1d_forward<
    T: Triton,
    D: Num,
    const PAD_LEFT: i32,
    const PAD_RIGHT: i32,
    const BLOCK_OL: i32,
>(
    input_ptr: T::Pointer<D>,
    output_ptr: T::Pointer<D>,
    B: i32,
    C: i32,
    L: i32,
    OL: i32,
    value: f32,
) where
    T::I32Tensor: Tensor<i32, 1>,
    T::I32Tensor: Comparison<i32, BoolTensor = T::BoolTensor>,
    T::BoolTensor: BitAnd<Output = T::BoolTensor>,
    T::BoolTensor: BitOr<Output = T::BoolTensor>,
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

    let ip_range = ol_range - PAD_LEFT;
    let in_bounds = ip_range.ge(0) & ip_range.lt(L);
    let combined_mask = ol_mask & in_bounds;

    let value_vec = T::cast::<f32, D>(T::full::<f32>(&[BLOCK_OL], value), None, false);
    let tile = T::load(
        input_ptr.add_offsets(ip_range + in_bc_base),
        Some(combined_mask),
        Some(value_vec),
        &[],
        None,
        None,
        None,
        false,
    );
    let result = T::where_(ol_mask & in_bounds, tile, value_vec);

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

/// 1-D constant padding backward pass.
///
/// Gradient flows only through input positions; padded positions have zero grad.
/// Tile over output positions and store `dy` back to `dx` at input offset.
/// `dx` must be zero-initialised before launch.
#[kernel]
pub fn constant_pad1d_backward<
    T: Triton,
    D: Num,
    const PAD_LEFT: i32,
    const PAD_RIGHT: i32,
    const BLOCK_OL: i32,
>(
    dy_ptr: T::Pointer<D>,
    dx_ptr: T::Pointer<D>,
    B: i32,
    C: i32,
    L: i32,
    OL: i32,
) where
    T::I32Tensor: Tensor<i32, 1>,
    T::I32Tensor: Comparison<i32, BoolTensor = T::BoolTensor>,
    T::BoolTensor: BitAnd<Output = T::BoolTensor>,
    T::BoolTensor: BitOr<Output = T::BoolTensor>,
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

    let dy_bc_base = (b * C + c) * OL;
    let dx_bc_base = (b * C + c) * L;

    let ip_range = ol_range - PAD_LEFT;
    let in_bounds = ip_range.ge(0) & ip_range.lt(L);
    let load_mask = ol_mask & in_bounds;

    let dy_offsets = ol_range + dy_bc_base;
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

    let dx_offsets = ip_range + dx_bc_base;
    T::store(
        dx_ptr.add_offsets(dx_offsets),
        dy_tile,
        Some(load_mask),
        &[],
        None,
        None,
    );
}

pub struct ConstantPad1dOp<'a, T: Num> {
    pub forward: ConstantPad1dForward<T>,
    pub backward: ConstantPad1dBackward<T>,
    _marker: core::marker::PhantomData<&'a ()>,
}
