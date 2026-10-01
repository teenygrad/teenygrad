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

/// 2-D constant padding forward pass.
///
/// Grid: `pid = ((b * C + c) * OH + oh) * num_ow_tiles + ow_tile`
///
/// `OH = PT + H + PB`, `OW = PL + W + PR`.
/// Positions outside input region are filled with `value`.
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
pub fn constant_pad2d_forward<
    T: Triton,
    D: Num,
    const PT: i32,
    const PB: i32,
    const PL: i32,
    const PR: i32,
    const BLOCK_OW: i32,
>(
    #[tile(name = "B", extent = _B)]
    #[tile(extent = C)]
    #[tile(
        extent = H,
        window(stride = 1, pad = PT, kernel = 1, output = OH)
    )]
    #[tile(
        block = BLOCK_OW,
        extent = W,
        window(stride = 1, pad = PL, kernel = 1, output = OW)
    )]
    input_ptr: In<T::Pointer<D>>,
    #[tile(name = "B", extent = _B)]
    #[tile(extent = C)]
    #[tile(extent = OH)]
    #[tile(block = BLOCK_OW, extent = OW)]
    output_ptr: Out<T::Pointer<D>>,
    _B: i32,
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
    let value_vec = T::cast::<f32, D>(T::full::<f32>(&[BLOCK_OW], value), None, false);
    let out_offsets = ow_range + ((b * C + c) * OH + oh) * OW;

    if ih < 0 || ih >= H {
        T::store(
            output_ptr.add_offsets(out_offsets),
            value_vec,
            Some(ow_mask),
            &[],
            None,
            None,
        );
        return;
    }

    let iw_range = ow_range - PL;
    let w_in_bounds = iw_range.ge(0) & iw_range.lt(W);
    let combined_mask = ow_mask & w_in_bounds;
    let in_bc_base = (b * C + c) * H * W + ih * W;

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
    dy_ptr: In<T::Pointer<D>>,
    dx_ptr: Out<T::Pointer<D>>,
    _B: i32,
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
    if ih < 0 || ih >= H {
        return;
    }
    let iw_range = ow_range - PL;

    let w_in_bounds = iw_range.ge(0) & iw_range.lt(W);

    let dy_bc_base = ((b * C + c) * OH + oh) * OW;
    let dx_bc_base = (b * C + c) * H * W + ih * W;

    let store_mask = ow_mask & w_in_bounds;

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
