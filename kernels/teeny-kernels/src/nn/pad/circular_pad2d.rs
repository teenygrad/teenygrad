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

/// 2-D circular padding forward pass.
///
/// Grid: `pid = ((b*C+c)*OH+oh) * num_ow_tiles + ow_tile`
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
pub fn circular_pad2d_forward<
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

/// 2-D circular padding backward pass.
#[kernel]
pub fn circular_pad2d_backward<
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

pub struct CircularPad2dOp<'a, T: Num> {
    pub forward: CircularPad2dForward<T>,
    pub backward: CircularPad2dBackward<T>,
    _marker: core::marker::PhantomData<&'a ()>,
}
