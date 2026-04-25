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

/// 1-D reflection padding forward pass.
///
/// Grid: `pid = (b * C + c) * num_ol_tiles + ol_tile`
///
/// For output position `op`:
/// - `ip = op - PAD_LEFT`
/// - if `ip < 0`: source = input[-ip]
/// - if `ip >= L`: source = input[2*(L-1) - ip]
/// - else: source = input[ip]
///
/// **Constraints**: `PAD_LEFT < L`, `PAD_RIGHT < L`.
#[kernel]
pub fn reflection_pad1d_forward<
    T: Triton,
    D: Float,
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

    let ip_raw = ol_range - PAD_LEFT;
    let left_cond = ip_raw.lt(0);
    let right_cond = ip_raw.ge(L);

    // Reflected index for each case (safe to compute for all lanes)
    let ip_left = ip_raw * (-1);          // -ip: always in [1, PAD_LEFT] for left pad
    let ip_right = ip_raw * (-1) + (2 * (L - 1)); // 2*(L-1) - ip for right pad

    // Load all three candidate positions. Masks keep each load safe:
    // - left load: mask = left_cond & ol_mask (only left-pad lanes)
    // - right load: mask = right_cond & ol_mask
    // - center load: mask = in_bounds & ol_mask
    let in_bounds = ip_raw.ge(0) & ip_raw.lt(L);

    let zeros = T::zeros::<D>(&[BLOCK_OL]);
    let val_center = T::load(
        input_ptr.add_offsets(ip_raw + in_bc_base),
        Some(ol_mask & in_bounds),
        Some(zeros),
        &[],
        None,
        None,
        None,
        false,
    );
    let val_left = T::load(
        input_ptr.add_offsets(ip_left + in_bc_base),
        Some(ol_mask & left_cond),
        Some(zeros),
        &[],
        None,
        None,
        None,
        false,
    );
    let val_right = T::load(
        input_ptr.add_offsets(ip_right + in_bc_base),
        Some(ol_mask & right_cond),
        Some(zeros),
        &[],
        None,
        None,
        None,
        false,
    );

    let result = T::where_(left_cond, val_left, T::where_(right_cond, val_right, val_center));

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

/// 1-D reflection padding backward pass.
///
/// Each output gradient position maps back to one input position via the same
/// reflection rule. Multiple output positions may map to the same input
/// position (the boundary elements reflect), so `atomic_add` is used.
/// `dx` must be zero-initialised before launch.
#[kernel]
pub fn reflection_pad1d_backward<
    T: Triton,
    D: Float,
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

    let ip_raw = ol_range - PAD_LEFT;
    let left_cond = ip_raw.lt(0);
    let right_cond = ip_raw.ge(L);
    let in_bounds = ip_raw.ge(0) & ip_raw.lt(L);

    let ip_left = ip_raw * (-1);
    let ip_right = ip_raw * (-1) + (2 * (L - 1));

    // Scatter to center (input) positions
    T::atomic_add(
        dx_ptr.add_offsets(ip_raw + dx_bc_base),
        dy_tile,
        Some(ol_mask & in_bounds),
        None,
        None,
    );
    // Scatter to reflected left positions
    T::atomic_add(
        dx_ptr.add_offsets(ip_left + dx_bc_base),
        dy_tile,
        Some(ol_mask & left_cond),
        None,
        None,
    );
    // Scatter to reflected right positions
    T::atomic_add(
        dx_ptr.add_offsets(ip_right + dx_bc_base),
        dy_tile,
        Some(ol_mask & right_cond),
        None,
        None,
    );
}
