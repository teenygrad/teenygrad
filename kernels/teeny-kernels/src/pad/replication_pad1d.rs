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

/// 1-D replication padding forward pass.
///
/// Grid: `pid = (b * C + c) * num_ol_tiles + ol_tile`
///
/// Clamps the input index to `[0, L-1]` (replication of the boundary value).
#[kernel]
pub fn replication_pad1d_forward<
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
    let cond_left = ip_raw.lt(0);
    let cond_right = ip_raw.ge(L);
    let in_bounds = ip_raw.ge(0) & ip_raw.lt(L);

    // `ip_raw * 0` produces an I32Tensor of zeros (index 0 = left boundary).
    // `ip_raw * 0 + (L - 1)` produces an I32Tensor of (L-1) (right boundary).
    let zero_ip = ip_raw * 0;
    let lm1_ip = ip_raw * 0 + (L - 1);

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
        input_ptr.add_offsets(zero_ip + in_bc_base),
        Some(ol_mask & cond_left),
        Some(zeros),
        &[],
        None,
        None,
        None,
        false,
    );
    let val_right = T::load(
        input_ptr.add_offsets(lm1_ip + in_bc_base),
        Some(ol_mask & cond_right),
        Some(zeros),
        &[],
        None,
        None,
        None,
        false,
    );

    let result = T::where_(cond_left, val_left, T::where_(cond_right, val_right, val_center));

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

/// 1-D replication padding backward pass.
///
/// Gradients accumulate to boundary elements for padded lanes.
/// `dx` must be zero-initialised before launch.
#[kernel]
pub fn replication_pad1d_backward<
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
    let cond_left = ip_raw.lt(0);
    let cond_right = ip_raw.ge(L);
    let in_bounds = ip_raw.ge(0) & ip_raw.lt(L);

    let zero_ip = ip_raw * 0;
    let lm1_ip = ip_raw * 0 + (L - 1);

    T::atomic_add(
        dx_ptr.add_offsets(ip_raw + dx_bc_base),
        dy_tile,
        Some(ol_mask & in_bounds),
        None,
        None,
    );
    T::atomic_add(
        dx_ptr.add_offsets(zero_ip + dx_bc_base),
        dy_tile,
        Some(ol_mask & cond_left),
        None,
        None,
    );
    T::atomic_add(
        dx_ptr.add_offsets(lm1_ip + dx_bc_base),
        dy_tile,
        Some(ol_mask & cond_right),
        None,
        None,
    );
}

pub struct ReplicationPad1dOp<'a, T: Num> {
    pub forward: ReplicationPad1dForward<T>,
    pub backward: ReplicationPad1dBackward<T>,
    _marker: core::marker::PhantomData<&'a ()>,
}
