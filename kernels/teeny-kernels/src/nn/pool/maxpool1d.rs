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

use teeny_core::dtype::Num;
use teeny_macros::kernel;
use teeny_triton::triton::{
    types::{AddOffsets, Comparison, Tensor},
    *,
};

/// 1-D max-pooling forward pass.
///
/// Grid: `pid = (b * C + c) * num_ol_tiles + ol_tile`
///
/// Initialises the accumulator to `-inf` and reduces over KL positions.
///
/// **Constraints**: no padding; `OL = (L - KL) / STRIDE + 1`.
#[kernel]
pub fn maxpool1d_forward<
    T: Triton,
    D: Num,
    const KL: i32,
    const STRIDE: i32,
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

    let mut acc = T::cast::<f32, D>(T::full::<f32>(&[BLOCK_OL], -3.4028235e38_f32), None, false);

    let loop_bound = KL;
    for kl in 0..loop_bound {
        let il_range = ol_range * STRIDE + kl;
        let in_offsets = il_range + in_bc_base;
        let tile = T::load(
            input_ptr.add_offsets(in_offsets),
            Some(ol_mask),
            Some(T::cast::<f32, D>(T::full::<f32>(&[BLOCK_OL], -3.4028235e38_f32), None, false)),
            &[],
            None,
            None,
            None,
            false,
        );
        acc = T::maximum(acc, tile);
    }

    let out_offsets = ol_range + out_bc_base;
    T::store(
        output_ptr.add_offsets(out_offsets),
        acc,
        Some(ol_mask),
        &[],
        None,
        None,
    );
}

/// 1-D max-pooling backward pass.
///
/// Re-scans the input window to find the max position, then scatters `dy` to
/// all input elements that equal the stored output maximum. `dx` must be
/// zero-initialised before launch.
#[kernel]
pub fn maxpool1d_backward<
    T: Triton,
    D: Num,
    const KL: i32,
    const STRIDE: i32,
    const BLOCK_OL: i32,
>(
    dy_ptr: T::Pointer<D>,
    x_ptr: T::Pointer<D>,
    y_ptr: T::Pointer<D>,
    dx_ptr: T::Pointer<D>,
    B: i32,
    C: i32,
    L: i32,
    OL: i32,
) where
    T::I32Tensor: Tensor<i32, 1>,
    T::I32Tensor: Comparison<i32, BoolTensor = T::BoolTensor>,
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

    let dy_offsets = ol_range + out_bc_base;
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

    let y_tile = T::load(
        y_ptr.add_offsets(dy_offsets),
        Some(ol_mask),
        Some(T::zeros::<D>(&[BLOCK_OL])),
        &[],
        None,
        None,
        None,
        false,
    );

    let loop_bound = KL;
    for kl in 0..loop_bound {
        let il_range = ol_range * STRIDE + kl;
        let in_offsets = il_range + in_bc_base;
        let x_tile = T::load(
            x_ptr.add_offsets(in_offsets),
            Some(ol_mask),
            Some(T::cast::<f32, D>(T::full::<f32>(&[BLOCK_OL], -3.4028235e38_f32), None, false)),
            &[],
            None,
            None,
            None,
            false,
        );
        let is_max = T::eq(x_tile, y_tile);
        let grad = T::where_(is_max, dy_tile, T::zeros::<D>(&[BLOCK_OL]));
        T::atomic_add(
            dx_ptr.add_offsets(in_offsets),
            grad,
            Some(ol_mask),
            None,
            None,
        );
    }
}

pub struct Maxpool1dOp<'a, T: Num> {
    pub forward: Maxpool1dForward<T>,
    pub backward: Maxpool1dBackward<T>,
    _marker: core::marker::PhantomData<&'a ()>,
}
