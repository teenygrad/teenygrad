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

use core::marker::PhantomData;
use teeny_core::dtype::Num;
use teeny_macros::kernel;
use teeny_triton::triton::{
    types::{AddOffsets, Comparison},
    *,
};

/// Channel-wise concatenation forward — NC layout.
///
/// Copies one input chunk into the channel region `[chunk_offset, chunk_offset + chunk_c)`
/// of a wide output NC tensor. Call once per input tensor to build the full concat.
///
/// Index mapping:
///   `y[n * c_total + chunk_offset + ci] = x[n * chunk_c + ci]`
///
/// This is the structural inverse of `channel_chunk_forward`. The backward
/// of this op is `channel_chunk_forward` with the same parameters.
///
/// Grid: `n_spatial * cdiv(chunk_c, BLOCK_SIZE)` CTAs.
#[kernel]
pub fn channel_cat_forward<T: Triton, D: Num, const BLOCK_SIZE: i32>(
    x_ptr: T::Pointer<D>,  // one input: n_spatial * chunk_c  (narrow NC)
    y_ptr: T::Pointer<D>,  // output:    n_spatial * c_total  (wide NC)
    chunk_c: i32,          // input channels for this chunk
    c_total: i32,          // total output channels
    chunk_offset: i32,     // first output channel index: k * chunk_c
) where
    T::I32Tensor: types::Tensor<i32, 1>,
    T::I32Tensor: Comparison<i32, BoolTensor = T::BoolTensor>,
    T::Pointer<D>: AddOffsets<i32, 1, T::I32Tensor, Output = T::Tensor<T::Pointer<D>>>,
{
    let pid = T::program_id(Axis::X);
    let num_c_tiles = T::cdiv(chunk_c, BLOCK_SIZE);
    let pid_n = pid / num_c_tiles;
    let ci_tile = pid % num_c_tiles;
    let ci_start = ci_tile * BLOCK_SIZE;

    let ci_offsets = T::arange(0, BLOCK_SIZE) + ci_start;
    let in_bounds = ci_offsets.lt(chunk_c);

    let in_offsets  = ci_offsets + (pid_n * chunk_c);
    let out_offsets = ci_offsets + (pid_n * c_total + chunk_offset);

    let x = T::load(
        x_ptr.add_offsets(in_offsets),
        Some(in_bounds),
        None,
        &[],
        None,
        None,
        None,
        false,
    );
    T::store(y_ptr.add_offsets(out_offsets), x, Some(in_bounds), &[], None, None);
}

/// Channel-wise cat backward — extracts the gradient slice for one input
/// from the combined upstream gradient tensor.
///
/// Index mapping:
///   `dx[n * chunk_c + ci] = dy[n * c_total + chunk_offset + ci]`
///
/// No atomic operations are required: this kernel reads from a specific
/// disjoint channel range of `dy` and writes to its own output buffer.
///
/// Grid: same as `channel_cat_forward`.
#[kernel]
pub fn channel_cat_backward<T: Triton, D: Num, const BLOCK_SIZE: i32>(
    dy_ptr: T::Pointer<D>,  // upstream grad: n_spatial * c_total  (wide NC)
    dx_ptr: T::Pointer<D>,  // input grad:    n_spatial * chunk_c  (narrow NC)
    chunk_c: i32,
    c_total: i32,
    chunk_offset: i32,
) where
    T::I32Tensor: types::Tensor<i32, 1>,
    T::I32Tensor: Comparison<i32, BoolTensor = T::BoolTensor>,
    T::Pointer<D>: AddOffsets<i32, 1, T::I32Tensor, Output = T::Tensor<T::Pointer<D>>>,
{
    let pid = T::program_id(Axis::X);
    let num_c_tiles = T::cdiv(chunk_c, BLOCK_SIZE);
    let pid_n = pid / num_c_tiles;
    let ci_tile = pid % num_c_tiles;
    let ci_start = ci_tile * BLOCK_SIZE;

    let ci_offsets = T::arange(0, BLOCK_SIZE) + ci_start;
    let in_bounds = ci_offsets.lt(chunk_c);

    let dy_offsets = ci_offsets + (pid_n * c_total + chunk_offset);
    let dx_offsets = ci_offsets + (pid_n * chunk_c);

    let grad = T::load(
        dy_ptr.add_offsets(dy_offsets),
        Some(in_bounds),
        None,
        &[],
        None,
        None,
        None,
        false,
    );
    T::store(dx_ptr.add_offsets(dx_offsets), grad, Some(in_bounds), &[], None, None);
}

pub struct ChannelCatOp<'a, D: Num> {
    pub forward: ChannelCatForward<D>,
    pub backward: ChannelCatBackward<D>,
    _marker: PhantomData<&'a ()>,
}
