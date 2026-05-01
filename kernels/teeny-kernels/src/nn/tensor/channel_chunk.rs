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

/// Channel-wise chunk (split) forward — NC layout.
///
/// Extracts one contiguous channel slice `[chunk_offset, chunk_offset + chunk_c)`
/// from a wide NC tensor and writes it into a narrow NC output tensor.
///
/// Index mapping (NC layout, index = `n * C + c`):
///   `y[n * chunk_c + ci] = x[n * c_total + chunk_offset + ci]`
///
/// This is the structural inverse of `channel_cat_forward`. The backward
/// of this op is `channel_cat_forward` with the same parameters.
///
/// Grid: `n_spatial * cdiv(chunk_c, BLOCK_SIZE)` CTAs.
/// `pid` is decoded into `(pid_n, ci_tile)` via scalar integer division —
/// no tensor-level division or modulo is required.
#[kernel]
pub fn channel_chunk_forward<T: Triton, D: Num, const BLOCK_SIZE: i32>(
    x_ptr: T::Pointer<D>,  // input:  n_spatial * c_total  (wide NC)
    y_ptr: T::Pointer<D>,  // output: n_spatial * chunk_c  (narrow NC)
    c_total: i32,          // total input channels
    chunk_c: i32,          // output channels per chunk
    chunk_offset: i32,     // first input channel index: k * chunk_c
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

    let in_offsets  = ci_offsets + (pid_n * c_total + chunk_offset);
    let out_offsets = ci_offsets + (pid_n * chunk_c);

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

/// Channel-wise chunk backward — propagates the gradient of one chunk
/// output back into the full-width input gradient tensor.
///
/// Index mapping:
///   `dx[n * c_total + chunk_offset + ci] = dy[n * chunk_c + ci]`
///
/// No atomic operations are required: each chunk's backward writes to a
/// disjoint channel range `[chunk_offset, chunk_offset + chunk_c)` of `dx`.
///
/// Grid: same as `channel_chunk_forward`.
#[kernel]
pub fn channel_chunk_backward<T: Triton, D: Num, const BLOCK_SIZE: i32>(
    dy_ptr: T::Pointer<D>,  // upstream grad: n_spatial * chunk_c  (narrow NC)
    dx_ptr: T::Pointer<D>,  // input grad:    n_spatial * c_total  (wide NC)
    c_total: i32,
    chunk_c: i32,
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

    let dy_offsets = ci_offsets + (pid_n * chunk_c);
    let dx_offsets = ci_offsets + (pid_n * c_total + chunk_offset);

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

pub struct ChannelChunkOp<'a, D: Num> {
    pub forward: ChannelChunkForward<D>,
    pub backward: ChannelChunkBackward<D>,
    _marker: PhantomData<&'a ()>,
}
