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

/// Runtime op for channel_chunk forward + backward.
///
/// Single-launch: extracts one channel slice from a wide NCHW input.
/// `chunk_c` and `chunk_offset` are fixed at graph construction time
/// from `Op::ChannelChunk`.
pub struct ChannelChunkRuntimeOp<D: Num + Send + Sync + 'static> {
    fwd: ChannelChunkForward<D>,
    bwd: ChannelChunkBackward<D>,
    chunk_c: usize,
    chunk_offset: usize,
}

impl<D: Num + Send + Sync + 'static> ChannelChunkRuntimeOp<D> {
    pub fn new(block_size: i32, chunk_c: usize, chunk_offset: usize) -> Self {
        Self {
            fwd: ChannelChunkForward::<D>::new(block_size),
            bwd: ChannelChunkBackward::<D>::new(block_size),
            chunk_c,
            chunk_offset,
        }
    }

    pub fn forward_source(&self) -> &str { &self.fwd.source }
    pub fn backward_source(&self) -> &str { &self.bwd.source }
    pub fn kernel_name(&self) -> &str { self.fwd.name }
}

impl<D: Num + Send + Sync + 'static> teeny_core::model::RuntimeOp for ChannelChunkRuntimeOp<D> {
    fn n_activation_inputs(&self) -> usize { 1 }

    fn param_shapes(&self, _: &[&[usize]], _: &[usize]) -> Vec<Vec<usize>> { Vec::new() }

    fn pack_args(
        &self,
        inputs: &[(teeny_core::model::RawPtr, &[usize])],
        _params: &[teeny_core::model::RawPtr],
        output: teeny_core::model::RawPtr,
        _output_shape: &[usize],
        _output_row_stride: i32,
        visitor: &mut dyn teeny_core::device::program::ArgVisitor,
    ) {
        // Input is NCHW [B, C_total, H, W]. Treat as NC: N=B, C=C_total*H*W.
        let input_shape = inputs[0].1;
        let c_total = (input_shape[1] * input_shape[2] * input_shape[3]) as i32;
        let chunk_c = self.chunk_c as i32;
        // Scale chunk_offset from channel index to NC-layout offset (×H×W).
        let h = input_shape[2];
        let w = input_shape[3];
        let chunk_offset = (self.chunk_offset * h * w) as i32;

        visitor.visit_ptr(inputs[0].0);
        visitor.visit_ptr(output);
        visitor.visit_i32(c_total);
        visitor.visit_i32(chunk_c * (h as i32) * (w as i32));  // chunk_c in NC units
        visitor.visit_i32(chunk_offset);
    }

    fn block(&self) -> [u32; 3] { [self.fwd.block_size as u32, 1, 1] }

    fn grid(&self, output_shape: &[usize]) -> [u32; 3] {
        // Grid over n_spatial * ceil(chunk_c_nc / block_size)
        // output_shape = [B, chunk_c, H, W]
        let n_spatial = output_shape[0];
        let chunk_c_nc = output_shape[1] * output_shape[2] * output_shape[3];
        let num_tiles = chunk_c_nc.div_ceil(self.fwd.block_size as usize);
        [(n_spatial * num_tiles) as u32, 1, 1]
    }

    #[cfg(feature = "training")]
    fn has_backward(&self) -> bool { true }

    #[cfg(feature = "training")]
    fn pack_backward_args(
        &self,
        inputs: &[(teeny_core::model::RawPtr, &[usize])],
        _params: &[teeny_core::model::RawPtr],
        _output: teeny_core::model::RawPtr,
        _output_shape: &[usize],
        grad_output: teeny_core::model::RawPtr,
        _grad_output_row_stride: i32,
        grad_inputs: &[teeny_core::model::RawPtr],
        _grad_params: &[teeny_core::model::RawPtr],
        visitor: &mut dyn teeny_core::device::program::ArgVisitor,
    ) {
        // channel_chunk_backward: (dy_ptr, dx_ptr, c_total, chunk_c, chunk_offset)
        // dy = grad_output (narrow [B, chunk_c, H, W])
        // dx = grad_inputs[0] (wide [B, C_total, H, W])
        let input_shape = inputs[0].1;  // original input shape [B, C_total, H, W]
        let h = input_shape[2];
        let w = input_shape[3];
        let c_total = (input_shape[1] * h * w) as i32;
        let chunk_c_nc = (self.chunk_c * h * w) as i32;
        let chunk_offset = (self.chunk_offset * h * w) as i32;

        visitor.visit_ptr(grad_output);
        visitor.visit_ptr(grad_inputs[0]);
        visitor.visit_i32(c_total);
        visitor.visit_i32(chunk_c_nc);
        visitor.visit_i32(chunk_offset);
    }

    #[cfg(feature = "training")]
    fn backward_grid(&self, input_shapes: &[&[usize]], _output_shape: &[usize]) -> [u32; 3] {
        // Grid over input grad: n_spatial * ceil(chunk_c_nc / block_size)
        let s = input_shapes[0];
        let n_spatial = s[0];
        let chunk_c_nc = self.chunk_c * s[2] * s[3];
        let num_tiles = chunk_c_nc.div_ceil(self.fwd.block_size as usize);
        [(n_spatial * num_tiles) as u32, 1, 1]
    }
}
