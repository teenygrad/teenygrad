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

/// Combined runtime op for channel-cat forward + backward.
///
/// Forward: launches one `channel_cat_forward` call per input chunk, all
/// writing to disjoint channel ranges of the shared output buffer.
///
/// Backward: launches one `channel_cat_backward` call per input chunk,
/// each extracting the gradient slice for its chunk from the full concat
/// upstream gradient.
pub struct ChannelCatRuntimeOp<D: Num + Send + Sync + 'static> {
    fwd: ChannelCatForward<D>,
    bwd: ChannelCatBackward<D>,
    n_inputs: usize,
}

impl<D: Num + Send + Sync + 'static> ChannelCatRuntimeOp<D> {
    pub fn new(block_size: i32, n_inputs: usize) -> Self {
        Self {
            fwd: ChannelCatForward::<D>::new(block_size),
            bwd: ChannelCatBackward::<D>::new(block_size),
            n_inputs,
        }
    }

    /// Returns the compiled forward kernel source for embedding in KernelExecutable.
    pub fn forward_source(&self) -> &str { &self.fwd.source }
    /// Returns the compiled backward kernel source.
    pub fn backward_source(&self) -> &str { &self.bwd.source }
    /// Returns the kernel name (for the forward kernel).
    pub fn kernel_name(&self) -> &str { self.fwd.name }
}

impl<D: Num + Send + Sync + 'static> teeny_core::model::RuntimeOp for ChannelCatRuntimeOp<D> {
    fn n_activation_inputs(&self) -> usize { self.n_inputs }

    fn param_shapes(&self, _: &[&[usize]], _: &[usize]) -> Vec<Vec<usize>> {
        Vec::new()
    }

    // pack_args is used only as a fallback (n_launches==1 never occurs for ChannelCat).
    fn pack_args(
        &self,
        inputs: &[(teeny_core::model::RawPtr, &[usize])],
        params: &[teeny_core::model::RawPtr],
        output: teeny_core::model::RawPtr,
        output_shape: &[usize],
        output_row_stride: i32,
        visitor: &mut dyn teeny_core::device::program::ArgVisitor,
    ) {
        self.pack_args_for_launch(0, inputs, params, output, output_shape, output_row_stride, visitor);
    }

    fn n_launches(&self) -> usize { self.n_inputs }

    fn pack_args_for_launch(
        &self,
        launch_idx: usize,
        inputs: &[(teeny_core::model::RawPtr, &[usize])],
        _params: &[teeny_core::model::RawPtr],
        output: teeny_core::model::RawPtr,
        _output_shape: &[usize],
        _output_row_stride: i32,
        visitor: &mut dyn teeny_core::device::program::ArgVisitor,
    ) {
        // Each input is NCHW [B, C_i, H, W]. Treat as NC where N=B, C=C_i*H*W.
        let chunk_offset: i32 = inputs[..launch_idx].iter()
            .map(|(_, s)| (s[1] * s[2] * s[3]) as i32)
            .sum();
        let x_ptr = inputs[launch_idx].0;
        let input_shape = inputs[launch_idx].1;
        let chunk_c = (input_shape[1] * input_shape[2] * input_shape[3]) as i32;
        let c_total: i32 = inputs.iter().map(|(_, s)| (s[1] * s[2] * s[3]) as i32).sum();

        visitor.visit_ptr(x_ptr);
        visitor.visit_ptr(output);
        visitor.visit_i32(chunk_c);
        visitor.visit_i32(c_total);
        visitor.visit_i32(chunk_offset);
    }

    fn grid_for_launch(
        &self,
        launch_idx: usize,
        input_shapes: &[&[usize]],
        _output_shape: &[usize],
    ) -> [u32; 3] {
        let s = input_shapes[launch_idx];
        let n_spatial = s[0];
        let chunk_c = s[1] * s[2] * s[3];
        let num_tiles = chunk_c.div_ceil(self.fwd.block_size as usize);
        [(n_spatial * num_tiles) as u32, 1, 1]
    }

    fn block(&self) -> [u32; 3] { [self.fwd.block_size as u32, 1, 1] }

    fn grid(&self, output_shape: &[usize]) -> [u32; 3] {
        let n_spatial = output_shape[0];
        let c = output_shape[1] * output_shape[2] * output_shape[3];
        let num_tiles = c.div_ceil(self.fwd.block_size as usize);
        [(n_spatial * num_tiles) as u32, 1, 1]
    }

    #[cfg(feature = "training")]
    fn has_backward(&self) -> bool { true }

    #[cfg(feature = "training")]
    fn n_backward_launches(&self) -> usize { self.n_inputs }

    #[cfg(feature = "training")]
    fn pack_backward_args(
        &self,
        inputs: &[(teeny_core::model::RawPtr, &[usize])],
        params: &[teeny_core::model::RawPtr],
        output: teeny_core::model::RawPtr,
        output_shape: &[usize],
        grad_output: teeny_core::model::RawPtr,
        grad_output_row_stride: i32,
        grad_inputs: &[teeny_core::model::RawPtr],
        grad_params: &[teeny_core::model::RawPtr],
        visitor: &mut dyn teeny_core::device::program::ArgVisitor,
    ) {
        self.pack_backward_args_for_launch(
            0, inputs, params, output, output_shape,
            grad_output, grad_output_row_stride, grad_inputs, grad_params, visitor,
        );
    }

    #[cfg(feature = "training")]
    #[allow(clippy::too_many_arguments)]
    fn pack_backward_args_for_launch(
        &self,
        launch_idx: usize,
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
        let chunk_offset: i32 = inputs[..launch_idx].iter()
            .map(|(_, s)| (s[1] * s[2] * s[3]) as i32)
            .sum();
        let input_shape = inputs[launch_idx].1;
        let chunk_c = (input_shape[1] * input_shape[2] * input_shape[3]) as i32;
        let c_total: i32 = inputs.iter().map(|(_, s)| (s[1] * s[2] * s[3]) as i32).sum();

        visitor.visit_ptr(grad_output);
        visitor.visit_ptr(grad_inputs[launch_idx]);
        visitor.visit_i32(chunk_c);
        visitor.visit_i32(c_total);
        visitor.visit_i32(chunk_offset);
    }

    #[cfg(feature = "training")]
    fn backward_grid(&self, input_shapes: &[&[usize]], output_shape: &[usize]) -> [u32; 3] {
        self.backward_grid_for_launch(0, input_shapes, output_shape)
    }

    #[cfg(feature = "training")]
    fn backward_grid_for_launch(
        &self,
        launch_idx: usize,
        input_shapes: &[&[usize]],
        output_shape: &[usize],
    ) -> [u32; 3] {
        self.grid_for_launch(launch_idx, input_shapes, output_shape)
    }
}
