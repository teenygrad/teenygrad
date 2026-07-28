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

//! Channel bias add Triton kernels.
//!
//! Layout: input `x` and output `y` are NC-layout, where N = B*H*W and C =
//! number of channels. Bias is a (C,) vector. Element `x[n, c]` lives at flat
//! offset `n * C + c`.
//!
//! Parallelism: **one CTA per channel**. Each CTA iterates over all N spatial
//! elements in `BLOCK_N`-wide tiles, adding the per-channel scalar bias.

#![allow(non_snake_case)]

use teeny_core::dtype::Float;
use teeny_macros::kernel;
use teeny_triton::triton::{
    types::{AddOffsets, Comparison},
    *,
};

// ─── Forward ─────────────────────────────────────────────────────────────────

/// Adds a (C,) bias to a tensor in NC layout (N = B*H*W, C = channels).
///
/// Grid: `[C]` — one CTA per channel.
#[kernel]
pub fn channel_bias_add_forward<T: Triton, D: Float, const BLOCK_N: i32>(
    x_ptr: T::Pointer<D>,
    bias_ptr: T::Pointer<D>,
    y_ptr: T::Pointer<D>,
    N: i32,
    C: i32,
) where
    T::I32Tensor: types::Tensor<i32, 1>,
    T::I32Tensor: Comparison<i32, BoolTensor = T::BoolTensor>,
    T::Pointer<D>: AddOffsets<i32, 1, T::I32Tensor, Output = T::Tensor<T::Pointer<D>>>,
{
    let c = T::program_id(Axis::X);
    let c_idx = T::arange(0, 1) + c;

    // Load bias[c] as shape [1], broadcast to [BLOCK_N].
    let bias = T::broadcast_to(
        T::load(
            bias_ptr.add_offsets(c_idx),
            None,
            None,
            &[],
            None,
            None,
            None,
            false,
        ),
        &[BLOCK_N],
    );

    let zeros = T::zeros::<D>(&[BLOCK_N]);
    let mut n_start: i32 = 0;
    while n_start < N {
        let offsets_n = T::arange(0, BLOCK_N) + n_start;
        let mask = offsets_n.lt(N);
        let elem_offsets = offsets_n * C + c;

        let x_tile = T::load(
            x_ptr.add_offsets(elem_offsets),
            Some(mask),
            Some(zeros),
            &[],
            None,
            None,
            None,
            false,
        );
        T::store(
            y_ptr.add_offsets(elem_offsets),
            x_tile + bias,
            Some(mask),
            &[],
            None,
            None,
        );

        n_start += BLOCK_N;
    }
}

// ─── Backward ────────────────────────────────────────────────────────────────

/// Backward pass for channel bias add.
///
/// dx = dy (identity), dbias[c] = sum over N of dy[n, c].
///
/// Grid: `[C]` — one CTA per channel.
#[kernel]
pub fn channel_bias_add_backward<T: Triton, D: Float, const BLOCK_N: i32>(
    dy_ptr: T::Pointer<D>,
    dx_ptr: T::Pointer<D>,
    dbias_ptr: T::Pointer<D>,
    N: i32,
    C: i32,
) where
    T::I32Tensor: types::Tensor<i32, 1>,
    T::I32Tensor: Comparison<i32, BoolTensor = T::BoolTensor>,
    T::Pointer<D>: AddOffsets<i32, 1, T::I32Tensor, Output = T::Tensor<T::Pointer<D>>>,
{
    let c = T::program_id(Axis::X);
    let c_idx = T::arange(0, 1) + c;

    let zeros = T::zeros::<D>(&[BLOCK_N]);
    let mut acc = T::zeros::<D>(&[1]);
    let mut n_start: i32 = 0;

    while n_start < N {
        let offsets_n = T::arange(0, BLOCK_N) + n_start;
        let mask = offsets_n.lt(N);
        let elem_offsets = offsets_n * C + c;

        let dy_tile = T::load(
            dy_ptr.add_offsets(elem_offsets),
            Some(mask),
            Some(zeros),
            &[],
            None,
            None,
            None,
            false,
        );

        // dx = dy (pass-through gradient)
        T::store(
            dx_ptr.add_offsets(elem_offsets),
            dy_tile,
            Some(mask),
            &[],
            None,
            None,
        );

        // Accumulate bias gradient (keepdim so shape stays [1])
        acc = acc + T::sum(dy_tile, None, true);

        n_start += BLOCK_N;
    }

    // Write accumulated bias gradient with atomic add.
    T::atomic_add(dbias_ptr.add_offsets(c_idx), acc, None, None, None);
}

// ─── NCHW Bias Add (for Conv2d with bias) ────────────────────────────────────

/// Adds a (C,) bias to a tensor in NCHW layout.
///
/// Grid: `[C, B]` — one CTA per (channel, batch) pair; each CTA iterates over
/// H*W spatial positions in `BLOCK_HW`-wide tiles.
#[kernel]
pub fn nchw_bias_add_forward<T: Triton, D: Float, const BLOCK_HW: i32>(
    x_ptr: T::Pointer<D>,
    bias_ptr: T::Pointer<D>,
    y_ptr: T::Pointer<D>,
    C: i32,
    HW: i32,
) where
    T::I32Tensor: types::Tensor<i32, 1>,
    T::I32Tensor: Comparison<i32, BoolTensor = T::BoolTensor>,
    T::Pointer<D>: AddOffsets<i32, 1, T::I32Tensor, Output = T::Tensor<T::Pointer<D>>>,
{
    let c = T::program_id(Axis::X);
    let b = T::program_id(Axis::Y);
    let c_idx = T::arange(0, 1) + c;

    let bias = T::broadcast_to(
        T::load(
            bias_ptr.add_offsets(c_idx),
            None,
            None,
            &[],
            None,
            None,
            None,
            false,
        ),
        &[BLOCK_HW],
    );

    let zeros = T::zeros::<D>(&[BLOCK_HW]);
    let batch_channel_offset: i32 = b * C * HW + c * HW;
    let mut hw_start: i32 = 0;
    while hw_start < HW {
        let offsets = T::arange(0, BLOCK_HW) + hw_start;
        let mask = offsets.lt(HW);
        let elem_offsets = offsets + batch_channel_offset;
        let x_tile = T::load(
            x_ptr.add_offsets(elem_offsets),
            Some(mask),
            Some(zeros),
            &[],
            None,
            None,
            None,
            false,
        );
        T::store(
            y_ptr.add_offsets(elem_offsets),
            x_tile + bias,
            Some(mask),
            &[],
            None,
            None,
        );
        hw_start += BLOCK_HW;
    }
}

/// NCHW bias add backward: dx = dy, dbias[c] = sum over (B, H, W) of dy.
///
/// Grid: `[C, B]` — one CTA per (channel, batch) pair; single while loop over
/// H*W to avoid nested loops (which ICE the teenyc compiler).
#[kernel]
pub fn nchw_bias_add_backward<T: Triton, D: Float, const BLOCK_HW: i32>(
    dy_ptr: T::Pointer<D>,
    dx_ptr: T::Pointer<D>,
    dbias_ptr: T::Pointer<D>,
    C: i32,
    HW: i32,
) where
    T::I32Tensor: types::Tensor<i32, 1>,
    T::I32Tensor: Comparison<i32, BoolTensor = T::BoolTensor>,
    T::Pointer<D>: AddOffsets<i32, 1, T::I32Tensor, Output = T::Tensor<T::Pointer<D>>>,
{
    let c = T::program_id(Axis::X);
    let b = T::program_id(Axis::Y);
    let c_idx = T::arange(0, 1) + c;
    let zeros = T::zeros::<D>(&[BLOCK_HW]);
    let mut dbias_acc = T::zeros::<D>(&[1]);
    let batch_channel_offset: i32 = b * C * HW + c * HW;

    let mut hw_start: i32 = 0;
    while hw_start < HW {
        let offsets = T::arange(0, BLOCK_HW) + hw_start;
        let mask = offsets.lt(HW);
        let elem_offsets = offsets + batch_channel_offset;
        let dy_tile = T::load(
            dy_ptr.add_offsets(elem_offsets),
            Some(mask),
            Some(zeros),
            &[],
            None,
            None,
            None,
            false,
        );
        T::store(
            dx_ptr.add_offsets(elem_offsets),
            dy_tile,
            Some(mask),
            &[],
            None,
            None,
        );
        dbias_acc = dbias_acc + T::sum(dy_tile, None, true);
        hw_start += BLOCK_HW;
    }
    T::atomic_add(dbias_ptr.add_offsets(c_idx), dbias_acc, None, None, None);
}

/// RuntimeOp for adding a (C,) bias to an NCHW-layout tensor.
///
/// Used by the graph lowering when `Op::Conv2d { has_bias: true }` is encountered.
/// Grid: `[C, B]` — one CTA per (channel, batch-item).
pub struct NchwBiasAddRuntimeOp<D: Float + Send + Sync + 'static> {
    fwd: NchwBiasAddForward<D>,
    bwd: NchwBiasAddBackward<D>,
    block_hw: i32,
}

impl<D: Float + Send + Sync + 'static> NchwBiasAddRuntimeOp<D> {
    pub fn new(block_hw: i32) -> Self {
        Self {
            fwd: NchwBiasAddForward::<D>::new(block_hw),
            bwd: NchwBiasAddBackward::<D>::new(block_hw),
            block_hw,
        }
    }
    pub fn forward_source(&self) -> &str {
        &self.fwd.source
    }
    pub fn backward_source(&self) -> &str {
        &self.bwd.source
    }
    pub fn kernel_name(&self) -> &str {
        self.fwd.name
    }
}

impl<D: Float + Send + Sync + 'static> teeny_core::model::RuntimeOp for NchwBiasAddRuntimeOp<D> {
    fn n_activation_inputs(&self) -> usize {
        1
    }

    fn param_shapes(&self, input_shapes: &[&[usize]], _output_shape: &[usize]) -> Vec<Vec<usize>> {
        let c = input_shapes[0][1];
        vec![vec![c]]
    }

    fn param_names(&self) -> &'static [&'static str] {
        &["bias"]
    }

    fn pack_args(
        &self,
        inputs: &[(teeny_core::model::RawPtr, &[usize])],
        params: &[teeny_core::model::RawPtr],
        output: teeny_core::model::RawPtr,
        output_shape: &[usize],
        _output_row_stride: i32,
        visitor: &mut dyn teeny_core::device::program::ArgVisitor,
    ) {
        let c = output_shape[1] as i32;
        let hw = (output_shape[2] * output_shape[3]) as i32;
        visitor.visit_ptr(inputs[0].0); // x_ptr
        visitor.visit_ptr(params[0]); // bias_ptr
        visitor.visit_ptr(output); // y_ptr
        visitor.visit_i32(c);
        visitor.visit_i32(hw);
    }

    fn block(&self) -> [u32; 3] {
        [self.block_hw as u32, 1, 1]
    }

    fn grid(&self, output_shape: &[usize]) -> [u32; 3] {
        [output_shape[1] as u32, output_shape[0] as u32, 1]
    }

    #[cfg(feature = "training")]
    fn has_backward(&self) -> bool {
        true
    }

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
        grad_params: &[teeny_core::model::RawPtr],
        visitor: &mut dyn teeny_core::device::program::ArgVisitor,
    ) {
        let in_shape = inputs[0].1; // [B, C, H, W]
        let c = in_shape[1] as i32;
        let hw = (in_shape[2] * in_shape[3]) as i32;
        visitor.visit_ptr(grad_output); // dy_ptr
        visitor.visit_ptr(grad_inputs[0]); // dx_ptr
        visitor.visit_ptr(grad_params[0]); // dbias_ptr
        // B is encoded in grid.y — not passed as a kernel arg.
        visitor.visit_i32(c);
        visitor.visit_i32(hw);
    }

    #[cfg(feature = "training")]
    fn backward_block(&self) -> [u32; 3] {
        [self.block_hw as u32, 1, 1]
    }

    #[cfg(feature = "training")]
    fn backward_grid(&self, input_shapes: &[&[usize]], _output_shape: &[usize]) -> [u32; 3] {
        // Grid [C, B] — one CTA per (channel, batch) pair, mirroring the kernel.
        [input_shapes[0][1] as u32, input_shapes[0][0] as u32, 1]
    }
}

// ─── RuntimeOp ───────────────────────────────────────────────────────────────

/// Combined forward + backward RuntimeOp for channel bias add.
///
/// Forward kernel arguments: `x_ptr`, `bias_ptr`, `y_ptr`, `N_SPATIAL`, `C`.
/// Backward kernel arguments: `dy_ptr`, `dx_ptr`, `dbias_ptr`, `N_SPATIAL`, `C`.
///
/// Grid: `[C, 1, 1]` for both forward and backward.
pub struct ChannelBiasAddRuntimeOp<D: Float + Send + Sync + 'static> {
    fwd: ChannelBiasAddForward<D>,
    bwd: ChannelBiasAddBackward<D>,
    /// Output channel count, fixed at construction time.
    c_out: usize,
}

impl<D: Float + Send + Sync + 'static> ChannelBiasAddRuntimeOp<D> {
    pub fn new(block_n: i32, c_out: usize) -> Self {
        Self {
            fwd: ChannelBiasAddForward::<D>::new(block_n),
            bwd: ChannelBiasAddBackward::<D>::new(block_n),
            c_out,
        }
    }

    pub fn forward_source(&self) -> &str {
        &self.fwd.source
    }
    pub fn backward_source(&self) -> &str {
        &self.bwd.source
    }
    pub fn kernel_name(&self) -> &str {
        self.fwd.name
    }
}

impl<D: Float + Send + Sync + 'static> teeny_core::model::RuntimeOp for ChannelBiasAddRuntimeOp<D> {
    fn n_activation_inputs(&self) -> usize {
        1
    }

    fn param_shapes(&self, _input_shapes: &[&[usize]], _output_shape: &[usize]) -> Vec<Vec<usize>> {
        // Bias shape is (C_out,)
        vec![vec![self.c_out]]
    }

    fn param_names(&self) -> &'static [&'static str] {
        &["bias"]
    }

    fn pack_args(
        &self,
        inputs: &[(teeny_core::model::RawPtr, &[usize])],
        params: &[teeny_core::model::RawPtr],
        output: teeny_core::model::RawPtr,
        output_shape: &[usize],
        _output_row_stride: i32,
        visitor: &mut dyn teeny_core::device::program::ArgVisitor,
    ) {
        // inputs[0].1 = [B, C, H, W]
        let input_shape = inputs[0].1;
        let b = input_shape[0];
        let c = output_shape[1];
        let h = input_shape[2];
        let w = input_shape[3];
        let n_spatial = (b * h * w) as i32;

        visitor.visit_ptr(inputs[0].0); // x_ptr
        visitor.visit_ptr(params[0]); // bias_ptr
        visitor.visit_ptr(output); // y_ptr
        visitor.visit_i32(n_spatial); // N
        visitor.visit_i32(c as i32); // C
    }

    fn block(&self) -> [u32; 3] {
        [128, 1, 1]
    }

    fn grid(&self, output_shape: &[usize]) -> [u32; 3] {
        // One CTA per output channel.
        [output_shape[1] as u32, 1, 1]
    }

    #[cfg(feature = "training")]
    fn has_backward(&self) -> bool {
        true
    }

    #[cfg(feature = "training")]
    fn pack_backward_args(
        &self,
        inputs: &[(teeny_core::model::RawPtr, &[usize])],
        _params: &[teeny_core::model::RawPtr],
        _output: teeny_core::model::RawPtr,
        output_shape: &[usize],
        grad_output: teeny_core::model::RawPtr,
        _grad_output_row_stride: i32,
        grad_inputs: &[teeny_core::model::RawPtr],
        grad_params: &[teeny_core::model::RawPtr],
        visitor: &mut dyn teeny_core::device::program::ArgVisitor,
    ) {
        let input_shape = inputs[0].1;
        let b = input_shape[0];
        let c = output_shape[1];
        let h = input_shape[2];
        let w = input_shape[3];
        let n_spatial = (b * h * w) as i32;

        visitor.visit_ptr(grad_output); // dy_ptr
        visitor.visit_ptr(grad_inputs[0]); // dx_ptr
        visitor.visit_ptr(grad_params[0]); // dbias_ptr
        visitor.visit_i32(n_spatial); // N
        visitor.visit_i32(c as i32); // C
    }

    #[cfg(feature = "training")]
    fn backward_grid(&self, _input_shapes: &[&[usize]], output_shape: &[usize]) -> [u32; 3] {
        [output_shape[1] as u32, 1, 1]
    }
}
