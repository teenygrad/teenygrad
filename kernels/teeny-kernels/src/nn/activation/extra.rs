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

//! Additional activation kernels: Swish, PRelu, LogSoftmax, Hardmax,
//! ThresholdedRelu, Shrink.

#![allow(non_snake_case)]

use teeny_macros::kernel;
use teeny_triton::triton::{
    types::{AddOffsets, Comparison},
    *,
};

// ── Swish (= SiLU: x * sigmoid(x)) ───────────────────────────────────────────

/// Forward: y = x * sigmoid(x) = x / (1 + exp(-x))
#[kernel]
pub fn swish_forward<T: Triton, const BLOCK_SIZE: i32>(
    x_ptr: T::Pointer<f32>,
    y_ptr: T::Pointer<f32>,
    n_elements: i32,
) where
    T::I32Tensor: types::Tensor<i32, 1>,
    T::I32Tensor: Comparison<i32, BoolTensor = T::BoolTensor>,
    T::Pointer<f32>: AddOffsets<i32, 1, T::I32Tensor, Output = T::Tensor<T::Pointer<f32>>>,
{
    let pid = T::program_id(Axis::X);
    let block_start = pid * BLOCK_SIZE;
    let offsets = T::arange(0, BLOCK_SIZE) + block_start;
    let in_bounds = offsets.lt(n_elements);
    let x = T::load(
        x_ptr.add_offsets(offsets),
        Some(in_bounds),
        None,
        &[],
        None,
        None,
        None,
        false,
    );
    let one = T::full::<f32>(&[BLOCK_SIZE], 1.0_f32);
    let neg1 = T::full::<f32>(&[BLOCK_SIZE], -1.0_f32);
    let sig = one / (one + T::exp(neg1 * x));
    let y = x * sig;
    T::store(
        y_ptr.add_offsets(offsets),
        y,
        Some(in_bounds),
        &[],
        None,
        None,
    );
}

/// Backward: dx = (sigmoid(x) + x * sigmoid(x) * (1 - sigmoid(x))) * dy
///             = (sig + x * sig * (1 - sig)) * dy
#[kernel]
pub fn swish_backward<T: Triton, const BLOCK_SIZE: i32>(
    dy_ptr: T::Pointer<f32>,
    x_ptr: T::Pointer<f32>,
    dx_ptr: T::Pointer<f32>,
    n_elements: i32,
) where
    T::I32Tensor: types::Tensor<i32, 1>,
    T::I32Tensor: Comparison<i32, BoolTensor = T::BoolTensor>,
    T::Pointer<f32>: AddOffsets<i32, 1, T::I32Tensor, Output = T::Tensor<T::Pointer<f32>>>,
{
    let pid = T::program_id(Axis::X);
    let block_start = pid * BLOCK_SIZE;
    let offsets = T::arange(0, BLOCK_SIZE) + block_start;
    let in_bounds = offsets.lt(n_elements);
    let dy = T::load(
        dy_ptr.add_offsets(offsets),
        Some(in_bounds),
        None,
        &[],
        None,
        None,
        None,
        false,
    );
    let x = T::load(
        x_ptr.add_offsets(offsets),
        Some(in_bounds),
        None,
        &[],
        None,
        None,
        None,
        false,
    );
    let one = T::full::<f32>(&[BLOCK_SIZE], 1.0_f32);
    let neg1 = T::full::<f32>(&[BLOCK_SIZE], -1.0_f32);
    let sig = one / (one + T::exp(neg1 * x));
    let dx = (sig + x * sig * (one - sig)) * dy;
    T::store(
        dx_ptr.add_offsets(offsets),
        dx,
        Some(in_bounds),
        &[],
        None,
        None,
    );
}

impl teeny_core::model::RuntimeOp for SwishForward {
    fn n_activation_inputs(&self) -> usize {
        1
    }
    fn param_shapes(&self, _: &[&[usize]], _: &[usize]) -> Vec<Vec<usize>> {
        vec![]
    }
    fn pack_args(
        &self,
        inputs: &[(teeny_core::model::RawPtr, &[usize])],
        _: &[teeny_core::model::RawPtr],
        output: teeny_core::model::RawPtr,
        output_shape: &[usize],
        _: i32,
        visitor: &mut dyn teeny_core::device::program::ArgVisitor,
    ) {
        let n: usize = output_shape.iter().product();
        visitor.visit_ptr(inputs[0].0);
        visitor.visit_ptr(output);
        visitor.visit_i32(n as i32);
    }
    fn block(&self) -> [u32; 3] {
        [self.block_size as u32, 1, 1]
    }
    fn grid(&self, output_shape: &[usize]) -> [u32; 3] {
        let n: usize = output_shape.iter().product();
        [n.div_ceil(self.block_size as usize) as u32, 1, 1]
    }
    #[cfg(feature = "training")]
    fn has_backward(&self) -> bool {
        true
    }
    #[cfg(feature = "training")]
    fn pack_backward_args(
        &self,
        inputs: &[(teeny_core::model::RawPtr, &[usize])],
        _: &[teeny_core::model::RawPtr],
        _: teeny_core::model::RawPtr,
        output_shape: &[usize],
        grad_output: teeny_core::model::RawPtr,
        _: i32,
        grad_inputs: &[teeny_core::model::RawPtr],
        _: &[teeny_core::model::RawPtr],
        visitor: &mut dyn teeny_core::device::program::ArgVisitor,
    ) {
        let n: usize = output_shape.iter().product();
        visitor.visit_ptr(grad_output);
        visitor.visit_ptr(inputs[0].0);
        visitor.visit_ptr(grad_inputs[0]);
        visitor.visit_i32(n as i32);
    }
    #[cfg(feature = "training")]
    fn backward_block(&self) -> [u32; 3] {
        [self.block_size as u32, 1, 1]
    }
    #[cfg(feature = "training")]
    fn backward_grid(&self, _: &[&[usize]], output_shape: &[usize]) -> [u32; 3] {
        let n: usize = output_shape.iter().product();
        [n.div_ceil(self.block_size as usize) as u32, 1, 1]
    }
}

// ── PRelu (2-input: x, slope) ─────────────────────────────────────────────────

/// Forward: y = max(0, x) + slope * min(0, x)
/// The slope tensor has the same shape as x (or broadcastable; kernel assumes same shape here).
#[kernel]
pub fn prelu_forward<T: Triton, const BLOCK_SIZE: i32>(
    x_ptr: T::Pointer<f32>,
    slope_ptr: T::Pointer<f32>,
    y_ptr: T::Pointer<f32>,
    n_elements: i32,
) where
    T::I32Tensor: types::Tensor<i32, 1>,
    T::I32Tensor: Comparison<i32, BoolTensor = T::BoolTensor>,
    T::Pointer<f32>: AddOffsets<i32, 1, T::I32Tensor, Output = T::Tensor<T::Pointer<f32>>>,
{
    let pid = T::program_id(Axis::X);
    let block_start = pid * BLOCK_SIZE;
    let offsets = T::arange(0, BLOCK_SIZE) + block_start;
    let in_bounds = offsets.lt(n_elements);
    let x = T::load(
        x_ptr.add_offsets(offsets),
        Some(in_bounds),
        None,
        &[],
        None,
        None,
        None,
        false,
    );
    let slope = T::load(
        slope_ptr.add_offsets(offsets),
        Some(in_bounds),
        None,
        &[],
        None,
        None,
        None,
        false,
    );
    let zero = T::zeros_like(x);
    let pos = T::maximum(x, zero);
    let neg = slope * T::minimum(x, zero);
    let y = pos + neg;
    T::store(
        y_ptr.add_offsets(offsets),
        y,
        Some(in_bounds),
        &[],
        None,
        None,
    );
}

/// Backward: dx = dy if x >= 0 else slope * dy;
///           dslope = dy * min(x, 0) = dy * x if x < 0 else 0
#[kernel]
pub fn prelu_backward<T: Triton, const BLOCK_SIZE: i32>(
    dy_ptr: T::Pointer<f32>,
    x_ptr: T::Pointer<f32>,
    slope_ptr: T::Pointer<f32>,
    dx_ptr: T::Pointer<f32>,
    dslope_ptr: T::Pointer<f32>,
    n_elements: i32,
) where
    T::I32Tensor: types::Tensor<i32, 1>,
    T::I32Tensor: Comparison<i32, BoolTensor = T::BoolTensor>,
    T::Pointer<f32>: AddOffsets<i32, 1, T::I32Tensor, Output = T::Tensor<T::Pointer<f32>>>,
{
    let pid = T::program_id(Axis::X);
    let block_start = pid * BLOCK_SIZE;
    let offsets = T::arange(0, BLOCK_SIZE) + block_start;
    let in_bounds = offsets.lt(n_elements);
    let dy = T::load(
        dy_ptr.add_offsets(offsets),
        Some(in_bounds),
        None,
        &[],
        None,
        None,
        None,
        false,
    );
    let x = T::load(
        x_ptr.add_offsets(offsets),
        Some(in_bounds),
        None,
        &[],
        None,
        None,
        None,
        false,
    );
    let slope = T::load(
        slope_ptr.add_offsets(offsets),
        Some(in_bounds),
        None,
        &[],
        None,
        None,
        None,
        false,
    );
    let zero = T::zeros_like(x);
    let x_pos = T::ge(x, zero);
    let dx = T::where_(x_pos, dy, slope * dy);
    let dslope = T::where_(x_pos, zero, x * dy);
    T::store(
        dx_ptr.add_offsets(offsets),
        dx,
        Some(in_bounds),
        &[],
        None,
        None,
    );
    T::store(
        dslope_ptr.add_offsets(offsets),
        dslope,
        Some(in_bounds),
        &[],
        None,
        None,
    );
}

impl teeny_core::model::RuntimeOp for PreluForward {
    fn n_activation_inputs(&self) -> usize {
        2
    }
    fn param_shapes(&self, _: &[&[usize]], _: &[usize]) -> Vec<Vec<usize>> {
        vec![]
    }
    fn pack_args(
        &self,
        inputs: &[(teeny_core::model::RawPtr, &[usize])],
        _: &[teeny_core::model::RawPtr],
        output: teeny_core::model::RawPtr,
        output_shape: &[usize],
        _: i32,
        visitor: &mut dyn teeny_core::device::program::ArgVisitor,
    ) {
        let n: usize = output_shape.iter().product();
        visitor.visit_ptr(inputs[0].0); // x
        visitor.visit_ptr(inputs[1].0); // slope
        visitor.visit_ptr(output);
        visitor.visit_i32(n as i32);
    }
    fn block(&self) -> [u32; 3] {
        [self.block_size as u32, 1, 1]
    }
    fn grid(&self, output_shape: &[usize]) -> [u32; 3] {
        let n: usize = output_shape.iter().product();
        [n.div_ceil(self.block_size as usize) as u32, 1, 1]
    }
    #[cfg(feature = "training")]
    fn has_backward(&self) -> bool {
        true
    }
    #[cfg(feature = "training")]
    fn pack_backward_args(
        &self,
        inputs: &[(teeny_core::model::RawPtr, &[usize])],
        _: &[teeny_core::model::RawPtr],
        _: teeny_core::model::RawPtr,
        output_shape: &[usize],
        grad_output: teeny_core::model::RawPtr,
        _: i32,
        grad_inputs: &[teeny_core::model::RawPtr],
        _: &[teeny_core::model::RawPtr],
        visitor: &mut dyn teeny_core::device::program::ArgVisitor,
    ) {
        let n: usize = output_shape.iter().product();
        visitor.visit_ptr(grad_output);
        visitor.visit_ptr(inputs[0].0); // x
        visitor.visit_ptr(inputs[1].0); // slope
        visitor.visit_ptr(grad_inputs[0]); // dx
        visitor.visit_ptr(grad_inputs[1]); // dslope
        visitor.visit_i32(n as i32);
    }
    #[cfg(feature = "training")]
    fn backward_block(&self) -> [u32; 3] {
        [self.block_size as u32, 1, 1]
    }
    #[cfg(feature = "training")]
    fn backward_grid(&self, _: &[&[usize]], output_shape: &[usize]) -> [u32; 3] {
        let n: usize = output_shape.iter().product();
        [n.div_ceil(self.block_size as usize) as u32, 1, 1]
    }
}

// ── ThresholdedRelu ───────────────────────────────────────────────────────────

/// Forward: y = x if x > alpha else 0
#[kernel]
pub fn thresholded_relu_forward<T: Triton, const BLOCK_SIZE: i32>(
    x_ptr: T::Pointer<f32>,
    y_ptr: T::Pointer<f32>,
    n_elements: i32,
    alpha: f32,
) where
    T::I32Tensor: types::Tensor<i32, 1>,
    T::I32Tensor: Comparison<i32, BoolTensor = T::BoolTensor>,
    T::Pointer<f32>: AddOffsets<i32, 1, T::I32Tensor, Output = T::Tensor<T::Pointer<f32>>>,
{
    let pid = T::program_id(Axis::X);
    let block_start = pid * BLOCK_SIZE;
    let offsets = T::arange(0, BLOCK_SIZE) + block_start;
    let in_bounds = offsets.lt(n_elements);
    let x = T::load(
        x_ptr.add_offsets(offsets),
        Some(in_bounds),
        None,
        &[],
        None,
        None,
        None,
        false,
    );
    let alpha_t = T::full::<f32>(&[BLOCK_SIZE], alpha);
    let above = T::gt(x, alpha_t);
    let y = T::where_(above, x, T::zeros_like(x));
    T::store(
        y_ptr.add_offsets(offsets),
        y,
        Some(in_bounds),
        &[],
        None,
        None,
    );
}

/// Backward: dx = dy if x > alpha else 0
#[kernel]
pub fn thresholded_relu_backward<T: Triton, const BLOCK_SIZE: i32>(
    dy_ptr: T::Pointer<f32>,
    x_ptr: T::Pointer<f32>,
    dx_ptr: T::Pointer<f32>,
    n_elements: i32,
    alpha: f32,
) where
    T::I32Tensor: types::Tensor<i32, 1>,
    T::I32Tensor: Comparison<i32, BoolTensor = T::BoolTensor>,
    T::Pointer<f32>: AddOffsets<i32, 1, T::I32Tensor, Output = T::Tensor<T::Pointer<f32>>>,
{
    let pid = T::program_id(Axis::X);
    let block_start = pid * BLOCK_SIZE;
    let offsets = T::arange(0, BLOCK_SIZE) + block_start;
    let in_bounds = offsets.lt(n_elements);
    let dy = T::load(
        dy_ptr.add_offsets(offsets),
        Some(in_bounds),
        None,
        &[],
        None,
        None,
        None,
        false,
    );
    let x = T::load(
        x_ptr.add_offsets(offsets),
        Some(in_bounds),
        None,
        &[],
        None,
        None,
        None,
        false,
    );
    let alpha_t = T::full::<f32>(&[BLOCK_SIZE], alpha);
    let above = T::gt(x, alpha_t);
    let dx = T::where_(above, dy, T::zeros_like(dy));
    T::store(
        dx_ptr.add_offsets(offsets),
        dx,
        Some(in_bounds),
        &[],
        None,
        None,
    );
}

/// RuntimeOp wrapper for ThresholdedRelu that stores the alpha scalar.
pub struct ThresholdedReluRuntimeOp {
    pub kernel: ThresholdedReluForward,
    pub backward_kernel: ThresholdedReluBackward,
    pub alpha: f32,
}

impl ThresholdedReluRuntimeOp {
    pub fn new(block_size: i32, alpha: f32) -> Self {
        Self {
            kernel: ThresholdedReluForward::new(block_size),
            backward_kernel: ThresholdedReluBackward::new(block_size),
            alpha,
        }
    }
    pub fn forward_source(&self) -> &str {
        &self.kernel.source
    }
    pub fn backward_source(&self) -> &str {
        &self.backward_kernel.source
    }
    pub fn kernel_name(&self) -> &str {
        self.kernel.name
    }
}

impl teeny_core::model::RuntimeOp for ThresholdedReluRuntimeOp {
    fn n_activation_inputs(&self) -> usize {
        1
    }
    fn param_shapes(&self, _: &[&[usize]], _: &[usize]) -> Vec<Vec<usize>> {
        vec![]
    }
    fn pack_args(
        &self,
        inputs: &[(teeny_core::model::RawPtr, &[usize])],
        _: &[teeny_core::model::RawPtr],
        output: teeny_core::model::RawPtr,
        output_shape: &[usize],
        _: i32,
        visitor: &mut dyn teeny_core::device::program::ArgVisitor,
    ) {
        let n: usize = output_shape.iter().product();
        visitor.visit_ptr(inputs[0].0);
        visitor.visit_ptr(output);
        visitor.visit_i32(n as i32);
        visitor.visit_f32(self.alpha);
    }
    fn block(&self) -> [u32; 3] {
        [self.kernel.block_size as u32, 1, 1]
    }
    fn grid(&self, output_shape: &[usize]) -> [u32; 3] {
        let n: usize = output_shape.iter().product();
        [n.div_ceil(self.kernel.block_size as usize) as u32, 1, 1]
    }
    #[cfg(feature = "training")]
    fn has_backward(&self) -> bool {
        true
    }
    #[cfg(feature = "training")]
    fn pack_backward_args(
        &self,
        inputs: &[(teeny_core::model::RawPtr, &[usize])],
        _: &[teeny_core::model::RawPtr],
        _: teeny_core::model::RawPtr,
        output_shape: &[usize],
        grad_output: teeny_core::model::RawPtr,
        _: i32,
        grad_inputs: &[teeny_core::model::RawPtr],
        _: &[teeny_core::model::RawPtr],
        visitor: &mut dyn teeny_core::device::program::ArgVisitor,
    ) {
        let n: usize = output_shape.iter().product();
        visitor.visit_ptr(grad_output);
        visitor.visit_ptr(inputs[0].0);
        visitor.visit_ptr(grad_inputs[0]);
        visitor.visit_i32(n as i32);
        visitor.visit_f32(self.alpha);
    }
    #[cfg(feature = "training")]
    fn backward_block(&self) -> [u32; 3] {
        [self.kernel.block_size as u32, 1, 1]
    }
    #[cfg(feature = "training")]
    fn backward_grid(&self, _: &[&[usize]], output_shape: &[usize]) -> [u32; 3] {
        let n: usize = output_shape.iter().product();
        [n.div_ceil(self.kernel.block_size as usize) as u32, 1, 1]
    }
}

// ── Shrink ────────────────────────────────────────────────────────────────────

/// Forward: y = x - bias if x > lambd, x + bias if x < -lambd, else 0
#[kernel]
pub fn shrink_forward<T: Triton, const BLOCK_SIZE: i32>(
    x_ptr: T::Pointer<f32>,
    y_ptr: T::Pointer<f32>,
    n_elements: i32,
    lambd: f32,
    bias: f32,
) where
    T::I32Tensor: types::Tensor<i32, 1>,
    T::I32Tensor: Comparison<i32, BoolTensor = T::BoolTensor>,
    T::Pointer<f32>: AddOffsets<i32, 1, T::I32Tensor, Output = T::Tensor<T::Pointer<f32>>>,
{
    let pid = T::program_id(Axis::X);
    let block_start = pid * BLOCK_SIZE;
    let offsets = T::arange(0, BLOCK_SIZE) + block_start;
    let in_bounds = offsets.lt(n_elements);
    let x = T::load(
        x_ptr.add_offsets(offsets),
        Some(in_bounds),
        None,
        &[],
        None,
        None,
        None,
        false,
    );
    let lam = T::full::<f32>(&[BLOCK_SIZE], lambd);
    let neg_lam = T::full::<f32>(&[BLOCK_SIZE], -lambd);
    let b = T::full::<f32>(&[BLOCK_SIZE], bias);
    let x_gt = T::gt(x, lam);
    let x_lt = T::lt(x, neg_lam);
    let y_upper = x - b;
    let y_lower = x + b;
    let y_mid = T::where_(x_lt, y_lower, T::zeros_like(x));
    let y = T::where_(x_gt, y_upper, y_mid);
    T::store(
        y_ptr.add_offsets(offsets),
        y,
        Some(in_bounds),
        &[],
        None,
        None,
    );
}

/// Backward: dx = dy if |x| > lambd else 0
#[kernel]
pub fn shrink_backward<T: Triton, const BLOCK_SIZE: i32>(
    dy_ptr: T::Pointer<f32>,
    x_ptr: T::Pointer<f32>,
    dx_ptr: T::Pointer<f32>,
    n_elements: i32,
    lambd: f32,
) where
    T::I32Tensor: types::Tensor<i32, 1>,
    T::I32Tensor: Comparison<i32, BoolTensor = T::BoolTensor>,
    T::Pointer<f32>: AddOffsets<i32, 1, T::I32Tensor, Output = T::Tensor<T::Pointer<f32>>>,
{
    let pid = T::program_id(Axis::X);
    let block_start = pid * BLOCK_SIZE;
    let offsets = T::arange(0, BLOCK_SIZE) + block_start;
    let in_bounds = offsets.lt(n_elements);
    let dy = T::load(
        dy_ptr.add_offsets(offsets),
        Some(in_bounds),
        None,
        &[],
        None,
        None,
        None,
        false,
    );
    let x = T::load(
        x_ptr.add_offsets(offsets),
        Some(in_bounds),
        None,
        &[],
        None,
        None,
        None,
        false,
    );
    let lam = T::full::<f32>(&[BLOCK_SIZE], lambd);
    let outside = T::gt(T::abs(x), lam);
    let dx = T::where_(outside, dy, T::zeros_like(dy));
    T::store(
        dx_ptr.add_offsets(offsets),
        dx,
        Some(in_bounds),
        &[],
        None,
        None,
    );
}

/// RuntimeOp wrapper for Shrink that stores lambd and bias.
pub struct ShrinkRuntimeOp {
    pub kernel: ShrinkForward,
    pub backward_kernel: ShrinkBackward,
    pub lambd: f32,
    pub bias: f32,
}

impl ShrinkRuntimeOp {
    pub fn new(block_size: i32, lambd: f32, bias: f32) -> Self {
        Self {
            kernel: ShrinkForward::new(block_size),
            backward_kernel: ShrinkBackward::new(block_size),
            lambd,
            bias,
        }
    }
    pub fn forward_source(&self) -> &str {
        &self.kernel.source
    }
    pub fn backward_source(&self) -> &str {
        &self.backward_kernel.source
    }
    pub fn kernel_name(&self) -> &str {
        self.kernel.name
    }
}

impl teeny_core::model::RuntimeOp for ShrinkRuntimeOp {
    fn n_activation_inputs(&self) -> usize {
        1
    }
    fn param_shapes(&self, _: &[&[usize]], _: &[usize]) -> Vec<Vec<usize>> {
        vec![]
    }
    fn pack_args(
        &self,
        inputs: &[(teeny_core::model::RawPtr, &[usize])],
        _: &[teeny_core::model::RawPtr],
        output: teeny_core::model::RawPtr,
        output_shape: &[usize],
        _: i32,
        visitor: &mut dyn teeny_core::device::program::ArgVisitor,
    ) {
        let n: usize = output_shape.iter().product();
        visitor.visit_ptr(inputs[0].0);
        visitor.visit_ptr(output);
        visitor.visit_i32(n as i32);
        visitor.visit_f32(self.lambd);
        visitor.visit_f32(self.bias);
    }
    fn block(&self) -> [u32; 3] {
        [self.kernel.block_size as u32, 1, 1]
    }
    fn grid(&self, output_shape: &[usize]) -> [u32; 3] {
        let n: usize = output_shape.iter().product();
        [n.div_ceil(self.kernel.block_size as usize) as u32, 1, 1]
    }
    #[cfg(feature = "training")]
    fn has_backward(&self) -> bool {
        true
    }
    #[cfg(feature = "training")]
    fn pack_backward_args(
        &self,
        inputs: &[(teeny_core::model::RawPtr, &[usize])],
        _: &[teeny_core::model::RawPtr],
        _: teeny_core::model::RawPtr,
        output_shape: &[usize],
        grad_output: teeny_core::model::RawPtr,
        _: i32,
        grad_inputs: &[teeny_core::model::RawPtr],
        _: &[teeny_core::model::RawPtr],
        visitor: &mut dyn teeny_core::device::program::ArgVisitor,
    ) {
        let n: usize = output_shape.iter().product();
        visitor.visit_ptr(grad_output);
        visitor.visit_ptr(inputs[0].0);
        visitor.visit_ptr(grad_inputs[0]);
        visitor.visit_i32(n as i32);
        visitor.visit_f32(self.lambd);
    }
    #[cfg(feature = "training")]
    fn backward_block(&self) -> [u32; 3] {
        [self.kernel.block_size as u32, 1, 1]
    }
    #[cfg(feature = "training")]
    fn backward_grid(&self, _: &[&[usize]], output_shape: &[usize]) -> [u32; 3] {
        let n: usize = output_shape.iter().product();
        [n.div_ceil(self.kernel.block_size as usize) as u32, 1, 1]
    }
}

// ── LogSoftmax (row-wise) ─────────────────────────────────────────────────────
//
// Grid: one CTA per row. BLOCK_SIZE must equal n_cols (power of 2).

/// Forward: y = x - log(sum(exp(x)))  [numerically stable: subtract max first]
#[kernel]
pub fn log_softmax_forward<T: Triton, const BLOCK_SIZE: i32>(
    x_ptr: T::Pointer<f32>,
    y_ptr: T::Pointer<f32>,
    _n_rows: i32,
    n_cols: i32,
) where
    T::I32Tensor: types::Tensor<i32, 1>,
    T::I32Tensor: Comparison<i32, BoolTensor = T::BoolTensor>,
    T::Pointer<f32>: AddOffsets<i32, 1, T::I32Tensor, Output = T::Tensor<T::Pointer<f32>>>,
{
    let pid = T::program_id(Axis::X);
    let row_offset = pid * n_cols;
    let col_offsets = T::arange(0, BLOCK_SIZE);
    let offsets = col_offsets + row_offset;
    let x = T::load(
        x_ptr.add_offsets(offsets),
        None,
        None,
        &[],
        None,
        None,
        None,
        false,
    );
    let m = T::max(x, Some(0), true); // max for numerical stability
    let x_m = x - m;
    let log_sum = T::log(T::sum(T::exp(x_m), Some(0), true));
    let y = x_m - log_sum;
    T::store(y_ptr.add_offsets(offsets), y, None, &[], None, None);
}

/// Backward: dx = dy - softmax(x) * sum(dy)
#[kernel]
pub fn log_softmax_backward<T: Triton, const BLOCK_SIZE: i32>(
    dy_ptr: T::Pointer<f32>,
    y_ptr: T::Pointer<f32>,
    dx_ptr: T::Pointer<f32>,
    _n_rows: i32,
    n_cols: i32,
) where
    T::I32Tensor: types::Tensor<i32, 1>,
    T::I32Tensor: Comparison<i32, BoolTensor = T::BoolTensor>,
    T::Pointer<f32>: AddOffsets<i32, 1, T::I32Tensor, Output = T::Tensor<T::Pointer<f32>>>,
{
    let pid = T::program_id(Axis::X);
    let row_offset = pid * n_cols;
    let col_offsets = T::arange(0, BLOCK_SIZE);
    let offsets = col_offsets + row_offset;
    let dy = T::load(
        dy_ptr.add_offsets(offsets),
        None,
        None,
        &[],
        None,
        None,
        None,
        false,
    );
    let y = T::load(
        y_ptr.add_offsets(offsets),
        None,
        None,
        &[],
        None,
        None,
        None,
        false,
    );
    // softmax(x) = exp(log_softmax(x))
    let sm = T::exp(y);
    let sum_dy = T::sum(dy, Some(0), true);
    let dx = dy - sm * sum_dy;
    T::store(dx_ptr.add_offsets(offsets), dx, None, &[], None, None);
}

impl teeny_core::model::RuntimeOp for LogSoftmaxForward {
    fn n_activation_inputs(&self) -> usize {
        1
    }
    fn param_shapes(&self, _: &[&[usize]], _: &[usize]) -> Vec<Vec<usize>> {
        vec![]
    }
    fn pack_args(
        &self,
        inputs: &[(teeny_core::model::RawPtr, &[usize])],
        _: &[teeny_core::model::RawPtr],
        output: teeny_core::model::RawPtr,
        output_shape: &[usize],
        _: i32,
        visitor: &mut dyn teeny_core::device::program::ArgVisitor,
    ) {
        let n_rows = output_shape.first().copied().unwrap_or(1) as i32;
        let n_cols = output_shape.last().copied().unwrap_or(1) as i32;
        visitor.visit_ptr(inputs[0].0);
        visitor.visit_ptr(output);
        visitor.visit_i32(n_rows);
        visitor.visit_i32(n_cols);
    }
    fn block(&self) -> [u32; 3] {
        [self.block_size as u32, 1, 1]
    }
    fn grid(&self, output_shape: &[usize]) -> [u32; 3] {
        [output_shape.first().copied().unwrap_or(1) as u32, 1, 1]
    }
    #[cfg(feature = "training")]
    fn has_backward(&self) -> bool {
        true
    }
    #[cfg(feature = "training")]
    fn pack_backward_args(
        &self,
        _: &[(teeny_core::model::RawPtr, &[usize])],
        _: &[teeny_core::model::RawPtr],
        output: teeny_core::model::RawPtr,
        output_shape: &[usize],
        grad_output: teeny_core::model::RawPtr,
        _: i32,
        grad_inputs: &[teeny_core::model::RawPtr],
        _: &[teeny_core::model::RawPtr],
        visitor: &mut dyn teeny_core::device::program::ArgVisitor,
    ) {
        let n_rows = output_shape.first().copied().unwrap_or(1) as i32;
        let n_cols = output_shape.last().copied().unwrap_or(1) as i32;
        visitor.visit_ptr(grad_output);
        visitor.visit_ptr(output);
        visitor.visit_ptr(grad_inputs[0]);
        visitor.visit_i32(n_rows);
        visitor.visit_i32(n_cols);
    }
    #[cfg(feature = "training")]
    fn backward_block(&self) -> [u32; 3] {
        [self.block_size as u32, 1, 1]
    }
    #[cfg(feature = "training")]
    fn backward_grid(&self, _: &[&[usize]], output_shape: &[usize]) -> [u32; 3] {
        [output_shape.first().copied().unwrap_or(1) as u32, 1, 1]
    }
}
