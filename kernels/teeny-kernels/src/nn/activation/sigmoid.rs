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

use teeny_core::dtype::Float;
use teeny_macros::kernel;
use teeny_triton::triton::{
    types::{AddOffsets, Comparison},
    *,
};

// ── Sigmoid ──────────────────────────────────────────────────────────────────

/// Forward: y = 1 / (1 + exp(-x))
#[kernel(backward = SigmoidBackward)]
pub fn sigmoid_forward<T: Triton, D: Float, const BLOCK_SIZE: i32>(
    x_ptr: InPtr<T::Pointer<D>>,
    y_ptr: OutPtr<T::Pointer<D>>,
    n_elements: i32,
) where
    T::I32Tensor: types::Tensor<i32, 1>,
    T::I32Tensor: Comparison<i32, BoolTensor = T::BoolTensor>,
    T::Pointer<D>: AddOffsets<i32, 1, T::I32Tensor, Output = T::Tensor<T::Pointer<D>>>,
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
    let one = T::full(&[BLOCK_SIZE], D::from_f64(1.0));
    let neg1 = T::full(&[BLOCK_SIZE], D::from_f64(-1.0));
    let y = one / (one + T::exp(neg1 * x));
    T::store(
        y_ptr.add_offsets(offsets),
        y,
        Some(in_bounds),
        &[],
        None,
        None,
    );
}

/// Backward: dx = dy * y * (1 - y) = dy * (y - y²)
#[kernel]
pub fn sigmoid_backward<T: Triton, D: Float, const BLOCK_SIZE: i32>(
    dy_ptr: InPtr<T::Pointer<D>>,
    y_ptr: InPtr<T::Pointer<D>>,
    dx_ptr: OutPtr<T::Pointer<D>>,
    n_elements: i32,
) where
    T::I32Tensor: types::Tensor<i32, 1>,
    T::I32Tensor: Comparison<i32, BoolTensor = T::BoolTensor>,
    T::Pointer<D>: AddOffsets<i32, 1, T::I32Tensor, Output = T::Tensor<T::Pointer<D>>>,
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
    let y = T::load(
        y_ptr.add_offsets(offsets),
        Some(in_bounds),
        None,
        &[],
        None,
        None,
        None,
        false,
    );

    let dx = dy * (y - y * y);
    T::store(
        dx_ptr.add_offsets(offsets),
        dx,
        Some(in_bounds),
        &[],
        None,
        None,
    );
}

// ── SiLU (Swish) ─────────────────────────────────────────────────────────────

/// Forward: y = x * sigmoid(x)
#[kernel(backward = SiluBackward)]
pub fn silu_forward<T: Triton, D: Float, const BLOCK_SIZE: i32>(
    x_ptr: InPtr<T::Pointer<D>>,
    y_ptr: OutPtr<T::Pointer<D>>,
    n_elements: i32,
) where
    T::I32Tensor: types::Tensor<i32, 1>,
    T::I32Tensor: Comparison<i32, BoolTensor = T::BoolTensor>,
    T::Pointer<D>: AddOffsets<i32, 1, T::I32Tensor, Output = T::Tensor<T::Pointer<D>>>,
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
    let one = T::full(&[BLOCK_SIZE], D::from_f64(1.0));
    let neg1 = T::full(&[BLOCK_SIZE], D::from_f64(-1.0));
    let s = one / (one + T::exp(neg1 * x));
    let y = x * s;
    T::store(
        y_ptr.add_offsets(offsets),
        y,
        Some(in_bounds),
        &[],
        None,
        None,
    );
}

/// Backward: dx = dy * (sigmoid(x) + y * (1 - sigmoid(x)))
///         = dy * (s + y - y*s)   where s = sigmoid(x)
#[kernel]
pub fn silu_backward<T: Triton, D: Float, const BLOCK_SIZE: i32>(
    dy_ptr: InPtr<T::Pointer<D>>,
    x_ptr: InPtr<T::Pointer<D>>,
    dx_ptr: OutPtr<T::Pointer<D>>,
    n_elements: i32,
) where
    T::I32Tensor: types::Tensor<i32, 1>,
    T::I32Tensor: Comparison<i32, BoolTensor = T::BoolTensor>,
    T::Pointer<D>: AddOffsets<i32, 1, T::I32Tensor, Output = T::Tensor<T::Pointer<D>>>,
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
    let one = T::full(&[BLOCK_SIZE], D::from_f64(1.0));
    let neg1 = T::full(&[BLOCK_SIZE], D::from_f64(-1.0));
    let s = one / (one + T::exp(neg1 * x));
    let y = x * s;
    // d(silu)/dx = s + x*s*(1-s) = s + y - y*s
    let dx = dy * (s + y - y * s);
    T::store(
        dx_ptr.add_offsets(offsets),
        dx,
        Some(in_bounds),
        &[],
        None,
        None,
    );
}

// ── LogSigmoid ────────────────────────────────────────────────────────────────

/// Forward: y = log(sigmoid(x)) = -log(1 + exp(-x))
#[kernel(backward = LogsigmoidBackward)]
pub fn logsigmoid_forward<T: Triton, D: Float, const BLOCK_SIZE: i32>(
    x_ptr: InPtr<T::Pointer<D>>,
    y_ptr: OutPtr<T::Pointer<D>>,
    n_elements: i32,
) where
    T::I32Tensor: types::Tensor<i32, 1>,
    T::I32Tensor: Comparison<i32, BoolTensor = T::BoolTensor>,
    T::Pointer<D>: AddOffsets<i32, 1, T::I32Tensor, Output = T::Tensor<T::Pointer<D>>>,
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
    let one = T::full(&[BLOCK_SIZE], D::from_f64(1.0));
    let neg1 = T::full(&[BLOCK_SIZE], D::from_f64(-1.0));
    // -log(1 + exp(-x)) = log(1/(1+exp(-x))) = log(sigmoid(x))
    // But we want to avoid negating the result: use (neg1 * log(1 + exp(neg1*x)))
    // Actually: y = neg1 * log(one + T::exp(neg1 * x))
    // But neg1 * log(...) would require negating a tensor result.
    // Use subtraction: y = T::zeros_like(x) - T::log(one + T::exp(neg1 * x))
    let zeros = T::zeros_like(x);
    let y = zeros - T::log(one + T::exp(neg1 * x));
    T::store(
        y_ptr.add_offsets(offsets),
        y,
        Some(in_bounds),
        &[],
        None,
        None,
    );
}

/// Backward: dx = dy * sigmoid(-x) = dy / (1 + exp(x))
#[kernel]
pub fn logsigmoid_backward<T: Triton, D: Float, const BLOCK_SIZE: i32>(
    dy_ptr: InPtr<T::Pointer<D>>,
    x_ptr: InPtr<T::Pointer<D>>,
    dx_ptr: OutPtr<T::Pointer<D>>,
    n_elements: i32,
) where
    T::I32Tensor: types::Tensor<i32, 1>,
    T::I32Tensor: Comparison<i32, BoolTensor = T::BoolTensor>,
    T::Pointer<D>: AddOffsets<i32, 1, T::I32Tensor, Output = T::Tensor<T::Pointer<D>>>,
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
    let one = T::full(&[BLOCK_SIZE], D::from_f64(1.0));
    // sigmoid(-x) = 1 / (1 + exp(x))
    let dx = dy / (one + T::exp(x));
    T::store(
        dx_ptr.add_offsets(offsets),
        dx,
        Some(in_bounds),
        &[],
        None,
        None,
    );
}

pub struct SigmoidOp<D: Float> {
    pub forward: SigmoidForward<D>,
    pub backward: SigmoidBackward<D>,
}

pub struct SiluOp<D: Float> {
    pub forward: SiluForward<D>,
    pub backward: SiluBackward<D>,
}

pub struct LogsigmoidOp<D: Float> {
    pub forward: LogsigmoidForward<D>,
    pub backward: LogsigmoidBackward<D>,
}

// ── RuntimeOp for Sigmoid forward ────────────────────────────────────────────

impl<D: Float + Send + Sync + 'static> teeny_core::model::RuntimeOp for SigmoidForward<D> {
    fn n_activation_inputs(&self) -> usize {
        1
    }

    fn param_shapes(&self, _: &[&[usize]], _: &[usize]) -> Vec<Vec<usize>> {
        Vec::new()
    }

    fn pack_args(
        &self,
        inputs: &[(teeny_core::model::RawPtr, &[usize])],
        _params: &[teeny_core::model::RawPtr],
        output: teeny_core::model::RawPtr,
        output_shape: &[usize],
        _output_row_stride: i32,
        visitor: &mut dyn teeny_core::device::program::ArgVisitor,
    ) {
        let n: usize = output_shape.iter().product();
        visitor.visit_ptr(inputs[0].0);
        visitor.visit_ptr(output);
        visitor.visit_i32(n as i32);
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
        _inputs: &[(teeny_core::model::RawPtr, &[usize])],
        _params: &[teeny_core::model::RawPtr],
        output: teeny_core::model::RawPtr,
        output_shape: &[usize],
        grad_output: teeny_core::model::RawPtr,
        _grad_output_row_stride: i32,
        grad_inputs: &[teeny_core::model::RawPtr],
        _grad_params: &[teeny_core::model::RawPtr],
        visitor: &mut dyn teeny_core::device::program::ArgVisitor,
    ) {
        let n: usize = output_shape.iter().product();
        visitor.visit_ptr(grad_output); // dy_ptr
        visitor.visit_ptr(output); // y_ptr (saved output, not x)
        visitor.visit_ptr(grad_inputs[0]); // dx_ptr
        visitor.visit_i32(n as i32);
    }

    #[cfg(feature = "training")]
    fn backward_grid(&self, _: &[&[usize]], output_shape: &[usize]) -> [u32; 3] {
        let n: usize = output_shape.iter().product();
        [n.div_ceil(self.block_size as usize) as u32, 1, 1]
    }
}

// ── RuntimeOp for SiLU forward ────────────────────────────────────────────────

impl<D: Float + Send + Sync + 'static> teeny_core::model::RuntimeOp for SiluForward<D> {
    fn n_activation_inputs(&self) -> usize {
        1
    }

    fn param_shapes(&self, _: &[&[usize]], _: &[usize]) -> Vec<Vec<usize>> {
        Vec::new()
    }

    fn pack_args(
        &self,
        inputs: &[(teeny_core::model::RawPtr, &[usize])],
        _params: &[teeny_core::model::RawPtr],
        output: teeny_core::model::RawPtr,
        output_shape: &[usize],
        _output_row_stride: i32,
        visitor: &mut dyn teeny_core::device::program::ArgVisitor,
    ) {
        let n: usize = output_shape.iter().product();
        visitor.visit_ptr(inputs[0].0);
        visitor.visit_ptr(output);
        visitor.visit_i32(n as i32);
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
        _params: &[teeny_core::model::RawPtr],
        _output: teeny_core::model::RawPtr,
        output_shape: &[usize],
        grad_output: teeny_core::model::RawPtr,
        _grad_output_row_stride: i32,
        grad_inputs: &[teeny_core::model::RawPtr],
        _grad_params: &[teeny_core::model::RawPtr],
        visitor: &mut dyn teeny_core::device::program::ArgVisitor,
    ) {
        let n: usize = output_shape.iter().product();
        visitor.visit_ptr(grad_output); // dy_ptr
        visitor.visit_ptr(inputs[0].0); // x_ptr (saved activation)
        visitor.visit_ptr(grad_inputs[0]); // dx_ptr
        visitor.visit_i32(n as i32);
    }

    #[cfg(feature = "training")]
    fn backward_grid(&self, _: &[&[usize]], output_shape: &[usize]) -> [u32; 3] {
        let n: usize = output_shape.iter().product();
        [n.div_ceil(self.block_size as usize) as u32, 1, 1]
    }
}
