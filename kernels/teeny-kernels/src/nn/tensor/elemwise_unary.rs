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

use teeny_core::dtype::{Float, Num};
use teeny_macros::kernel;
use teeny_triton::triton::{
    types::{AddOffsets, Comparison},
    *,
};

// ── Helper macro for standard unary Float RuntimeOp ──────────────────────────
//
// All float-only unary ops share the same RuntimeOp: 1 input, 1 output, n i32.
// Backward packs: dy, x, dx, n.

macro_rules! impl_float_unary_runtime_op {
    ($Fwd:ident) => {
        impl<D: Float + Send + Sync + 'static> teeny_core::model::RuntimeOp for $Fwd<D> {
            fn n_activation_inputs(&self) -> usize { 1 }
            fn param_shapes(&self, _: &[&[usize]], _: &[usize]) -> Vec<Vec<usize>> { vec![] }
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
            fn block(&self) -> [u32; 3] { [self.block_size as u32, 1, 1] }
            fn grid(&self, output_shape: &[usize]) -> [u32; 3] {
                let n: usize = output_shape.iter().product();
                [n.div_ceil(self.block_size as usize) as u32, 1, 1]
            }
            #[cfg(feature = "training")]
            fn has_backward(&self) -> bool { true }
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
            fn backward_block(&self) -> [u32; 3] { [self.block_size as u32, 1, 1] }
            #[cfg(feature = "training")]
            fn backward_grid(&self, _: &[&[usize]], output_shape: &[usize]) -> [u32; 3] {
                let n: usize = output_shape.iter().product();
                [n.div_ceil(self.block_size as usize) as u32, 1, 1]
            }
        }
    };
}

macro_rules! impl_float_unary_runtime_op_no_bwd {
    ($Fwd:ident) => {
        impl<D: Float + Send + Sync + 'static> teeny_core::model::RuntimeOp for $Fwd<D> {
            fn n_activation_inputs(&self) -> usize { 1 }
            fn param_shapes(&self, _: &[&[usize]], _: &[usize]) -> Vec<Vec<usize>> { vec![] }
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
            fn block(&self) -> [u32; 3] { [self.block_size as u32, 1, 1] }
            fn grid(&self, output_shape: &[usize]) -> [u32; 3] {
                let n: usize = output_shape.iter().product();
                [n.div_ceil(self.block_size as usize) as u32, 1, 1]
            }
        }
    };
}

macro_rules! impl_num_unary_runtime_op {
    ($Fwd:ident) => {
        impl<D: Num + Send + Sync + 'static> teeny_core::model::RuntimeOp for $Fwd<D> {
            fn n_activation_inputs(&self) -> usize { 1 }
            fn param_shapes(&self, _: &[&[usize]], _: &[usize]) -> Vec<Vec<usize>> { vec![] }
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
            fn block(&self) -> [u32; 3] { [self.block_size as u32, 1, 1] }
            fn grid(&self, output_shape: &[usize]) -> [u32; 3] {
                let n: usize = output_shape.iter().product();
                [n.div_ceil(self.block_size as usize) as u32, 1, 1]
            }
        }
    };
}

macro_rules! impl_num_unary_runtime_op_with_bwd {
    ($Fwd:ident) => {
        impl<D: Num + Send + Sync + 'static> teeny_core::model::RuntimeOp for $Fwd<D> {
            fn n_activation_inputs(&self) -> usize { 1 }
            fn param_shapes(&self, _: &[&[usize]], _: &[usize]) -> Vec<Vec<usize>> { vec![] }
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
            fn block(&self) -> [u32; 3] { [self.block_size as u32, 1, 1] }
            fn grid(&self, output_shape: &[usize]) -> [u32; 3] {
                let n: usize = output_shape.iter().product();
                [n.div_ceil(self.block_size as usize) as u32, 1, 1]
            }
            #[cfg(feature = "training")]
            fn has_backward(&self) -> bool { true }
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
            fn backward_block(&self) -> [u32; 3] { [self.block_size as u32, 1, 1] }
            #[cfg(feature = "training")]
            fn backward_grid(&self, _: &[&[usize]], output_shape: &[usize]) -> [u32; 3] {
                let n: usize = output_shape.iter().product();
                [n.div_ceil(self.block_size as usize) as u32, 1, 1]
            }
        }
    };
}

// ── Neg backward RuntimeOp (dy only, no saved input) ─────────────────────────
macro_rules! impl_num_neg_bwd_runtime_op {
    ($Fwd:ident) => {
        impl<D: Num + Send + Sync + 'static> teeny_core::model::RuntimeOp for $Fwd<D> {
            fn n_activation_inputs(&self) -> usize { 1 }
            fn param_shapes(&self, _: &[&[usize]], _: &[usize]) -> Vec<Vec<usize>> { vec![] }
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
            fn block(&self) -> [u32; 3] { [self.block_size as u32, 1, 1] }
            fn grid(&self, output_shape: &[usize]) -> [u32; 3] {
                let n: usize = output_shape.iter().product();
                [n.div_ceil(self.block_size as usize) as u32, 1, 1]
            }
            #[cfg(feature = "training")]
            fn has_backward(&self) -> bool { true }
            #[cfg(feature = "training")]
            fn pack_backward_args(
                &self,
                _: &[(teeny_core::model::RawPtr, &[usize])],
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
                visitor.visit_ptr(grad_inputs[0]);
                visitor.visit_i32(n as i32);
            }
            #[cfg(feature = "training")]
            fn backward_block(&self) -> [u32; 3] { [self.block_size as u32, 1, 1] }
            #[cfg(feature = "training")]
            fn backward_grid(&self, _: &[&[usize]], output_shape: &[usize]) -> [u32; 3] {
                let n: usize = output_shape.iter().product();
                [n.div_ceil(self.block_size as usize) as u32, 1, 1]
            }
        }
    };
}

// ── Abs ───────────────────────────────────────────────────────────────────────

/// Forward: y = |x|
#[kernel]
pub fn elemwise_abs_forward<T: Triton, D: Num, const BLOCK_SIZE: i32>(
    x_ptr: T::Pointer<D>,
    y_ptr: T::Pointer<D>,
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
    let x = T::load(x_ptr.add_offsets(offsets), Some(in_bounds), None, &[], None, None, None, false);
    let y = T::abs(x);
    T::store(y_ptr.add_offsets(offsets), y, Some(in_bounds), &[], None, None);
}

/// Backward: dx = sign(x) * dy  where sign = 1 if x>0, -1 if x<0, 0 if x==0
#[kernel]
pub fn elemwise_abs_backward<T: Triton, D: Num, const BLOCK_SIZE: i32>(
    dy_ptr: T::Pointer<D>,
    x_ptr: T::Pointer<D>,
    dx_ptr: T::Pointer<D>,
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
    let dy = T::load(dy_ptr.add_offsets(offsets), Some(in_bounds), None, &[], None, None, None, false);
    let x  = T::load(x_ptr.add_offsets(offsets),  Some(in_bounds), None, &[], None, None, None, false);
    let zeros = T::zeros_like(x);
    let ones  = T::cast::<i32, D>(T::full::<i32>(&[BLOCK_SIZE], 1), None, false);
    let neg   = T::cast::<i32, D>(T::full::<i32>(&[BLOCK_SIZE], -1), None, false);
    let pos_mask = T::gt(x, zeros);
    let neg_mask = T::lt(x, zeros);
    let sign = T::where_(pos_mask, ones, T::where_(neg_mask, neg, zeros));
    let dx = sign * dy;
    T::store(dx_ptr.add_offsets(offsets), dx, Some(in_bounds), &[], None, None);
}

impl_num_unary_runtime_op_with_bwd!(ElemwiseAbsForward);

// ── Neg ───────────────────────────────────────────────────────────────────────

/// Forward: y = -x
#[kernel]
pub fn elemwise_neg_forward<T: Triton, D: Num, const BLOCK_SIZE: i32>(
    x_ptr: T::Pointer<D>,
    y_ptr: T::Pointer<D>,
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
    let x = T::load(x_ptr.add_offsets(offsets), Some(in_bounds), None, &[], None, None, None, false);
    let y = -x;
    T::store(y_ptr.add_offsets(offsets), y, Some(in_bounds), &[], None, None);
}

/// Backward: dx = -dy
#[kernel]
pub fn elemwise_neg_backward<T: Triton, D: Num, const BLOCK_SIZE: i32>(
    dy_ptr: T::Pointer<D>,
    dx_ptr: T::Pointer<D>,
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
    let dy = T::load(dy_ptr.add_offsets(offsets), Some(in_bounds), None, &[], None, None, None, false);
    let dx = -dy;
    T::store(dx_ptr.add_offsets(offsets), dx, Some(in_bounds), &[], None, None);
}

impl_num_neg_bwd_runtime_op!(ElemwiseNegForward);

// ── Sign ──────────────────────────────────────────────────────────────────────

/// Forward: y = 1 if x > 0, -1 if x < 0, 0 if x == 0
#[kernel]
pub fn elemwise_sign_forward<T: Triton, D: Num, const BLOCK_SIZE: i32>(
    x_ptr: T::Pointer<D>,
    y_ptr: T::Pointer<D>,
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
    let x = T::load(x_ptr.add_offsets(offsets), Some(in_bounds), None, &[], None, None, None, false);
    let zeros = T::zeros_like(x);
    let ones  = T::cast::<i32, D>(T::full::<i32>(&[BLOCK_SIZE], 1), None, false);
    let neg   = T::cast::<i32, D>(T::full::<i32>(&[BLOCK_SIZE], -1), None, false);
    let pos_mask = T::gt(x, zeros);
    let neg_mask = T::lt(x, zeros);
    let y = T::where_(pos_mask, ones, T::where_(neg_mask, neg, zeros));
    T::store(y_ptr.add_offsets(offsets), y, Some(in_bounds), &[], None, None);
}

impl_num_unary_runtime_op!(ElemwiseSignForward);

// ── IsNaN ─────────────────────────────────────────────────────────────────────

/// Forward: y = 1.0 if x is NaN else 0.0
#[kernel]
pub fn elemwise_isnan_forward<T: Triton, D: Float, const BLOCK_SIZE: i32>(
    x_ptr: T::Pointer<D>,
    y_ptr: T::Pointer<D>,
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
    let x = T::load(x_ptr.add_offsets(offsets), Some(in_bounds), None, &[], None, None, None, false);
    // Triton uses ordered comparison: eq(NaN, NaN) = False.
    // Exploit: where(eq(x, x), 0, 1) yields 1 for NaN, 0 otherwise.
    let one  = T::full::<D>(&[BLOCK_SIZE], D::from_f64(1.0));
    let zero = T::full::<D>(&[BLOCK_SIZE], D::from_f64(0.0));
    let is_not_nan = T::eq(x, x);  // False for NaN (ordered), True for normal
    let y = T::where_(is_not_nan, zero, one);
    T::store(y_ptr.add_offsets(offsets), y, Some(in_bounds), &[], None, None);
}

impl<D: Float + Send + Sync + 'static> teeny_core::model::RuntimeOp for ElemwiseIsnanForward<D> {
    fn n_activation_inputs(&self) -> usize { 1 }
    fn param_shapes(&self, _: &[&[usize]], _: &[usize]) -> Vec<Vec<usize>> { vec![] }
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
    fn block(&self) -> [u32; 3] { [self.block_size as u32, 1, 1] }
    fn grid(&self, output_shape: &[usize]) -> [u32; 3] {
        let n: usize = output_shape.iter().product();
        [n.div_ceil(self.block_size as usize) as u32, 1, 1]
    }
}

// ── Ceil (D: Float) ───────────────────────────────────────────────────────────

/// Forward: y = ceil(x)
#[kernel]
pub fn elemwise_ceil_forward<T: Triton, D: Float, const BLOCK_SIZE: i32>(
    x_ptr: T::Pointer<D>,
    y_ptr: T::Pointer<D>,
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
    let x = T::load(x_ptr.add_offsets(offsets), Some(in_bounds), None, &[], None, None, None, false);
    let y = T::ceil(x);
    T::store(y_ptr.add_offsets(offsets), y, Some(in_bounds), &[], None, None);
}

impl_float_unary_runtime_op_no_bwd!(ElemwiseCeilForward);

// ── Floor (D: Float) ──────────────────────────────────────────────────────────

/// Forward: y = floor(x)
#[kernel]
pub fn elemwise_floor_forward<T: Triton, D: Float, const BLOCK_SIZE: i32>(
    x_ptr: T::Pointer<D>,
    y_ptr: T::Pointer<D>,
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
    let x = T::load(x_ptr.add_offsets(offsets), Some(in_bounds), None, &[], None, None, None, false);
    let y = T::floor(x);
    T::store(y_ptr.add_offsets(offsets), y, Some(in_bounds), &[], None, None);
}

impl_float_unary_runtime_op_no_bwd!(ElemwiseFloorForward);

// ── Sqrt (D: Float) ───────────────────────────────────────────────────────────

/// Forward: y = sqrt(x)
#[kernel]
pub fn elemwise_sqrt_forward<T: Triton, D: Float, const BLOCK_SIZE: i32>(
    x_ptr: T::Pointer<D>,
    y_ptr: T::Pointer<D>,
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
    let x = T::load(x_ptr.add_offsets(offsets), Some(in_bounds), None, &[], None, None, None, false);
    let y = T::sqrt(x);
    T::store(y_ptr.add_offsets(offsets), y, Some(in_bounds), &[], None, None);
}

/// Backward: dx = dy / (2 * sqrt(x))
#[kernel]
pub fn elemwise_sqrt_backward<T: Triton, D: Float, const BLOCK_SIZE: i32>(
    dy_ptr: T::Pointer<D>,
    x_ptr: T::Pointer<D>,
    dx_ptr: T::Pointer<D>,
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
    let dy = T::load(dy_ptr.add_offsets(offsets), Some(in_bounds), None, &[], None, None, None, false);
    let x  = T::load(x_ptr.add_offsets(offsets),  Some(in_bounds), None, &[], None, None, None, false);
    let two = T::cast::<f32, D>(T::full::<f32>(&[BLOCK_SIZE], 2.0_f32), None, false);
    let dx = dy / (two * T::sqrt(x));
    T::store(dx_ptr.add_offsets(offsets), dx, Some(in_bounds), &[], None, None);
}

impl_float_unary_runtime_op!(ElemwiseSqrtForward);

// ── Reciprocal (D: Float) ─────────────────────────────────────────────────────

/// Forward: y = 1 / x
#[kernel]
pub fn elemwise_reciprocal_forward<T: Triton, D: Float, const BLOCK_SIZE: i32>(
    x_ptr: T::Pointer<D>,
    y_ptr: T::Pointer<D>,
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
    let x = T::load(x_ptr.add_offsets(offsets), Some(in_bounds), None, &[], None, None, None, false);
    let one = T::cast::<f32, D>(T::full::<f32>(&[BLOCK_SIZE], 1.0_f32), None, false);
    let y = one / x;
    T::store(y_ptr.add_offsets(offsets), y, Some(in_bounds), &[], None, None);
}

/// Backward: dx = -dy / x^2
#[kernel]
pub fn elemwise_reciprocal_backward<T: Triton, D: Float, const BLOCK_SIZE: i32>(
    dy_ptr: T::Pointer<D>,
    x_ptr: T::Pointer<D>,
    dx_ptr: T::Pointer<D>,
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
    let dy = T::load(dy_ptr.add_offsets(offsets), Some(in_bounds), None, &[], None, None, None, false);
    let x  = T::load(x_ptr.add_offsets(offsets),  Some(in_bounds), None, &[], None, None, None, false);
    let dx = -(dy / (x * x));
    T::store(dx_ptr.add_offsets(offsets), dx, Some(in_bounds), &[], None, None);
}

impl_float_unary_runtime_op!(ElemwiseReciprocalForward);

// ── Exp (D: Float) ────────────────────────────────────────────────────────────

/// Forward: y = exp(x)
#[kernel]
pub fn elemwise_exp_forward<T: Triton, D: Float, const BLOCK_SIZE: i32>(
    x_ptr: T::Pointer<D>,
    y_ptr: T::Pointer<D>,
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
    let x = T::load(x_ptr.add_offsets(offsets), Some(in_bounds), None, &[], None, None, None, false);
    let y = T::exp(x);
    T::store(y_ptr.add_offsets(offsets), y, Some(in_bounds), &[], None, None);
}

/// Backward: dx = exp(x) * dy
#[kernel]
pub fn elemwise_exp_backward<T: Triton, D: Float, const BLOCK_SIZE: i32>(
    dy_ptr: T::Pointer<D>,
    x_ptr: T::Pointer<D>,
    dx_ptr: T::Pointer<D>,
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
    let dy = T::load(dy_ptr.add_offsets(offsets), Some(in_bounds), None, &[], None, None, None, false);
    let x  = T::load(x_ptr.add_offsets(offsets),  Some(in_bounds), None, &[], None, None, None, false);
    let dx = T::exp(x) * dy;
    T::store(dx_ptr.add_offsets(offsets), dx, Some(in_bounds), &[], None, None);
}

impl_float_unary_runtime_op!(ElemwiseExpForward);

// ── Log (D: Float) ────────────────────────────────────────────────────────────

/// Forward: y = log(x)
#[kernel]
pub fn elemwise_log_forward<T: Triton, D: Float, const BLOCK_SIZE: i32>(
    x_ptr: T::Pointer<D>,
    y_ptr: T::Pointer<D>,
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
    let x = T::load(x_ptr.add_offsets(offsets), Some(in_bounds), None, &[], None, None, None, false);
    let y = T::log(x);
    T::store(y_ptr.add_offsets(offsets), y, Some(in_bounds), &[], None, None);
}

/// Backward: dx = dy / x
#[kernel]
pub fn elemwise_log_backward<T: Triton, D: Float, const BLOCK_SIZE: i32>(
    dy_ptr: T::Pointer<D>,
    x_ptr: T::Pointer<D>,
    dx_ptr: T::Pointer<D>,
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
    let dy = T::load(dy_ptr.add_offsets(offsets), Some(in_bounds), None, &[], None, None, None, false);
    let x  = T::load(x_ptr.add_offsets(offsets),  Some(in_bounds), None, &[], None, None, None, false);
    let dx = dy / x;
    T::store(dx_ptr.add_offsets(offsets), dx, Some(in_bounds), &[], None, None);
}

impl_float_unary_runtime_op!(ElemwiseLogForward);

// ── Erf (D: Float) ────────────────────────────────────────────────────────────

/// Forward: y = erf(x)
#[kernel]
pub fn elemwise_erf_forward<T: Triton, D: Float, const BLOCK_SIZE: i32>(
    x_ptr: T::Pointer<D>,
    y_ptr: T::Pointer<D>,
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
    let x = T::load(x_ptr.add_offsets(offsets), Some(in_bounds), None, &[], None, None, None, false);
    let y = T::erf(x);
    T::store(y_ptr.add_offsets(offsets), y, Some(in_bounds), &[], None, None);
}

/// Backward: dx = 2/sqrt(pi) * exp(-x^2) * dy  where 2/sqrt(pi) ~= 1.1283791670955126
#[kernel]
pub fn elemwise_erf_backward<T: Triton, D: Float, const BLOCK_SIZE: i32>(
    dy_ptr: T::Pointer<D>,
    x_ptr: T::Pointer<D>,
    dx_ptr: T::Pointer<D>,
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
    let dy = T::load(dy_ptr.add_offsets(offsets), Some(in_bounds), None, &[], None, None, None, false);
    let x  = T::load(x_ptr.add_offsets(offsets),  Some(in_bounds), None, &[], None, None, None, false);
    // 2/sqrt(pi) = 1.1283791670955126
    let coeff = T::cast::<f32, D>(
        T::full::<f32>(&[BLOCK_SIZE], 1.1283791670955126_f32),
        None, false,
    );
    let dx = coeff * T::exp(-(x * x)) * dy;
    T::store(dx_ptr.add_offsets(offsets), dx, Some(in_bounds), &[], None, None);
}

impl_float_unary_runtime_op!(ElemwiseErfForward);

// ── Sin (D: Float) ────────────────────────────────────────────────────────────

/// Forward: y = sin(x)
#[kernel]
pub fn elemwise_sin_forward<T: Triton, D: Float, const BLOCK_SIZE: i32>(
    x_ptr: T::Pointer<D>,
    y_ptr: T::Pointer<D>,
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
    let x = T::load(x_ptr.add_offsets(offsets), Some(in_bounds), None, &[], None, None, None, false);
    let y = T::sin(x);
    T::store(y_ptr.add_offsets(offsets), y, Some(in_bounds), &[], None, None);
}

/// Backward: dx = cos(x) * dy
#[kernel]
pub fn elemwise_sin_backward<T: Triton, D: Float, const BLOCK_SIZE: i32>(
    dy_ptr: T::Pointer<D>,
    x_ptr: T::Pointer<D>,
    dx_ptr: T::Pointer<D>,
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
    let dy = T::load(dy_ptr.add_offsets(offsets), Some(in_bounds), None, &[], None, None, None, false);
    let x  = T::load(x_ptr.add_offsets(offsets),  Some(in_bounds), None, &[], None, None, None, false);
    let dx = T::cos(x) * dy;
    T::store(dx_ptr.add_offsets(offsets), dx, Some(in_bounds), &[], None, None);
}

impl_float_unary_runtime_op!(ElemwiseSinForward);

// ── Cos (D: Float) ────────────────────────────────────────────────────────────

/// Forward: y = cos(x)
#[kernel]
pub fn elemwise_cos_forward<T: Triton, D: Float, const BLOCK_SIZE: i32>(
    x_ptr: T::Pointer<D>,
    y_ptr: T::Pointer<D>,
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
    let x = T::load(x_ptr.add_offsets(offsets), Some(in_bounds), None, &[], None, None, None, false);
    let y = T::cos(x);
    T::store(y_ptr.add_offsets(offsets), y, Some(in_bounds), &[], None, None);
}

/// Backward: dx = -sin(x) * dy
#[kernel]
pub fn elemwise_cos_backward<T: Triton, D: Float, const BLOCK_SIZE: i32>(
    dy_ptr: T::Pointer<D>,
    x_ptr: T::Pointer<D>,
    dx_ptr: T::Pointer<D>,
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
    let dy = T::load(dy_ptr.add_offsets(offsets), Some(in_bounds), None, &[], None, None, None, false);
    let x  = T::load(x_ptr.add_offsets(offsets),  Some(in_bounds), None, &[], None, None, None, false);
    let dx = -(T::sin(x) * dy);
    T::store(dx_ptr.add_offsets(offsets), dx, Some(in_bounds), &[], None, None);
}

impl_float_unary_runtime_op!(ElemwiseCosForward);

// ── Tan (D: Float) ────────────────────────────────────────────────────────────

/// Forward: y = tan(x) = sin(x) / cos(x)
#[kernel]
pub fn elemwise_tan_forward<T: Triton, D: Float, const BLOCK_SIZE: i32>(
    x_ptr: T::Pointer<D>,
    y_ptr: T::Pointer<D>,
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
    let x = T::load(x_ptr.add_offsets(offsets), Some(in_bounds), None, &[], None, None, None, false);
    let y = T::sin(x) / T::cos(x);
    T::store(y_ptr.add_offsets(offsets), y, Some(in_bounds), &[], None, None);
}

/// Backward: dx = (1 + tan^2(x)) * dy
#[kernel]
pub fn elemwise_tan_backward<T: Triton, D: Float, const BLOCK_SIZE: i32>(
    dy_ptr: T::Pointer<D>,
    x_ptr: T::Pointer<D>,
    dx_ptr: T::Pointer<D>,
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
    let dy = T::load(dy_ptr.add_offsets(offsets), Some(in_bounds), None, &[], None, None, None, false);
    let x  = T::load(x_ptr.add_offsets(offsets),  Some(in_bounds), None, &[], None, None, None, false);
    let one = T::cast::<f32, D>(T::full::<f32>(&[BLOCK_SIZE], 1.0_f32), None, false);
    let tan = T::sin(x) / T::cos(x);
    let dx = (one + tan * tan) * dy;
    T::store(dx_ptr.add_offsets(offsets), dx, Some(in_bounds), &[], None, None);
}

impl_float_unary_runtime_op!(ElemwiseTanForward);

// ── Asin (D: Float) ───────────────────────────────────────────────────────────

/// Forward: y = asin(x) = atan(x / sqrt(1 - x^2))
#[kernel]
pub fn elemwise_asin_forward<T: Triton, D: Float, const BLOCK_SIZE: i32>(
    x_ptr: T::Pointer<D>,
    y_ptr: T::Pointer<D>,
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
    let x = T::load(x_ptr.add_offsets(offsets), Some(in_bounds), None, &[], None, None, None, false);
    let one = T::cast::<f32, D>(T::full::<f32>(&[BLOCK_SIZE], 1.0_f32), None, false);
    let y = T::atan(x / T::sqrt(one - x * x));
    T::store(y_ptr.add_offsets(offsets), y, Some(in_bounds), &[], None, None);
}

/// Backward: dx = dy / sqrt(1 - x^2)
#[kernel]
pub fn elemwise_asin_backward<T: Triton, D: Float, const BLOCK_SIZE: i32>(
    dy_ptr: T::Pointer<D>,
    x_ptr: T::Pointer<D>,
    dx_ptr: T::Pointer<D>,
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
    let dy = T::load(dy_ptr.add_offsets(offsets), Some(in_bounds), None, &[], None, None, None, false);
    let x  = T::load(x_ptr.add_offsets(offsets),  Some(in_bounds), None, &[], None, None, None, false);
    let one = T::cast::<f32, D>(T::full::<f32>(&[BLOCK_SIZE], 1.0_f32), None, false);
    let dx = dy / T::sqrt(one - x * x);
    T::store(dx_ptr.add_offsets(offsets), dx, Some(in_bounds), &[], None, None);
}

impl_float_unary_runtime_op!(ElemwiseAsinForward);

// ── Acos (D: Float) ───────────────────────────────────────────────────────────

/// Forward: y = acos(x) = pi/2 - atan(x / sqrt(1 - x^2))
#[kernel]
pub fn elemwise_acos_forward<T: Triton, D: Float, const BLOCK_SIZE: i32>(
    x_ptr: T::Pointer<D>,
    y_ptr: T::Pointer<D>,
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
    let x = T::load(x_ptr.add_offsets(offsets), Some(in_bounds), None, &[], None, None, None, false);
    let one     = T::cast::<f32, D>(T::full::<f32>(&[BLOCK_SIZE], 1.0_f32), None, false);
    let half_pi = T::cast::<f32, D>(
        T::full::<f32>(&[BLOCK_SIZE], 1.5707963267948966_f32), // π/2
        None, false,
    );
    let y = half_pi - T::atan(x / T::sqrt(one - x * x));
    T::store(y_ptr.add_offsets(offsets), y, Some(in_bounds), &[], None, None);
}

/// Backward: dx = -dy / sqrt(1 - x^2)
#[kernel]
pub fn elemwise_acos_backward<T: Triton, D: Float, const BLOCK_SIZE: i32>(
    dy_ptr: T::Pointer<D>,
    x_ptr: T::Pointer<D>,
    dx_ptr: T::Pointer<D>,
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
    let dy = T::load(dy_ptr.add_offsets(offsets), Some(in_bounds), None, &[], None, None, None, false);
    let x  = T::load(x_ptr.add_offsets(offsets),  Some(in_bounds), None, &[], None, None, None, false);
    let one = T::cast::<f32, D>(T::full::<f32>(&[BLOCK_SIZE], 1.0_f32), None, false);
    let dx = -(dy / T::sqrt(one - x * x));
    T::store(dx_ptr.add_offsets(offsets), dx, Some(in_bounds), &[], None, None);
}

impl_float_unary_runtime_op!(ElemwiseAcosForward);

// ── Atan (D: Float) ───────────────────────────────────────────────────────────

/// Forward: y = atan(x)
#[kernel]
pub fn elemwise_atan_forward<T: Triton, D: Float, const BLOCK_SIZE: i32>(
    x_ptr: T::Pointer<D>,
    y_ptr: T::Pointer<D>,
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
    let x = T::load(x_ptr.add_offsets(offsets), Some(in_bounds), None, &[], None, None, None, false);
    let y = T::atan(x);
    T::store(y_ptr.add_offsets(offsets), y, Some(in_bounds), &[], None, None);
}

/// Backward: dx = dy / (1 + x^2)
#[kernel]
pub fn elemwise_atan_backward<T: Triton, D: Float, const BLOCK_SIZE: i32>(
    dy_ptr: T::Pointer<D>,
    x_ptr: T::Pointer<D>,
    dx_ptr: T::Pointer<D>,
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
    let dy = T::load(dy_ptr.add_offsets(offsets), Some(in_bounds), None, &[], None, None, None, false);
    let x  = T::load(x_ptr.add_offsets(offsets),  Some(in_bounds), None, &[], None, None, None, false);
    let one = T::cast::<f32, D>(T::full::<f32>(&[BLOCK_SIZE], 1.0_f32), None, false);
    let dx = dy / (one + x * x);
    T::store(dx_ptr.add_offsets(offsets), dx, Some(in_bounds), &[], None, None);
}

impl_float_unary_runtime_op!(ElemwiseAtanForward);

// ── Sinh (D: Float) ───────────────────────────────────────────────────────────

/// Forward: y = sinh(x) = (exp(x) - exp(-x)) / 2
#[kernel]
pub fn elemwise_sinh_forward<T: Triton, D: Float, const BLOCK_SIZE: i32>(
    x_ptr: T::Pointer<D>,
    y_ptr: T::Pointer<D>,
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
    let x = T::load(x_ptr.add_offsets(offsets), Some(in_bounds), None, &[], None, None, None, false);
    let two = T::cast::<f32, D>(T::full::<f32>(&[BLOCK_SIZE], 2.0_f32), None, false);
    let y = (T::exp(x) - T::exp(-x)) / two;
    T::store(y_ptr.add_offsets(offsets), y, Some(in_bounds), &[], None, None);
}

/// Backward: dx = cosh(x) * dy
#[kernel]
pub fn elemwise_sinh_backward<T: Triton, D: Float, const BLOCK_SIZE: i32>(
    dy_ptr: T::Pointer<D>,
    x_ptr: T::Pointer<D>,
    dx_ptr: T::Pointer<D>,
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
    let dy = T::load(dy_ptr.add_offsets(offsets), Some(in_bounds), None, &[], None, None, None, false);
    let x  = T::load(x_ptr.add_offsets(offsets),  Some(in_bounds), None, &[], None, None, None, false);
    let two = T::cast::<f32, D>(T::full::<f32>(&[BLOCK_SIZE], 2.0_f32), None, false);
    let cosh_x = (T::exp(x) + T::exp(-x)) / two;
    let dx = cosh_x * dy;
    T::store(dx_ptr.add_offsets(offsets), dx, Some(in_bounds), &[], None, None);
}

impl_float_unary_runtime_op!(ElemwiseSinhForward);

// ── Cosh (D: Float) ───────────────────────────────────────────────────────────

/// Forward: y = cosh(x) = (exp(x) + exp(-x)) / 2
#[kernel]
pub fn elemwise_cosh_forward<T: Triton, D: Float, const BLOCK_SIZE: i32>(
    x_ptr: T::Pointer<D>,
    y_ptr: T::Pointer<D>,
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
    let x = T::load(x_ptr.add_offsets(offsets), Some(in_bounds), None, &[], None, None, None, false);
    let two = T::cast::<f32, D>(T::full::<f32>(&[BLOCK_SIZE], 2.0_f32), None, false);
    let y = (T::exp(x) + T::exp(-x)) / two;
    T::store(y_ptr.add_offsets(offsets), y, Some(in_bounds), &[], None, None);
}

/// Backward: dx = sinh(x) * dy
#[kernel]
pub fn elemwise_cosh_backward<T: Triton, D: Float, const BLOCK_SIZE: i32>(
    dy_ptr: T::Pointer<D>,
    x_ptr: T::Pointer<D>,
    dx_ptr: T::Pointer<D>,
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
    let dy = T::load(dy_ptr.add_offsets(offsets), Some(in_bounds), None, &[], None, None, None, false);
    let x  = T::load(x_ptr.add_offsets(offsets),  Some(in_bounds), None, &[], None, None, None, false);
    let two = T::cast::<f32, D>(T::full::<f32>(&[BLOCK_SIZE], 2.0_f32), None, false);
    let sinh_x = (T::exp(x) - T::exp(-x)) / two;
    let dx = sinh_x * dy;
    T::store(dx_ptr.add_offsets(offsets), dx, Some(in_bounds), &[], None, None);
}

impl_float_unary_runtime_op!(ElemwiseCoshForward);

// ── Asinh (D: Float) ──────────────────────────────────────────────────────────

/// Forward: y = asinh(x) = log(x + sqrt(x^2 + 1))
#[kernel]
pub fn elemwise_asinh_forward<T: Triton, D: Float, const BLOCK_SIZE: i32>(
    x_ptr: T::Pointer<D>,
    y_ptr: T::Pointer<D>,
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
    let x = T::load(x_ptr.add_offsets(offsets), Some(in_bounds), None, &[], None, None, None, false);
    let one = T::cast::<f32, D>(T::full::<f32>(&[BLOCK_SIZE], 1.0_f32), None, false);
    let y = T::log(x + T::sqrt(x * x + one));
    T::store(y_ptr.add_offsets(offsets), y, Some(in_bounds), &[], None, None);
}

/// Backward: dx = dy / sqrt(x^2 + 1)
#[kernel]
pub fn elemwise_asinh_backward<T: Triton, D: Float, const BLOCK_SIZE: i32>(
    dy_ptr: T::Pointer<D>,
    x_ptr: T::Pointer<D>,
    dx_ptr: T::Pointer<D>,
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
    let dy = T::load(dy_ptr.add_offsets(offsets), Some(in_bounds), None, &[], None, None, None, false);
    let x  = T::load(x_ptr.add_offsets(offsets),  Some(in_bounds), None, &[], None, None, None, false);
    let one = T::cast::<f32, D>(T::full::<f32>(&[BLOCK_SIZE], 1.0_f32), None, false);
    let dx = dy / T::sqrt(x * x + one);
    T::store(dx_ptr.add_offsets(offsets), dx, Some(in_bounds), &[], None, None);
}

impl_float_unary_runtime_op!(ElemwiseAsinhForward);

// ── Acosh (D: Float) ──────────────────────────────────────────────────────────

/// Forward: y = acosh(x) = log(x + sqrt(x^2 - 1))
#[kernel]
pub fn elemwise_acosh_forward<T: Triton, D: Float, const BLOCK_SIZE: i32>(
    x_ptr: T::Pointer<D>,
    y_ptr: T::Pointer<D>,
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
    let x = T::load(x_ptr.add_offsets(offsets), Some(in_bounds), None, &[], None, None, None, false);
    let one = T::cast::<f32, D>(T::full::<f32>(&[BLOCK_SIZE], 1.0_f32), None, false);
    let y = T::log(x + T::sqrt(x * x - one));
    T::store(y_ptr.add_offsets(offsets), y, Some(in_bounds), &[], None, None);
}

/// Backward: dx = dy / sqrt(x^2 - 1)
#[kernel]
pub fn elemwise_acosh_backward<T: Triton, D: Float, const BLOCK_SIZE: i32>(
    dy_ptr: T::Pointer<D>,
    x_ptr: T::Pointer<D>,
    dx_ptr: T::Pointer<D>,
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
    let dy = T::load(dy_ptr.add_offsets(offsets), Some(in_bounds), None, &[], None, None, None, false);
    let x  = T::load(x_ptr.add_offsets(offsets),  Some(in_bounds), None, &[], None, None, None, false);
    let one = T::cast::<f32, D>(T::full::<f32>(&[BLOCK_SIZE], 1.0_f32), None, false);
    let dx = dy / T::sqrt(x * x - one);
    T::store(dx_ptr.add_offsets(offsets), dx, Some(in_bounds), &[], None, None);
}

impl_float_unary_runtime_op!(ElemwiseAcoshForward);

// ── Atanh (D: Float) ──────────────────────────────────────────────────────────

/// Forward: y = atanh(x) = log((1+x)/(1-x)) / 2
#[kernel]
pub fn elemwise_atanh_forward<T: Triton, D: Float, const BLOCK_SIZE: i32>(
    x_ptr: T::Pointer<D>,
    y_ptr: T::Pointer<D>,
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
    let x = T::load(x_ptr.add_offsets(offsets), Some(in_bounds), None, &[], None, None, None, false);
    let one = T::cast::<f32, D>(T::full::<f32>(&[BLOCK_SIZE], 1.0_f32), None, false);
    let two = T::cast::<f32, D>(T::full::<f32>(&[BLOCK_SIZE], 2.0_f32), None, false);
    let y = T::log((one + x) / (one - x)) / two;
    T::store(y_ptr.add_offsets(offsets), y, Some(in_bounds), &[], None, None);
}

/// Backward: dx = dy / (1 - x^2)
#[kernel]
pub fn elemwise_atanh_backward<T: Triton, D: Float, const BLOCK_SIZE: i32>(
    dy_ptr: T::Pointer<D>,
    x_ptr: T::Pointer<D>,
    dx_ptr: T::Pointer<D>,
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
    let dy = T::load(dy_ptr.add_offsets(offsets), Some(in_bounds), None, &[], None, None, None, false);
    let x  = T::load(x_ptr.add_offsets(offsets),  Some(in_bounds), None, &[], None, None, None, false);
    let one = T::cast::<f32, D>(T::full::<f32>(&[BLOCK_SIZE], 1.0_f32), None, false);
    let dx = dy / (one - x * x);
    T::store(dx_ptr.add_offsets(offsets), dx, Some(in_bounds), &[], None, None);
}

impl_float_unary_runtime_op!(ElemwiseAtanhForward);
