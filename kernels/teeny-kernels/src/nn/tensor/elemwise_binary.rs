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

use teeny_core::dtype::{Float, Num};
use teeny_macros::{kernel, tiled_kernel};
use teeny_triton::triton::{
    types::{AddOffsets, Comparison},
    *,
};

// ── Helper macros for 2-input RuntimeOp ──────────────────────────────────────

macro_rules! impl_binary_num_runtime_op_with_bwd {
    ($Fwd:ident) => {
        impl<D: Num + Send + Sync + 'static> teeny_core::model::RuntimeOp for $Fwd<D> {
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
                visitor.visit_ptr(inputs[0].0); // a_ptr
                visitor.visit_ptr(inputs[1].0); // b_ptr
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
                visitor.visit_ptr(inputs[1].0);
                visitor.visit_ptr(grad_inputs[0]);
                visitor.visit_ptr(grad_inputs[1]);
                visitor.visit_i32(n as i32);
            }
            #[cfg(feature = "training")]
            fn backward_grid(&self, _: &[&[usize]], output_shape: &[usize]) -> [u32; 3] {
                let n: usize = output_shape.iter().product();
                [n.div_ceil(self.block_size as usize) as u32, 1, 1]
            }
        }
    };
}

macro_rules! impl_binary_float_runtime_op_with_bwd {
    ($Fwd:ident) => {
        impl<D: Float + Send + Sync + 'static> teeny_core::model::RuntimeOp for $Fwd<D> {
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
                visitor.visit_ptr(inputs[0].0);
                visitor.visit_ptr(inputs[1].0);
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
                visitor.visit_ptr(inputs[1].0);
                visitor.visit_ptr(grad_inputs[0]);
                visitor.visit_ptr(grad_inputs[1]);
                visitor.visit_i32(n as i32);
            }
            #[cfg(feature = "training")]
            fn backward_grid(&self, _: &[&[usize]], output_shape: &[usize]) -> [u32; 3] {
                let n: usize = output_shape.iter().product();
                [n.div_ceil(self.block_size as usize) as u32, 1, 1]
            }
        }
    };
}

/// Binary num op with no backward (comparison / logical ops)
macro_rules! impl_binary_num_runtime_op_no_bwd {
    ($Fwd:ident) => {
        impl<D: Num + Send + Sync + 'static> teeny_core::model::RuntimeOp for $Fwd<D> {
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
                visitor.visit_ptr(inputs[0].0);
                visitor.visit_ptr(inputs[1].0);
                visitor.visit_ptr(output);
                visitor.visit_i32(n as i32);
            }
            fn grid(&self, output_shape: &[usize]) -> [u32; 3] {
                let n: usize = output_shape.iter().product();
                [n.div_ceil(self.block_size as usize) as u32, 1, 1]
            }
        }
    };
}

// ── Mul ───────────────────────────────────────────────────────────────────────

/// Forward: out = a * b
#[tiled_kernel]
pub fn elemwise_mul_forward<T: Triton, D: Num, const BLOCK_SIZE: i32>(
    #[tile(block = BLOCK_SIZE, extent = n_elements)] a: In<Tile<T, D>>,
    #[tile(block = BLOCK_SIZE, extent = n_elements)] b: In<Tile<T, D>>,
    #[tile(block = BLOCK_SIZE, extent = n_elements)] out: Out<Tile<T, D>>,
    n_elements: i32,
) where
    T::I32Tensor: types::Tensor<i32, 1>,
    T::I32Tensor: Comparison<i32, BoolTensor = T::BoolTensor>,
    T::Pointer<D>: AddOffsets<i32, 1, T::I32Tensor, Output = T::Tensor<T::Pointer<D>>>,
{
    T::store(out.tensor, a.tensor * b.tensor, a.mask, &[], None, None);
}

/// Backward: da = dy * b,  db = dy * a
#[kernel]
pub fn elemwise_mul_backward<T: Triton, D: Num, const BLOCK_SIZE: i32>(
    dy_ptr: In<T::Pointer<D>>,
    a_ptr: In<T::Pointer<D>>,
    b_ptr: In<T::Pointer<D>>,
    da_ptr: Out<T::Pointer<D>>,
    db_ptr: Out<T::Pointer<D>>,
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
    let a = T::load(
        a_ptr.add_offsets(offsets),
        Some(in_bounds),
        None,
        &[],
        None,
        None,
        None,
        false,
    );
    let b = T::load(
        b_ptr.add_offsets(offsets),
        Some(in_bounds),
        None,
        &[],
        None,
        None,
        None,
        false,
    );
    T::store(
        da_ptr.add_offsets(offsets),
        dy * b,
        Some(in_bounds),
        &[],
        None,
        None,
    );
    T::store(
        db_ptr.add_offsets(offsets),
        dy * a,
        Some(in_bounds),
        &[],
        None,
        None,
    );
}

impl_binary_num_runtime_op_with_bwd!(ElemwiseMulForward);

// ── Sub ───────────────────────────────────────────────────────────────────────

/// Forward: out = a - b
#[tiled_kernel]
pub fn elemwise_sub_forward<T: Triton, D: Num, const BLOCK_SIZE: i32>(
    #[tile(block = BLOCK_SIZE, extent = n_elements)] a: In<Tile<T, D>>,
    #[tile(block = BLOCK_SIZE, extent = n_elements)] b: In<Tile<T, D>>,
    #[tile(block = BLOCK_SIZE, extent = n_elements)] out: Out<Tile<T, D>>,
    n_elements: i32,
) where
    T::I32Tensor: types::Tensor<i32, 1>,
    T::I32Tensor: Comparison<i32, BoolTensor = T::BoolTensor>,
    T::Pointer<D>: AddOffsets<i32, 1, T::I32Tensor, Output = T::Tensor<T::Pointer<D>>>,
{
    T::store(out.tensor, a.tensor - b.tensor, a.mask, &[], None, None);
}

/// Backward: da = dy,  db = -dy
#[kernel]
pub fn elemwise_sub_backward<T: Triton, D: Num, const BLOCK_SIZE: i32>(
    dy_ptr: In<T::Pointer<D>>,
    da_ptr: Out<T::Pointer<D>>,
    db_ptr: Out<T::Pointer<D>>,
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
    T::store(
        da_ptr.add_offsets(offsets),
        dy,
        Some(in_bounds),
        &[],
        None,
        None,
    );
    T::store(
        db_ptr.add_offsets(offsets),
        -dy,
        Some(in_bounds),
        &[],
        None,
        None,
    );
}

impl<D: Num + Send + Sync + 'static> teeny_core::model::RuntimeOp for ElemwiseSubForward<D> {
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
        visitor.visit_ptr(inputs[0].0);
        visitor.visit_ptr(inputs[1].0);
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
        visitor.visit_ptr(grad_inputs[1]);
        visitor.visit_i32(n as i32);
    }
    #[cfg(feature = "training")]
    fn backward_grid(&self, _: &[&[usize]], output_shape: &[usize]) -> [u32; 3] {
        let n: usize = output_shape.iter().product();
        [n.div_ceil(self.block_size as usize) as u32, 1, 1]
    }
}

// ── Div ───────────────────────────────────────────────────────────────────────

/// Forward: out = a / b
#[tiled_kernel]
pub fn elemwise_div_forward<T: Triton, D: Float, const BLOCK_SIZE: i32>(
    #[tile(block = BLOCK_SIZE, extent = n_elements)] a: In<Tile<T, D>>,
    #[tile(block = BLOCK_SIZE, extent = n_elements)] b: In<Tile<T, D>>,
    #[tile(block = BLOCK_SIZE, extent = n_elements)] out: Out<Tile<T, D>>,
    n_elements: i32,
) where
    T::I32Tensor: types::Tensor<i32, 1>,
    T::I32Tensor: Comparison<i32, BoolTensor = T::BoolTensor>,
    T::Pointer<D>: AddOffsets<i32, 1, T::I32Tensor, Output = T::Tensor<T::Pointer<D>>>,
{
    T::store(out.tensor, a.tensor / b.tensor, a.mask, &[], None, None);
}

/// Backward: da = dy / b,  db = -a * dy / b^2
#[kernel]
pub fn elemwise_div_backward<T: Triton, D: Float, const BLOCK_SIZE: i32>(
    dy_ptr: In<T::Pointer<D>>,
    a_ptr: In<T::Pointer<D>>,
    b_ptr: In<T::Pointer<D>>,
    da_ptr: Out<T::Pointer<D>>,
    db_ptr: Out<T::Pointer<D>>,
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
    let a = T::load(
        a_ptr.add_offsets(offsets),
        Some(in_bounds),
        None,
        &[],
        None,
        None,
        None,
        false,
    );
    let b = T::load(
        b_ptr.add_offsets(offsets),
        Some(in_bounds),
        None,
        &[],
        None,
        None,
        None,
        false,
    );
    T::store(
        da_ptr.add_offsets(offsets),
        dy / b,
        Some(in_bounds),
        &[],
        None,
        None,
    );
    T::store(
        db_ptr.add_offsets(offsets),
        -(a * dy / (b * b)),
        Some(in_bounds),
        &[],
        None,
        None,
    );
}

impl_binary_float_runtime_op_with_bwd!(ElemwiseDivForward);

// ── Pow (D: Float) ────────────────────────────────────────────────────────────

/// Forward: out = a ^ b = exp(b * log(a))
#[tiled_kernel]
pub fn elemwise_pow_forward<T: Triton, D: Float, const BLOCK_SIZE: i32>(
    #[tile(block = BLOCK_SIZE, extent = n_elements)] a: In<Tile<T, D>>,
    #[tile(block = BLOCK_SIZE, extent = n_elements)] b: In<Tile<T, D>>,
    #[tile(block = BLOCK_SIZE, extent = n_elements)] out: Out<Tile<T, D>>,
    n_elements: i32,
) where
    T::I32Tensor: types::Tensor<i32, 1>,
    T::I32Tensor: Comparison<i32, BoolTensor = T::BoolTensor>,
    T::Pointer<D>: AddOffsets<i32, 1, T::I32Tensor, Output = T::Tensor<T::Pointer<D>>>,
{
    let y = T::exp(b.tensor * T::log(a.tensor));
    T::store(out.tensor, y, a.mask, &[], None, None);
}

/// Backward: da = b * a^(b-1) * dy,  db = log(a) * a^b * dy
#[kernel]
pub fn elemwise_pow_backward<T: Triton, D: Float, const BLOCK_SIZE: i32>(
    dy_ptr: In<T::Pointer<D>>,
    a_ptr: In<T::Pointer<D>>,
    b_ptr: In<T::Pointer<D>>,
    da_ptr: Out<T::Pointer<D>>,
    db_ptr: Out<T::Pointer<D>>,
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
    let a = T::load(
        a_ptr.add_offsets(offsets),
        Some(in_bounds),
        None,
        &[],
        None,
        None,
        None,
        false,
    );
    let b = T::load(
        b_ptr.add_offsets(offsets),
        Some(in_bounds),
        None,
        &[],
        None,
        None,
        None,
        false,
    );
    let a_pow_b = T::exp(b * T::log(a)); // a^b
    // a^(b-1) = a^b / a  (avoids generic constant D::ONE)
    let a_pow_bm1 = a_pow_b / a;
    T::store(
        da_ptr.add_offsets(offsets),
        b * a_pow_bm1 * dy,
        Some(in_bounds),
        &[],
        None,
        None,
    );
    T::store(
        db_ptr.add_offsets(offsets),
        T::log(a) * a_pow_b * dy,
        Some(in_bounds),
        &[],
        None,
        None,
    );
}

impl_binary_float_runtime_op_with_bwd!(ElemwisePowForward);

// ── Mod ───────────────────────────────────────────────────────────────────────

/// Forward fmod: out = a - trunc(a/b)*b  (C-style float remainder)
#[tiled_kernel]
pub fn elemwise_fmod_forward<T: Triton, D: Float, const BLOCK_SIZE: i32>(
    #[tile(block = BLOCK_SIZE, extent = n_elements)] a: In<Tile<T, D>>,
    #[tile(block = BLOCK_SIZE, extent = n_elements)] b: In<Tile<T, D>>,
    #[tile(block = BLOCK_SIZE, extent = n_elements)] out: Out<Tile<T, D>>,
    n_elements: i32,
) where
    T::I32Tensor: types::Tensor<i32, 1>,
    T::I32Tensor: Comparison<i32, BoolTensor = T::BoolTensor>,
    T::Pointer<D>: AddOffsets<i32, 1, T::I32Tensor, Output = T::Tensor<T::Pointer<D>>>,
{
    // fmod: a.tensor - floor(a.tensor/b.tensor)*b.tensor  (use floor here; for true C fmod we'd need trunc)
    // Using floor makes this the Python-style modulo which is more broadly useful.
    let y = a.tensor - T::floor(a.tensor / b.tensor) * b.tensor;
    T::store(out.tensor, y, a.mask, &[], None, None);
}

impl<D: Float + Send + Sync + 'static> teeny_core::model::RuntimeOp for ElemwiseFmodForward<D> {
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
        visitor.visit_ptr(inputs[0].0);
        visitor.visit_ptr(inputs[1].0);
        visitor.visit_ptr(output);
        visitor.visit_i32(n as i32);
    }
    fn grid(&self, output_shape: &[usize]) -> [u32; 3] {
        let n: usize = output_shape.iter().product();
        [n.div_ceil(self.block_size as usize) as u32, 1, 1]
    }
}

// ── ElemMin / ElemMax (D: Num) ────────────────────────────────────────────────

/// Forward: out = min(a, b)
#[tiled_kernel]
pub fn elemwise_min_forward<T: Triton, D: Num, const BLOCK_SIZE: i32>(
    #[tile(block = BLOCK_SIZE, extent = n_elements)] a: In<Tile<T, D>>,
    #[tile(block = BLOCK_SIZE, extent = n_elements)] b: In<Tile<T, D>>,
    #[tile(block = BLOCK_SIZE, extent = n_elements)] out: Out<Tile<T, D>>,
    n_elements: i32,
) where
    T::I32Tensor: types::Tensor<i32, 1>,
    T::I32Tensor: Comparison<i32, BoolTensor = T::BoolTensor>,
    T::Pointer<D>: AddOffsets<i32, 1, T::I32Tensor, Output = T::Tensor<T::Pointer<D>>>,
{
    T::store(
        out.tensor,
        T::minimum(a.tensor, b.tensor),
        a.mask,
        &[],
        None,
        None,
    );
}

/// Backward: pass dy to the input that was smaller, 0 to the other.
#[kernel]
pub fn elemwise_min_backward<T: Triton, D: Num, const BLOCK_SIZE: i32>(
    dy_ptr: In<T::Pointer<D>>,
    a_ptr: In<T::Pointer<D>>,
    b_ptr: In<T::Pointer<D>>,
    da_ptr: Out<T::Pointer<D>>,
    db_ptr: Out<T::Pointer<D>>,
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
    let a = T::load(
        a_ptr.add_offsets(offsets),
        Some(in_bounds),
        None,
        &[],
        None,
        None,
        None,
        false,
    );
    let b = T::load(
        b_ptr.add_offsets(offsets),
        Some(in_bounds),
        None,
        &[],
        None,
        None,
        None,
        false,
    );
    let z = T::zeros_like(dy);
    let a_is_min = T::le(a, b);
    T::store(
        da_ptr.add_offsets(offsets),
        T::where_(a_is_min, dy, z),
        Some(in_bounds),
        &[],
        None,
        None,
    );
    T::store(
        db_ptr.add_offsets(offsets),
        T::where_(a_is_min, z, dy),
        Some(in_bounds),
        &[],
        None,
        None,
    );
}

impl_binary_num_runtime_op_with_bwd!(ElemwiseMinForward);

/// Forward: out = max(a, b)
#[tiled_kernel]
pub fn elemwise_max_forward<T: Triton, D: Num, const BLOCK_SIZE: i32>(
    #[tile(block = BLOCK_SIZE, extent = n_elements)] a: In<Tile<T, D>>,
    #[tile(block = BLOCK_SIZE, extent = n_elements)] b: In<Tile<T, D>>,
    #[tile(block = BLOCK_SIZE, extent = n_elements)] out: Out<Tile<T, D>>,
    n_elements: i32,
) where
    T::I32Tensor: types::Tensor<i32, 1>,
    T::I32Tensor: Comparison<i32, BoolTensor = T::BoolTensor>,
    T::Pointer<D>: AddOffsets<i32, 1, T::I32Tensor, Output = T::Tensor<T::Pointer<D>>>,
{
    T::store(
        out.tensor,
        T::maximum(a.tensor, b.tensor),
        a.mask,
        &[],
        None,
        None,
    );
}

/// Backward: pass dy to the input that was larger, 0 to the other.
#[kernel]
pub fn elemwise_max_backward<T: Triton, D: Num, const BLOCK_SIZE: i32>(
    dy_ptr: In<T::Pointer<D>>,
    a_ptr: In<T::Pointer<D>>,
    b_ptr: In<T::Pointer<D>>,
    da_ptr: Out<T::Pointer<D>>,
    db_ptr: Out<T::Pointer<D>>,
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
    let a = T::load(
        a_ptr.add_offsets(offsets),
        Some(in_bounds),
        None,
        &[],
        None,
        None,
        None,
        false,
    );
    let b = T::load(
        b_ptr.add_offsets(offsets),
        Some(in_bounds),
        None,
        &[],
        None,
        None,
        None,
        false,
    );
    let z = T::zeros_like(dy);
    let a_is_max = T::ge(a, b);
    T::store(
        da_ptr.add_offsets(offsets),
        T::where_(a_is_max, dy, z),
        Some(in_bounds),
        &[],
        None,
        None,
    );
    T::store(
        db_ptr.add_offsets(offsets),
        T::where_(a_is_max, z, dy),
        Some(in_bounds),
        &[],
        None,
        None,
    );
}

impl_binary_num_runtime_op_with_bwd!(ElemwiseMaxForward);

// ── ElemMean ──────────────────────────────────────────────────────────────────

/// Forward: out = (a + b) / 2
#[tiled_kernel]
pub fn elemwise_mean_forward<T: Triton, D: Float, const BLOCK_SIZE: i32>(
    #[tile(block = BLOCK_SIZE, extent = n_elements)] a: In<Tile<T, D>>,
    #[tile(block = BLOCK_SIZE, extent = n_elements)] b: In<Tile<T, D>>,
    #[tile(block = BLOCK_SIZE, extent = n_elements)] out: Out<Tile<T, D>>,
    n_elements: i32,
) where
    T::I32Tensor: types::Tensor<i32, 1>,
    T::I32Tensor: Comparison<i32, BoolTensor = T::BoolTensor>,
    T::Pointer<D>: AddOffsets<i32, 1, T::I32Tensor, Output = T::Tensor<T::Pointer<D>>>,
{
    let two = T::cast::<f32, D>(T::full::<f32>(&[BLOCK_SIZE], 2.0_f32), None, false);
    T::store(
        out.tensor,
        (a.tensor + b.tensor) / two,
        a.mask,
        &[],
        None,
        None,
    );
}

/// Backward: da = db = dy / 2
#[kernel]
pub fn elemwise_mean_backward<T: Triton, D: Float, const BLOCK_SIZE: i32>(
    dy_ptr: In<T::Pointer<D>>,
    da_ptr: Out<T::Pointer<D>>,
    db_ptr: Out<T::Pointer<D>>,
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
    let two = T::cast::<f32, D>(T::full::<f32>(&[BLOCK_SIZE], 2.0_f32), None, false);
    let half_dy = dy / two;
    T::store(
        da_ptr.add_offsets(offsets),
        half_dy,
        Some(in_bounds),
        &[],
        None,
        None,
    );
    T::store(
        db_ptr.add_offsets(offsets),
        half_dy,
        Some(in_bounds),
        &[],
        None,
        None,
    );
}

impl<D: Float + Send + Sync + 'static> teeny_core::model::RuntimeOp for ElemwiseMeanForward<D> {
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
        visitor.visit_ptr(inputs[0].0);
        visitor.visit_ptr(inputs[1].0);
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
        visitor.visit_ptr(grad_inputs[1]);
        visitor.visit_i32(n as i32);
    }
    #[cfg(feature = "training")]
    fn backward_grid(&self, _: &[&[usize]], output_shape: &[usize]) -> [u32; 3] {
        let n: usize = output_shape.iter().product();
        [n.div_ceil(self.block_size as usize) as u32, 1, 1]
    }
}

// ── ElemSum (binary add — identical to ElemwiseAdd semantics) ─────────────────

/// Forward: out = a + b  (binary ElemSum)
#[tiled_kernel]
pub fn elemwise_sum_forward<T: Triton, D: Num, const BLOCK_SIZE: i32>(
    #[tile(block = BLOCK_SIZE, extent = n_elements)] a: In<Tile<T, D>>,
    #[tile(block = BLOCK_SIZE, extent = n_elements)] b: In<Tile<T, D>>,
    #[tile(block = BLOCK_SIZE, extent = n_elements)] out: Out<Tile<T, D>>,
    n_elements: i32,
) where
    T::I32Tensor: types::Tensor<i32, 1>,
    T::I32Tensor: Comparison<i32, BoolTensor = T::BoolTensor>,
    T::Pointer<D>: AddOffsets<i32, 1, T::I32Tensor, Output = T::Tensor<T::Pointer<D>>>,
{
    T::store(out.tensor, a.tensor + b.tensor, a.mask, &[], None, None);
}

/// Backward: da = db = dy
#[kernel]
pub fn elemwise_sum_backward<T: Triton, D: Num, const BLOCK_SIZE: i32>(
    dy_ptr: In<T::Pointer<D>>,
    da_ptr: Out<T::Pointer<D>>,
    db_ptr: Out<T::Pointer<D>>,
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
    T::store(
        da_ptr.add_offsets(offsets),
        dy,
        Some(in_bounds),
        &[],
        None,
        None,
    );
    T::store(
        db_ptr.add_offsets(offsets),
        dy,
        Some(in_bounds),
        &[],
        None,
        None,
    );
}

impl<D: Num + Send + Sync + 'static> teeny_core::model::RuntimeOp for ElemwiseSumForward<D> {
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
        visitor.visit_ptr(inputs[0].0);
        visitor.visit_ptr(inputs[1].0);
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
        visitor.visit_ptr(grad_inputs[1]);
        visitor.visit_i32(n as i32);
    }
    #[cfg(feature = "training")]
    fn backward_grid(&self, _: &[&[usize]], output_shape: &[usize]) -> [u32; 3] {
        let n: usize = output_shape.iter().product();
        [n.div_ceil(self.block_size as usize) as u32, 1, 1]
    }
}

// ── Comparison ops (output 0.0/1.0 as float) ──────────────────────────────────

/// Forward: out = 1.0 if a == b else 0.0
#[tiled_kernel]
pub fn elemwise_equal_forward<T: Triton, D: Num, const BLOCK_SIZE: i32>(
    #[tile(block = BLOCK_SIZE, extent = n_elements)] a: In<Tile<T, D>>,
    #[tile(block = BLOCK_SIZE, extent = n_elements)] b: In<Tile<T, D>>,
    #[tile(block = BLOCK_SIZE, extent = n_elements)] out: Out<Tile<T, D>>,
    n_elements: i32,
) where
    T::I32Tensor: types::Tensor<i32, 1>,
    T::I32Tensor: Comparison<i32, BoolTensor = T::BoolTensor>,
    T::Pointer<D>: AddOffsets<i32, 1, T::I32Tensor, Output = T::Tensor<T::Pointer<D>>>,
{
    let cond = T::eq(a.tensor, b.tensor);
    let one = T::cast::<i32, D>(T::full::<i32>(&[BLOCK_SIZE], 1), None, false);
    let zero = T::zeros_like(a.tensor);
    T::store(
        out.tensor,
        T::where_(cond, one, zero),
        a.mask,
        &[],
        None,
        None,
    );
}

impl_binary_num_runtime_op_no_bwd!(ElemwiseEqualForward);

/// Forward: out = 1.0 if a > b else 0.0
#[tiled_kernel]
pub fn elemwise_greater_forward<T: Triton, D: Num, const BLOCK_SIZE: i32>(
    #[tile(block = BLOCK_SIZE, extent = n_elements)] a: In<Tile<T, D>>,
    #[tile(block = BLOCK_SIZE, extent = n_elements)] b: In<Tile<T, D>>,
    #[tile(block = BLOCK_SIZE, extent = n_elements)] out: Out<Tile<T, D>>,
    n_elements: i32,
) where
    T::I32Tensor: types::Tensor<i32, 1>,
    T::I32Tensor: Comparison<i32, BoolTensor = T::BoolTensor>,
    T::Pointer<D>: AddOffsets<i32, 1, T::I32Tensor, Output = T::Tensor<T::Pointer<D>>>,
{
    let cond = T::gt(a.tensor, b.tensor);
    let one = T::cast::<i32, D>(T::full::<i32>(&[BLOCK_SIZE], 1), None, false);
    let zero = T::zeros_like(a.tensor);
    T::store(
        out.tensor,
        T::where_(cond, one, zero),
        a.mask,
        &[],
        None,
        None,
    );
}

impl_binary_num_runtime_op_no_bwd!(ElemwiseGreaterForward);

/// Forward: out = 1.0 if a >= b else 0.0
#[tiled_kernel]
pub fn elemwise_greater_equal_forward<T: Triton, D: Num, const BLOCK_SIZE: i32>(
    #[tile(block = BLOCK_SIZE, extent = n_elements)] a: In<Tile<T, D>>,
    #[tile(block = BLOCK_SIZE, extent = n_elements)] b: In<Tile<T, D>>,
    #[tile(block = BLOCK_SIZE, extent = n_elements)] out: Out<Tile<T, D>>,
    n_elements: i32,
) where
    T::I32Tensor: types::Tensor<i32, 1>,
    T::I32Tensor: Comparison<i32, BoolTensor = T::BoolTensor>,
    T::Pointer<D>: AddOffsets<i32, 1, T::I32Tensor, Output = T::Tensor<T::Pointer<D>>>,
{
    let cond = T::ge(a.tensor, b.tensor);
    let one = T::cast::<i32, D>(T::full::<i32>(&[BLOCK_SIZE], 1), None, false);
    let zero = T::zeros_like(a.tensor);
    T::store(
        out.tensor,
        T::where_(cond, one, zero),
        a.mask,
        &[],
        None,
        None,
    );
}

impl_binary_num_runtime_op_no_bwd!(ElemwiseGreaterEqualForward);

/// Forward: out = 1.0 if a < b else 0.0
#[tiled_kernel]
pub fn elemwise_less_forward<T: Triton, D: Num, const BLOCK_SIZE: i32>(
    #[tile(block = BLOCK_SIZE, extent = n_elements)] a: In<Tile<T, D>>,
    #[tile(block = BLOCK_SIZE, extent = n_elements)] b: In<Tile<T, D>>,
    #[tile(block = BLOCK_SIZE, extent = n_elements)] out: Out<Tile<T, D>>,
    n_elements: i32,
) where
    T::I32Tensor: types::Tensor<i32, 1>,
    T::I32Tensor: Comparison<i32, BoolTensor = T::BoolTensor>,
    T::Pointer<D>: AddOffsets<i32, 1, T::I32Tensor, Output = T::Tensor<T::Pointer<D>>>,
{
    let cond = T::lt(a.tensor, b.tensor);
    let one = T::cast::<i32, D>(T::full::<i32>(&[BLOCK_SIZE], 1), None, false);
    let zero = T::zeros_like(a.tensor);
    T::store(
        out.tensor,
        T::where_(cond, one, zero),
        a.mask,
        &[],
        None,
        None,
    );
}

impl_binary_num_runtime_op_no_bwd!(ElemwiseLessForward);

/// Forward: out = 1.0 if a <= b else 0.0
#[tiled_kernel]
pub fn elemwise_less_equal_forward<T: Triton, D: Num, const BLOCK_SIZE: i32>(
    #[tile(block = BLOCK_SIZE, extent = n_elements)] a: In<Tile<T, D>>,
    #[tile(block = BLOCK_SIZE, extent = n_elements)] b: In<Tile<T, D>>,
    #[tile(block = BLOCK_SIZE, extent = n_elements)] out: Out<Tile<T, D>>,
    n_elements: i32,
) where
    T::I32Tensor: types::Tensor<i32, 1>,
    T::I32Tensor: Comparison<i32, BoolTensor = T::BoolTensor>,
    T::Pointer<D>: AddOffsets<i32, 1, T::I32Tensor, Output = T::Tensor<T::Pointer<D>>>,
{
    let cond = T::le(a.tensor, b.tensor);
    let one = T::cast::<i32, D>(T::full::<i32>(&[BLOCK_SIZE], 1), None, false);
    let zero = T::zeros_like(a.tensor);
    T::store(
        out.tensor,
        T::where_(cond, one, zero),
        a.mask,
        &[],
        None,
        None,
    );
}

impl_binary_num_runtime_op_no_bwd!(ElemwiseLessEqualForward);

// ── Where (3-input: condition, x, y) ─────────────────────────────────────────
//
// Condition is stored as same D type: 0 = false, non-zero = true.

/// Forward: out = x where cond != 0 else y
#[tiled_kernel]
pub fn elemwise_where_forward<T: Triton, D: Float, const BLOCK_SIZE: i32>(
    #[tile(block = BLOCK_SIZE, extent = n_elements)] cond: In<Tile<T, D>>,
    #[tile(block = BLOCK_SIZE, extent = n_elements)] x: In<Tile<T, D>>,
    #[tile(block = BLOCK_SIZE, extent = n_elements)] y: In<Tile<T, D>>,
    #[tile(block = BLOCK_SIZE, extent = n_elements)] out: Out<Tile<T, D>>,
    n_elements: i32,
) where
    T::I32Tensor: types::Tensor<i32, 1>,
    T::I32Tensor: Comparison<i32, BoolTensor = T::BoolTensor>,
    T::Pointer<D>: AddOffsets<i32, 1, T::I32Tensor, Output = T::Tensor<T::Pointer<D>>>,
{
    let zero = T::zeros_like(cond.tensor);
    let bool_cond = T::ne(cond.tensor, zero);
    T::store(
        out.tensor,
        T::where_(bool_cond, x.tensor, y.tensor),
        cond.mask,
        &[],
        None,
        None,
    );
}

/// Backward: dx = where(cond, dy, 0),  dy_in = where(cond, 0, dy)
#[kernel]
pub fn elemwise_where_backward<T: Triton, D: Float, const BLOCK_SIZE: i32>(
    dy_ptr: In<T::Pointer<D>>,
    cond_ptr: In<T::Pointer<D>>,
    dx_ptr: Out<T::Pointer<D>>,
    dy_in_ptr: Out<T::Pointer<D>>,
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
    let cond = T::load(
        cond_ptr.add_offsets(offsets),
        Some(in_bounds),
        None,
        &[],
        None,
        None,
        None,
        false,
    );
    let zero = T::zeros_like(dy);
    let bool_cond = T::ne(cond, T::zeros_like(cond));
    T::store(
        dx_ptr.add_offsets(offsets),
        T::where_(bool_cond, dy, zero),
        Some(in_bounds),
        &[],
        None,
        None,
    );
    T::store(
        dy_in_ptr.add_offsets(offsets),
        T::where_(bool_cond, zero, dy),
        Some(in_bounds),
        &[],
        None,
        None,
    );
}

impl<D: Float + Send + Sync + 'static> teeny_core::model::RuntimeOp for ElemwiseWhereForward<D> {
    fn n_activation_inputs(&self) -> usize {
        3
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
        visitor.visit_ptr(inputs[0].0); // cond
        visitor.visit_ptr(inputs[1].0); // x
        visitor.visit_ptr(inputs[2].0); // y
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
        visitor.visit_ptr(inputs[0].0); // cond
        visitor.visit_ptr(grad_inputs[1]); // dx
        visitor.visit_ptr(grad_inputs[2]); // dy_in
        visitor.visit_i32(n as i32);
    }
    #[cfg(feature = "training")]
    fn backward_grid(&self, _: &[&[usize]], output_shape: &[usize]) -> [u32; 3] {
        let n: usize = output_shape.iter().product();
        [n.div_ceil(self.block_size as usize) as u32, 1, 1]
    }
}

// ── Clip (3-input: x, min_val, max_val) ───────────────────────────────────────
//
// For simplicity this kernel takes min and max as f32 scalar kernel params
// rather than tensors.  The lowering packs them as f32.

/// Forward: out = clamp(x, min_val, max_val)
#[tiled_kernel]
pub fn elemwise_clip_forward<T: Triton, D: Float, const BLOCK_SIZE: i32>(
    #[tile(block = BLOCK_SIZE, extent = n_elements)] x: In<Tile<T, D>>,
    #[tile(block = BLOCK_SIZE, extent = n_elements)] out: Out<Tile<T, D>>,
    n_elements: i32,
    min_val: f32,
    max_val: f32,
) where
    T::I32Tensor: types::Tensor<i32, 1>,
    T::I32Tensor: Comparison<i32, BoolTensor = T::BoolTensor>,
    T::Pointer<D>: AddOffsets<i32, 1, T::I32Tensor, Output = T::Tensor<T::Pointer<D>>>,
{
    let lo = T::cast::<f32, D>(T::full::<f32>(&[BLOCK_SIZE], min_val), None, false);
    let hi = T::cast::<f32, D>(T::full::<f32>(&[BLOCK_SIZE], max_val), None, false);
    let y = T::clamp(x.tensor, lo, hi);
    T::store(out.tensor, y, x.mask, &[], None, None);
}

/// Backward: pass dy through only where x was in [min_val, max_val]
#[kernel]
pub fn elemwise_clip_backward<T: Triton, D: Float, const BLOCK_SIZE: i32>(
    dy_ptr: In<T::Pointer<D>>,
    x_ptr: In<T::Pointer<D>>,
    dx_ptr: Out<T::Pointer<D>>,
    n_elements: i32,
    min_val: f32,
    max_val: f32,
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
    let lo = T::cast::<f32, D>(T::full::<f32>(&[BLOCK_SIZE], min_val), None, false);
    let hi = T::cast::<f32, D>(T::full::<f32>(&[BLOCK_SIZE], max_val), None, false);
    let in_range = T::ge(x, lo) & T::le(x, hi);
    let dx = T::where_(in_range, dy, T::zeros_like(dy));
    T::store(
        dx_ptr.add_offsets(offsets),
        dx,
        Some(in_bounds),
        &[],
        None,
        None,
    );
}

/// A RuntimeOp wrapper for Clip that stores the min/max scalar params alongside
/// the kernel struct (which only stores block_size).
pub struct ClipRuntimeOp<D: Float + Send + Sync + 'static> {
    pub kernel: ElemwiseClipForward<D>,
    pub backward_kernel: ElemwiseClipBackward<D>,
    pub min_val: f32,
    pub max_val: f32,
}

impl<D: Float + Send + Sync + 'static> ClipRuntimeOp<D> {
    pub fn new(block_size: i32, min_val: f32, max_val: f32) -> Self {
        Self {
            kernel: ElemwiseClipForward::<D>::new(block_size),
            backward_kernel: ElemwiseClipBackward::<D>::new(block_size),
            min_val,
            max_val,
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

impl<D: Float + Send + Sync + 'static> teeny_core::model::RuntimeOp for ClipRuntimeOp<D> {
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
        visitor.visit_f32(self.min_val);
        visitor.visit_f32(self.max_val);
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
        visitor.visit_f32(self.min_val);
        visitor.visit_f32(self.max_val);
    }
    #[cfg(feature = "training")]
    fn backward_grid(&self, _: &[&[usize]], output_shape: &[usize]) -> [u32; 3] {
        let n: usize = output_shape.iter().product();
        [n.div_ceil(self.kernel.block_size as usize) as u32, 1, 1]
    }
}

// ── Row-broadcast Sub / Div ([R, C] op [R, 1]) ────────────────────────────────
//
// The flat kernels above index `b` by the output's flat index, so they are
// only right when `b` has as many elements as `a`. These broadcast a `[R, 1]`
// `b` over the columns of a `[R, C]` `a`, as decomposed softmax and layernorm
// need (`x - max(x)`, `exp / sum(exp)`). Each program covers one row and a
// `BLOCK_C`-wide run of columns. Both axes are bound -- the row at the literal
// block 1 -- so the spec says each unit covers one row rather than every row.
//
// `b` declares the row axis alone, which is a broadcast operand: the prelude
// loads its one element and widens it across the tile. The graph edge for `b`
// is `[R, 1]`, rank 2, so the lowering inserts that trailing size-1 dim into
// this spec (`TensorTileSpec::unsqueeze`).

/// Forward: out[r, c] = a[r, c] - b[r]
#[tiled_kernel]
#[tile_axes(R(block = 1), C(block = BLOCK_C))]
pub fn elemwise_sub_row_broadcast_forward<T: Triton, D: Num, const BLOCK_C: i32>(
    #[tile(R, C)] a: In<Tile<T, D>>,
    #[tile(R)] b: In<Tile<T, D>>,
    #[tile(R, C)] out: Out<Tile<T, D>>,
    R: i32,
    C: i32,
) where
    T::I32Tensor: types::Tensor<i32, 1>,
    T::I32Tensor: Comparison<i32, BoolTensor = T::BoolTensor>,
    T::Pointer<D>: AddOffsets<i32, 1, T::I32Tensor, Output = T::Tensor<T::Pointer<D>>>,
{
    T::store(out.tensor, a.tensor - b.tensor, a.mask, &[], None, None);
}

/// Forward: out[r, c] = a[r, c] / b[r]
#[tiled_kernel]
#[tile_axes(R(block = 1), C(block = BLOCK_C))]
pub fn elemwise_div_row_broadcast_forward<T: Triton, D: Float, const BLOCK_C: i32>(
    #[tile(R, C)] a: In<Tile<T, D>>,
    #[tile(R)] b: In<Tile<T, D>>,
    #[tile(R, C)] out: Out<Tile<T, D>>,
    R: i32,
    C: i32,
) where
    T::I32Tensor: types::Tensor<i32, 1>,
    T::I32Tensor: Comparison<i32, BoolTensor = T::BoolTensor>,
    T::Pointer<D>: AddOffsets<i32, 1, T::I32Tensor, Output = T::Tensor<T::Pointer<D>>>,
{
    T::store(out.tensor, a.tensor / b.tensor, a.mask, &[], None, None);
}

/// `R` and `C` of a row broadcast: every dim but the last is a row.
fn rows_and_cols(output_shape: &[usize]) -> (usize, usize) {
    let (&c, rows) = output_shape.split_last().unwrap_or((&1, &[]));
    (rows.iter().product::<usize>(), c)
}

macro_rules! impl_row_broadcast_runtime_op {
    ($Op:ident, $Fwd:ident, $Bound:ident) => {
        /// RuntimeOp for a row-broadcast kernel: `a`, `b`, `out`, then `R`
        /// and `C`, on a `[R · cdiv(C, BLOCK_C)]` grid. The prelude decodes
        /// the program id innermost-first, so the column tile is the inner
        /// factor.
        pub struct $Op<D: $Bound + Send + Sync + 'static> {
            pub kernel: $Fwd<D>,
            block_c: i32,
        }

        impl<D: $Bound + Send + Sync + 'static> $Op<D> {
            pub fn new(block_c: i32) -> Self {
                Self {
                    kernel: $Fwd::<D>::new(block_c),
                    block_c,
                }
            }
        }

        impl<D: $Bound + Send + Sync + 'static> teeny_core::model::RuntimeOp for $Op<D> {
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
                let (r, c) = rows_and_cols(output_shape);
                visitor.visit_ptr(inputs[0].0); // a
                visitor.visit_ptr(inputs[1].0); // b
                visitor.visit_ptr(output);
                visitor.visit_i32(r as i32);
                visitor.visit_i32(c as i32);
            }
            fn grid(&self, output_shape: &[usize]) -> [u32; 3] {
                let (r, c) = rows_and_cols(output_shape);
                [(r * c.div_ceil(self.block_c as usize)) as u32, 1, 1]
            }
        }
    };
}

impl_row_broadcast_runtime_op!(
    SubRowBroadcastRuntimeOp,
    ElemwiseSubRowBroadcastForward,
    Num
);
impl_row_broadcast_runtime_op!(
    DivRowBroadcastRuntimeOp,
    ElemwiseDivRowBroadcastForward,
    Float
);

#[cfg(test)]
mod row_broadcast_tests {
    use super::*;

    /// The row axis is bound at 1 and the columns at `BLOCK_C`, and `b`
    /// declares the row alone: rank 1 until the lowering unsqueezes it.
    #[test]
    fn test_the_row_broadcast_spec_binds_rows_at_one_and_b_on_rows_alone() {
        let spec = ElemwiseSubRowBroadcastForward::<f32>::tile_spec();
        spec.validate()
            .expect("a derived spec must be self-consistent");

        let out = spec.outputs[0];
        assert_eq!(out.rank, 2);
        let blocks: Vec<(&[usize], &str)> =
            out.axes.iter().map(|a| (a.dims, a.block_const)).collect();
        assert_eq!(blocks, vec![(&[0][..], "1"), (&[1][..], "BLOCK_C")]);
        assert_eq!(spec.inputs.len(), 2);
        assert_eq!(spec.inputs[0].rank, 2);
        assert_eq!(spec.inputs[1].rank, 1);
        assert_eq!(spec.inputs[1].axes[0].extent_param, "R");
        assert_eq!(rows_and_cols(&[4, 8, 16]), (32, 16));
    }

    /// Both kernels compile with the real teenyc: a broadcast operand on a
    /// two-blocked-axis tile is a prelude path no other kernel takes.
    #[test]
    fn test_the_row_broadcast_kernels_compile() {
        dotenv::dotenv().ok();
        let target = teeny_runtime::reference_target();
        let sub = ElemwiseSubRowBroadcastForward::<f32>::new(1024);
        teeny_runtime::compile_kernel(&sub, &target, true, false)
            .expect("the row-broadcast sub kernel must compile");
        let div = ElemwiseDivRowBroadcastForward::<f32>::new(1024);
        teeny_runtime::compile_kernel(&div, &target, true, false)
            .expect("the row-broadcast div kernel must compile");
    }
}
