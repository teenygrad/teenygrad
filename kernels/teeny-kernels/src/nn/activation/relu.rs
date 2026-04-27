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

use core::marker::PhantomData;
use teeny_core::dtype::Num;
use teeny_macros::kernel;
use teeny_triton::triton::{
    types::{AddOffsets, Comparison},
    *,
};

#[kernel]
pub fn relu_forward<T: Triton, D: Num, const BLOCK_SIZE: i32>(
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
    let y = T::zeros_like(x);
    let relu = T::maximum(x, y);

    // Masked loads in Triton return 0 for masked-off lanes, which gives ReLU.
    T::store(
        y_ptr.add_offsets(offsets),
        relu,
        Some(in_bounds),
        &[],
        None,
        None,
    );
}

#[kernel]
pub fn relu_backward<T: Triton, D: Num, const BLOCK_SIZE: i32>(
    dy_ptr: T::Pointer<D>,
    y_ptr: T::Pointer<D>,
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

    let grad_y = T::load(
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

    // where(y > 0, grad_y, 0) compiles to a predicated select; avoids an fp mul.
    let zeros = T::zeros_like(grad_y);
    let y_gt_zero = T::gt(y, T::zeros_like(y));
    let grad_x = T::where_(y_gt_zero, grad_y, zeros);

    T::store(
        dx_ptr.add_offsets(offsets),
        grad_x,
        Some(in_bounds),
        &[],
        None,
        None,
    );
}

impl<D: Num + Send + Sync + 'static> teeny_core::model::RuntimeOp for ReluForward<D> {
    fn n_activation_inputs(&self) -> usize { 1 }

    fn param_shapes(&self, _input_shapes: &[&[usize]], _output_shape: &[usize]) -> Vec<Vec<usize>> {
        Vec::new()
    }

    fn pack_args(
        &self,
        inputs: &[(teeny_core::model::RawPtr, &[usize])],
        _params: &[teeny_core::model::RawPtr],
        output: teeny_core::model::RawPtr,
        output_shape: &[usize],
        visitor: &mut dyn teeny_core::device::program::ArgVisitor,
    ) {
        let n: usize = output_shape.iter().product();
        visitor.visit_ptr(inputs[0].0);
        visitor.visit_ptr(output);
        visitor.visit_i32(n as i32);
    }

    fn block(&self) -> [u32; 3] { [128, 1, 1] }

    fn grid(&self, output_shape: &[usize]) -> [u32; 3] {
        let n: usize = output_shape.iter().product();
        [n.div_ceil(self.block_size as usize) as u32, 1, 1]
    }
}

pub struct ReluOp<'a, T: Num> {
    pub forward: ReluForward<T>,
    pub backward: ReluBackward<T>,
    _marker: PhantomData<&'a ()>,
}
