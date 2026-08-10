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

use teeny_core::dtype::Num;
use teeny_macros::kernel;
use teeny_triton::triton::{
    types::{AddOffsets, Comparison},
    *,
};

// ── Forward: out[i] = a[i] + b[i] ────────────────────────────────────────────

#[kernel]
pub fn elemwise_add_forward<T: Triton, D: Num, const BLOCK_SIZE: i32>(
    a_ptr: T::Pointer<D>,
    b_ptr: T::Pointer<D>,
    out_ptr: T::Pointer<D>,
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
        out_ptr.add_offsets(offsets),
        a + b,
        Some(in_bounds),
        &[],
        None,
        None,
    );
}

// ── Backward: grad_a[i] = dy[i],  grad_b[i] = dy[i] ─────────────────────────
//
// Add is the fan-out of the gradient: the upstream gradient flows unchanged
// to both inputs.

#[kernel]
pub fn elemwise_add_backward<T: Triton, D: Num, const BLOCK_SIZE: i32>(
    dy_ptr: T::Pointer<D>,
    grad_a_ptr: T::Pointer<D>,
    grad_b_ptr: T::Pointer<D>,
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
        grad_a_ptr.add_offsets(offsets),
        dy,
        Some(in_bounds),
        &[],
        None,
        None,
    );
    T::store(
        grad_b_ptr.add_offsets(offsets),
        dy,
        Some(in_bounds),
        &[],
        None,
        None,
    );
}

// ── RuntimeOp ─────────────────────────────────────────────────────────────────

impl<D: Num + Send + Sync + 'static> teeny_core::model::RuntimeOp for ElemwiseAddForward<D> {
    fn n_activation_inputs(&self) -> usize {
        2
    }

    fn param_shapes(&self, _input_shapes: &[&[usize]], _output_shape: &[usize]) -> Vec<Vec<usize>> {
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
        visitor.visit_ptr(inputs[0].0); // a_ptr
        visitor.visit_ptr(inputs[1].0); // b_ptr
        visitor.visit_ptr(output); // out_ptr
        visitor.visit_i32(n as i32); // n_elements
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
        visitor.visit_ptr(grad_inputs[0]); // grad_a_ptr
        visitor.visit_ptr(grad_inputs[1]); // grad_b_ptr
        visitor.visit_i32(n as i32); // n_elements
    }

    #[cfg(feature = "training")]
    fn backward_grid(&self, _input_shapes: &[&[usize]], output_shape: &[usize]) -> [u32; 3] {
        let n: usize = output_shape.iter().product();
        [n.div_ceil(self.block_size as usize) as u32, 1, 1]
    }
}
