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

use core::marker::PhantomData;
use teeny_core::dtype::Num;
use teeny_macros::{kernel, tiled_kernel};
use teeny_triton::triton::{
    types::{AddOffsets, Comparison},
    *,
};

#[tiled_kernel]
pub fn relu_forward<T: Triton, D: Num, const BLOCK_SIZE: i32>(
    #[tile(block = BLOCK_SIZE, extent = n_elements)] x: In<Tile<T, D>>,
    #[tile(block = BLOCK_SIZE, extent = n_elements)] y: Out<Tile<T, D>>,
    n_elements: i32,
) where
    T::I32Tensor: types::Tensor<i32, 1>,
    T::I32Tensor: Comparison<i32, BoolTensor = T::BoolTensor>,
    T::Pointer<D>: AddOffsets<i32, 1, T::I32Tensor, Output = T::Tensor<T::Pointer<D>>>,
{
    let relu = T::maximum(x.tensor, T::zeros_like(x.tensor));

    T::store(y.tensor, relu, x.mask, &[], None, None);
}

#[kernel]
pub fn relu_backward<T: Triton, D: Num, const BLOCK_SIZE: i32>(
    dy_ptr: In<T::Pointer<D>>,
    y_ptr: In<T::Pointer<D>>,
    dx_ptr: Out<T::Pointer<D>>,
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
    fn n_activation_inputs(&self) -> usize {
        1
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

    // relu_backward(dy_ptr, y_ptr, dx_ptr, n_elements)
    // dy_ptr = incoming gradient, y_ptr = forward output (activation), dx_ptr = outgoing gradient
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
        visitor.visit_ptr(output); // y_ptr (forward output as activation mask)
        visitor.visit_ptr(grad_inputs[0]); // dx_ptr
        visitor.visit_i32(n as i32); // n_elements
    }

    #[cfg(feature = "training")]
    fn backward_grid(&self, _input_shapes: &[&[usize]], output_shape: &[usize]) -> [u32; 3] {
        let n: usize = output_shape.iter().product();
        [n.div_ceil(self.block_size as usize) as u32, 1, 1]
    }
}

pub struct ReluOp<'a, T: Num> {
    pub forward: ReluForward<T>,
    pub backward: ReluBackward<T>,
    _marker: PhantomData<&'a ()>,
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `tile_spec()` is generated by `#[tiled_kernel]` from
    /// `relu_forward`'s own `#[tile(block = BLOCK_SIZE, extent =
    /// n_elements)]`-tagged `x`/`y`, so this asserts the attributes still
    /// say what the kernel means: one flat axis spanning every dim, shared
    /// between input and output, at whatever rank the graph node turns out
    /// to be.
    #[test]
    fn test_tile_spec_declares_one_flat_axis_shared_by_x_and_y() {
        for rank in 1..=4 {
            let spec = ReluForward::<f32>::tile_spec(rank);
            assert_eq!(spec.loop_spec, None);
            assert_eq!((spec.inputs.len(), spec.outputs.len()), (1, 1));
            assert_eq!((spec.inputs[0].param, spec.outputs[0].param), ("x", "y"));

            for tensor in [spec.inputs[0], spec.outputs[0]] {
                assert_eq!(tensor.rank, rank);
                assert_eq!(tensor.axes.len(), 1, "one flattened axis");
                assert_eq!(tensor.axes[0].dims, (0..rank).collect::<Vec<_>>());
                assert_eq!(tensor.axes[0].block_const, "BLOCK_SIZE");
                assert_eq!(tensor.axes[0].extent_param, "n_elements");
                assert_eq!(tensor.axes[0].window, None);
                assert_eq!(tensor.axes[0].divide_by, None);
                assert_eq!(tensor.reduction_axis, None);
            }
            spec.validate()
                .expect("a derived spec must be self-consistent");
        }
    }
}
