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
use teeny_macros::{kernel, tiled_kernel};
use teeny_triton::triton::{
    types::{AddOffsets, Comparison},
    *,
};

// ── SiLU (Swish) ─────────────────────────────────────────────────────────────

/// Forward: y = x * sigmoid(x)
#[tiled_kernel(backward = SiluBackward)]
pub fn silu_forward<T: Triton, D: Float, const BLOCK_SIZE: i32>(
    #[tile(block = BLOCK_SIZE, extent = n_elements)] x: In<Tile<T, D>>,
    #[tile(block = BLOCK_SIZE, extent = n_elements)] y: Out<Tile<T, D>>,
    n_elements: i32,
) where
    T::I32Tensor: types::Tensor<i32, 1>,
    T::I32Tensor: Comparison<i32, BoolTensor = T::BoolTensor>,
    T::Pointer<D>: AddOffsets<i32, 1, T::I32Tensor, Output = T::Tensor<T::Pointer<D>>>,
{
    let one = T::full(&[BLOCK_SIZE], D::from_f64(1.0));
    let neg1 = T::full(&[BLOCK_SIZE], D::from_f64(-1.0));
    let s = one / (one + T::exp(neg1 * x.tensor));
    let y1 = x.tensor * s;
    T::store(y.tensor, y1, x.mask, &[], None, None);
}

/// Backward: dx = dy * (sigmoid(x) + y * (1 - sigmoid(x)))
///         = dy * (s + y - y*s)   where s = sigmoid(x)
#[kernel]
pub fn silu_backward<T: Triton, D: Float, const BLOCK_SIZE: i32>(
    dy_ptr: In<T::Pointer<D>>,
    x_ptr: In<T::Pointer<D>>,
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

pub struct SiluOp<D: Float> {
    pub forward: SiluForward<D>,
    pub backward: SiluBackward<D>,
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

#[cfg(test)]
mod tests {
    use super::*;

    /// `silu_forward`'s `#[tile(...)]`-tagged `x`/`y` share one
    /// flattened axis, so an output tile propagates to the input
    /// unchanged. `sigmoid_forward`/`logsigmoid_forward` have no
    /// `#[tile(...)]` and so generate no spec -- teenygrad-1tl.2.
    #[test]
    fn test_silu_tile_spec_declares_one_flat_axis_shared_by_x_and_y() {
        for rank in 1..=4 {
            let spec = SiluForward::<f32>::tile_spec(rank);
            assert_eq!(spec.loop_spec, None);
            assert_eq!((spec.inputs.len(), spec.outputs.len()), (1, 1));
            assert_eq!((spec.inputs[0].param, spec.outputs[0].param), ("x", "y"));

            for tensor in [spec.inputs[0], spec.outputs[0]] {
                assert_eq!(tensor.rank, rank);
                assert_eq!(tensor.axes.len(), 1, "one flattened axis");
                assert_eq!(tensor.axes[0].dims, (0..rank).collect::<Vec<_>>());
                assert_eq!(tensor.axes[0].block_const, "BLOCK_SIZE");
                assert_eq!(tensor.axes[0].extent_param, "n_elements");
            }
            spec.validate()
                .expect("a derived spec must be self-consistent");
        }
    }
}
