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

//! The sigmoid gate of Inkling-2's output projection
//! (`Inkling2LoopGatedOutputProjection`), elementwise over two tensors of one shape:
//!
//! ```text
//! y = x * sigmoid(g)
//! ```
//!
//! `g` is the gate projection of the **layer input**, not of the attention output, so
//! the layer decides how much of the attention result to let through based on what it
//! was given. Both arguments are already `[T, heads * head_dim]`.
//!
//! Grid: `ceil(N / BLOCK)`.

use teeny_core::dtype::{AddOffsets, Comparison, Float, Tensor};
use teeny_macros::kernel;
use teeny_triton::triton::*;

/// `y = x * sigmoid(g)`.
#[allow(clippy::multiple_bound_locations)]
#[kernel]
pub fn gated_mul_forward<T: Triton, D: Float, const BLOCK: i32>(
    x_ptr: InPtr<T::Pointer<D>>,
    g_ptr: InPtr<T::Pointer<D>>,
    y_ptr: OutPtr<T::Pointer<D>>,
    N: i32,
) where
    T: Triton<I32Tensor = <T as Triton>::Tensor<i32>>,
    T::I32Tensor: Tensor<i32, 1>,
    T::I32Tensor: core::ops::Add<i32, Output = T::I32Tensor>,
    T::I32Tensor: core::ops::Mul<i32, Output = T::I32Tensor>,
    T::I32Tensor: Comparison<i32, BoolTensor = T::BoolTensor>,
    T::Pointer<D>: AddOffsets<i32, 1, T::I32Tensor, Output = T::Tensor<T::Pointer<D>>>,
{
    let pid = T::program_id(Axis::X);
    let offs = T::arange(0, BLOCK) + pid * BLOCK;
    let m = offs.lt(N);
    let zeros = T::zeros::<D>(&[BLOCK]);
    let x = T::load(
        x_ptr.add_offsets(offs),
        Some(m),
        Some(zeros),
        &[],
        None,
        None,
        None,
        false,
    );
    let g = T::load(
        g_ptr.add_offsets(offs),
        Some(m),
        Some(zeros),
        &[],
        None,
        None,
        None,
        false,
    );
    let one = T::full(&[BLOCK], D::from_f64(1.0));
    T::store(
        y_ptr.add_offsets(offs),
        x * (one / (one + T::exp(zeros - g))),
        Some(m),
        &[],
        None,
        None,
    );
}

// ---------------------------------------------------------------------------
// Graph-level op
// ---------------------------------------------------------------------------

use core::marker::PhantomData;
use std::any::Any;
use std::sync::Arc;
use teeny_core::device::program::ArgVisitor;
use teeny_core::dtype::FloatBytes;
use teeny_core::graph::{CustomOp, Shape};
use teeny_core::model::{RawPtr, RuntimeOp};

const BLOCK: i32 = 1024;

/// Graph-level `x * sigmoid(g)`; inputs `x` and `g` of one shape, no weights.
pub struct GatedMulOp<D: FloatBytes + Send + Sync + 'static> {
    _phantom: PhantomData<D>,
}

impl<D: FloatBytes + Send + Sync + 'static> Default for GatedMulOp<D> {
    fn default() -> Self {
        Self {
            _phantom: PhantomData,
        }
    }
}

impl<D: FloatBytes + Send + Sync + 'static> CustomOp for GatedMulOp<D> {
    fn name(&self) -> &str {
        "llm.gated_mul"
    }

    fn infer_output_shape(&self, input_shapes: &[&Shape]) -> Shape {
        input_shapes[0].to_vec()
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn lower(&self) -> Option<(String, String, String, Arc<dyn RuntimeOp>)> {
        let kernel = GatedMulForward::<D>::new(BLOCK);
        Some((
            "gated_mul_forward".to_string(),
            kernel.source,
            "entry_point".to_string(),
            Arc::new(GatedMulRuntimeOp::<D> {
                _phantom: PhantomData,
            }),
        ))
    }
}

/// Runtime dispatch for [`GatedMulOp`].
pub struct GatedMulRuntimeOp<D: FloatBytes + Send + Sync + 'static> {
    _phantom: PhantomData<D>,
}

impl<D: FloatBytes + Send + Sync + 'static> RuntimeOp for GatedMulRuntimeOp<D> {
    fn n_activation_inputs(&self) -> usize {
        2
    }

    fn param_shapes(&self, _input_shapes: &[&[usize]], _output_shape: &[usize]) -> Vec<Vec<usize>> {
        Vec::new()
    }

    fn pack_args(
        &self,
        inputs: &[(RawPtr, &[usize])],
        _params: &[RawPtr],
        output: RawPtr,
        output_shape: &[usize],
        _output_row_stride: i32,
        visitor: &mut dyn ArgVisitor,
    ) {
        visitor.visit_ptr(inputs[0].0);
        visitor.visit_ptr(inputs[1].0);
        visitor.visit_ptr(output);
        visitor.visit_i32(output_shape.iter().product::<usize>() as i32);
    }

    fn grid(&self, output_shape: &[usize]) -> [u32; 3] {
        [
            output_shape
                .iter()
                .product::<usize>()
                .div_ceil(BLOCK as usize) as u32,
            1,
            1,
        ]
    }
}
