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

//! Regrouping a tensor's rows: `[R, C]` to `[R * f, C / f]`, same elements in the
//! same order.
//!
//! Inkling-2 needs it once. The relative-position projection is per head: the
//! reference holds `relative_states` as `[T, heads, d_rel]` and multiplies by a
//! `[d_rel, rel_extent]` matrix, so the matmul's rows are `(token, head)` pairs. The
//! graph carries that tensor as `[T, heads * d_rel]`, which is the same memory, but
//! `Linear4Op` reads the row width as its reduction size and would use
//! `heads * d_rel`.
//!
//! The data does not move, only the shape, but the graph has no view: a node has one
//! shape, so relabelling would not reach the ops that read it. Hence a copy.
//!
//! Grid: `ceil(N / BLOCK)`.

use teeny_core::dtype::{AddOffsets, Comparison, Float, Tensor};
use teeny_macros::kernel;
use teeny_triton::triton::*;

/// The scaling above.
#[allow(clippy::multiple_bound_locations)]
#[kernel]
pub fn regroup_forward<T: Triton, D: Float, const BLOCK: i32>(
    x_ptr: InPtr<T::Pointer<D>>,
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
    T::store(y_ptr.add_offsets(offs), x, Some(m), &[], None, None);
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

/// Graph-level regroup: `[R, C]` to `[R * factor, C / factor]`.
pub struct RegroupOp<D: FloatBytes + Send + Sync + 'static> {
    /// How many rows each input row becomes.
    pub factor: usize,
    _phantom: PhantomData<D>,
}

impl<D: FloatBytes + Send + Sync + 'static> RegroupOp<D> {
    /// Splits every row into `factor` rows.
    pub fn new(factor: usize) -> Self {
        assert!(factor > 0, "regroup: factor must be positive");
        Self {
            factor,
            _phantom: PhantomData,
        }
    }
}

impl<D: FloatBytes + Send + Sync + 'static> CustomOp for RegroupOp<D> {
    fn name(&self) -> &str {
        "llm.regroup"
    }

    fn infer_output_shape(&self, input_shapes: &[&Shape]) -> Shape {
        vec![
            input_shapes[0][0].map(|r| r * self.factor),
            input_shapes[0][1].map(|c| c / self.factor),
        ]
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn lower(&self) -> Option<(String, String, String, Arc<dyn RuntimeOp>)> {
        let kernel = RegroupForward::<D>::new(BLOCK);
        Some((
            "regroup_forward".to_string(),
            kernel.source,
            "entry_point".to_string(),
            Arc::new(RegroupRuntimeOp::<D> {
                factor: self.factor,
                _phantom: PhantomData,
            }),
        ))
    }
}

/// Runtime dispatch for [`RegroupOp`].
pub struct RegroupRuntimeOp<D: FloatBytes + Send + Sync + 'static> {
    factor: usize,
    _phantom: PhantomData<D>,
}

impl<D: FloatBytes + Send + Sync + 'static> RuntimeOp for RegroupRuntimeOp<D> {
    fn n_activation_inputs(&self) -> usize {
        1
    }

    /// `factor` times the input's rows, each `factor` times narrower, which a graph
    /// that leaves the token count unknown cannot say until the rows are known.
    fn compute_concrete_output_shape(
        &self,
        input_shapes: &[&[usize]],
        _resolved: &[usize],
    ) -> Vec<usize> {
        vec![
            input_shapes[0][0] * self.factor,
            input_shapes[0][1] / self.factor,
        ]
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
