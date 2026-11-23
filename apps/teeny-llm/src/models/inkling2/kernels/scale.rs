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

//! Multiplication by a constant: `y = x * k`.
//!
//! Inkling-2 needs it once, for muP: the logits are divided by
//! `logits_mup_width_multiplier` (16.0 in Mini) before the final softcap. Folding
//! that into the unembedding weights at load time would also work and cost nothing,
//! but it would make a loaded weight differ from the checkpoint's, which is worse to
//! debug than one extra pass.
//!
//! Grid: `ceil(N / BLOCK)`.

use teeny_core::dtype::{AddOffsets, Comparison, Float, Tensor};
use teeny_macros::kernel;
use teeny_triton::triton::*;

/// The scaling above.
#[allow(clippy::multiple_bound_locations)]
#[kernel]
pub fn scale_forward<T: Triton, D: Float, const BLOCK: i32>(
    x_ptr: InPtr<T::Pointer<D>>,
    y_ptr: OutPtr<T::Pointer<D>>,
    N: i32,
    K: f32,
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
    T::store(
        y_ptr.add_offsets(offs),
        x * T::full(&[BLOCK], D::from_f64(K as f64)),
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

/// Graph-level `y = x * k`, shape preserving, no weights.
pub struct ScaleOp<D: FloatBytes + Send + Sync + 'static> {
    /// The constant.
    pub k: f32,
    _phantom: PhantomData<D>,
}

impl<D: FloatBytes + Send + Sync + 'static> ScaleOp<D> {
    /// Scaling by `k`.
    pub fn new(k: f32) -> Self {
        Self {
            k,
            _phantom: PhantomData,
        }
    }
}

impl<D: FloatBytes + Send + Sync + 'static> CustomOp for ScaleOp<D> {
    fn name(&self) -> &str {
        "llm.scale"
    }

    fn infer_output_shape(&self, input_shapes: &[&Shape]) -> Shape {
        input_shapes[0].to_vec()
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn lower(&self) -> Option<(String, String, String, Arc<dyn RuntimeOp>)> {
        let kernel = ScaleForward::<D>::new(BLOCK);
        Some((
            "scale_forward".to_string(),
            kernel.source,
            "entry_point".to_string(),
            Arc::new(ScaleRuntimeOp::<D> {
                k: self.k,
                _phantom: PhantomData,
            }),
        ))
    }
}

/// Runtime dispatch for [`ScaleOp`].
pub struct ScaleRuntimeOp<D: FloatBytes + Send + Sync + 'static> {
    k: f32,
    _phantom: PhantomData<D>,
}

impl<D: FloatBytes + Send + Sync + 'static> RuntimeOp for ScaleRuntimeOp<D> {
    fn n_activation_inputs(&self) -> usize {
        1
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
        visitor.visit_f32(self.k);
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
