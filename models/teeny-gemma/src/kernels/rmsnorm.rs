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

//! Gemma's RMSNorm (`Gemma3RMSNorm`): over each row of `R` elements,
//!
//! ```text
//! y = x / sqrt(mean(x^2) + eps) * (1 + w)
//! ```
//!
//! Note the `1 + w`: Gemma stores the scale offset from one. The row length need not
//! be the tensor's last axis: the per-head `q_norm`/`k_norm` normalise rows of
//! `head_dim` inside a `[T, heads * head_dim]` tensor.
//!
//! Grid: one program per row; `BLOCK` (a power of two) must be at least `R`.

use teeny_core::dtype::{AddOffsets, Comparison, Float, Tensor};
use teeny_macros::kernel;
use teeny_triton::triton::*;

/// The normalisation above.
#[allow(clippy::multiple_bound_locations)]
#[kernel]
pub fn gemma_rmsnorm_forward<T: Triton, D: Float, const BLOCK: i32>(
    x_ptr: InPtr<T::Pointer<D>>,
    w_ptr: InPtr<T::Pointer<D>>,
    y_ptr: OutPtr<T::Pointer<D>>,
    R: i32,
) where
    T: Triton<I32Tensor = <T as Triton>::Tensor<i32>>,
    T::I32Tensor: Tensor<i32, 1>,
    T::I32Tensor: core::ops::Add<i32, Output = T::I32Tensor>,
    T::I32Tensor: core::ops::Mul<i32, Output = T::I32Tensor>,
    T::I32Tensor: Comparison<i32, BoolTensor = T::BoolTensor>,
    T::Pointer<D>: AddOffsets<i32, 1, T::I32Tensor, Output = T::Tensor<T::Pointer<D>>>,
{
    let row = T::program_id(Axis::X);
    let offs = T::arange(0, BLOCK);
    let m = offs.lt(R);
    let zeros = T::zeros::<D>(&[BLOCK]);
    let x = T::load(
        x_ptr.add_offsets(offs + row * R),
        Some(m),
        Some(zeros),
        &[],
        None,
        None,
        None,
        false,
    );
    let w = T::load(
        w_ptr.add_offsets(offs),
        Some(m),
        Some(zeros),
        &[],
        None,
        None,
        None,
        false,
    );
    let n = T::cast::<i32, D>(T::zeros::<i32>(&[BLOCK]) + R, None, false);
    let ms = T::sum(x * x, Some(0), true) / n;
    let inv = T::rsqrt(ms + T::full(&[BLOCK], D::from_f64(1e-6)));
    let one = T::full(&[BLOCK], D::from_f64(1.0));
    T::store(
        y_ptr.add_offsets(offs + row * R),
        x * inv * (one + w),
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

/// Graph-level Gemma RMSNorm over rows of `row` elements. Shape-preserving. Weight
/// slot `weight` `[row]`.
pub struct GemmaRmsNormOp<D: FloatBytes + Send + Sync + 'static> {
    /// Elements per normalised row.
    pub row: usize,
    _phantom: PhantomData<D>,
}

impl<D: FloatBytes + Send + Sync + 'static> GemmaRmsNormOp<D> {
    /// Creates the op.
    pub fn new(row: usize) -> Self {
        Self {
            row,
            _phantom: PhantomData,
        }
    }

    fn block(&self) -> i32 {
        self.row.next_power_of_two() as i32
    }
}

impl<D: FloatBytes + Send + Sync + 'static> CustomOp for GemmaRmsNormOp<D> {
    fn name(&self) -> &str {
        "gemma.rmsnorm"
    }

    fn infer_output_shape(&self, input_shapes: &[&Shape]) -> Shape {
        input_shapes[0].clone()
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn lower(&self) -> Option<(String, String, String, Arc<dyn RuntimeOp>)> {
        let kernel = GemmaRmsnormForward::<D>::new(self.block());
        Some((
            "gemma_rmsnorm_forward".to_string(),
            kernel.source,
            "entry_point".to_string(),
            Arc::new(GemmaRmsNormRuntimeOp::<D> {
                row: self.row,
                _phantom: PhantomData,
            }),
        ))
    }
}

/// Runtime dispatch for [`GemmaRmsNormOp`].
pub struct GemmaRmsNormRuntimeOp<D: FloatBytes + Send + Sync + 'static> {
    row: usize,
    _phantom: PhantomData<D>,
}

impl<D: FloatBytes + Send + Sync + 'static> RuntimeOp for GemmaRmsNormRuntimeOp<D> {
    fn n_activation_inputs(&self) -> usize {
        1
    }

    fn param_shapes(&self, _input_shapes: &[&[usize]], _output_shape: &[usize]) -> Vec<Vec<usize>> {
        vec![vec![self.row]]
    }

    fn param_names(&self) -> &'static [&'static str] {
        &["weight"]
    }

    fn pack_args(
        &self,
        inputs: &[(RawPtr, &[usize])],
        params: &[RawPtr],
        output: RawPtr,
        _output_shape: &[usize],
        _output_row_stride: i32,
        visitor: &mut dyn ArgVisitor,
    ) {
        visitor.visit_ptr(inputs[0].0);
        visitor.visit_ptr(params[0]);
        visitor.visit_ptr(output);
        visitor.visit_i32(self.row as i32);
    }

    fn grid(&self, output_shape: &[usize]) -> [u32; 3] {
        let n: usize = output_shape.iter().product();
        assert!(
            n.is_multiple_of(self.row),
            "rmsnorm: size not a multiple of the row"
        );
        [(n / self.row) as u32, 1, 1]
    }
}
