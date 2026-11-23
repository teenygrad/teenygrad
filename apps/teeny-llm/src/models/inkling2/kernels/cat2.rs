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

//! Concatenation of two equally tall tensors along the last axis:
//! `[T, A]` and `[T, B]` to `[T, A + B]`.
//!
//! Inkling-2's loop controller projects `concat(norm(hidden), norm(message))` through
//! a single `[rank, 2 * hidden]` weight, so the halves have to meet before the
//! projection. Splitting that weight in two at load time and summing two linears
//! would be algebraically identical, but it would stop the slot matching the
//! checkpoint's `down_weight`, so the concatenation is done here instead.
//!
//! Grid: `ceil((A + B) / BLOCK)` by `T`.

use teeny_core::dtype::{AddOffsets, Comparison, Float, Tensor};
use teeny_macros::kernel;
use teeny_triton::triton::*;

/// The concatenation above.
#[allow(clippy::multiple_bound_locations)]
#[kernel]
pub fn cat2_forward<T: Triton, D: Float, const BLOCK: i32>(
    a_ptr: InPtr<T::Pointer<D>>,
    b_ptr: InPtr<T::Pointer<D>>,
    y_ptr: OutPtr<T::Pointer<D>>,
    A: i32,
    B: i32,
) where
    T: Triton<I32Tensor = <T as Triton>::Tensor<i32>>,
    T::I32Tensor: Tensor<i32, 1>,
    T::I32Tensor: core::ops::Add<i32, Output = T::I32Tensor>,
    T::I32Tensor: core::ops::Mul<i32, Output = T::I32Tensor>,
    T::I32Tensor: Comparison<i32, BoolTensor = T::BoolTensor>,
    T::BoolTensor: core::ops::BitAnd<Output = T::BoolTensor>,
    T::Pointer<D>: AddOffsets<i32, 1, T::I32Tensor, Output = T::Tensor<T::Pointer<D>>>,
{
    let bid = T::program_id(Axis::X);
    let row = T::program_id(Axis::Y);
    let col = T::arange(0, BLOCK) + bid * BLOCK;
    let zeros = T::zeros::<D>(&[BLOCK]);

    let from_a = col.lt(A);
    let from_b = col.ge(A) & col.lt(A + B);
    // Out-of-range lanes are masked, so the clamped addresses they would use never
    // reach memory.
    let a = T::load(
        a_ptr.add_offsets(T::minimum(col, T::zeros::<i32>(&[BLOCK]) + (A - 1)) + row * A),
        Some(from_a),
        Some(zeros),
        &[],
        None,
        None,
        None,
        false,
    );
    let b = T::load(
        b_ptr.add_offsets(T::maximum(col + (-A), T::zeros::<i32>(&[BLOCK])) + row * B),
        Some(from_b),
        Some(zeros),
        &[],
        None,
        None,
        None,
        false,
    );
    T::store(
        y_ptr.add_offsets(col + row * (A + B)),
        a + b,
        Some(from_a | from_b),
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

const BLOCK: i32 = 256;

/// Graph-level concatenation along the last axis; inputs `a` `[T, A]` and `b` `[T, B]`.
pub struct Cat2Op<D: FloatBytes + Send + Sync + 'static> {
    /// Width of the first input.
    pub a: usize,
    /// Width of the second.
    pub b: usize,
    _phantom: PhantomData<D>,
}

impl<D: FloatBytes + Send + Sync + 'static> Cat2Op<D> {
    /// A concatenation of widths `a` and `b`.
    pub fn new(a: usize, b: usize) -> Self {
        Self {
            a,
            b,
            _phantom: PhantomData,
        }
    }
}

impl<D: FloatBytes + Send + Sync + 'static> CustomOp for Cat2Op<D> {
    fn name(&self) -> &str {
        "llm.cat2"
    }
    fn infer_output_shape(&self, input_shapes: &[&Shape]) -> Shape {
        vec![input_shapes[0][0], Some(self.a + self.b)]
    }
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn lower(&self) -> Option<(String, String, String, Arc<dyn RuntimeOp>)> {
        let kernel = Cat2Forward::<D>::new(BLOCK);
        Some((
            "cat2_forward".to_string(),
            kernel.source,
            "entry_point".to_string(),
            Arc::new(Cat2RuntimeOp::<D> {
                a: self.a,
                b: self.b,
                _phantom: PhantomData,
            }),
        ))
    }
}

/// Runtime dispatch for [`Cat2Op`].
pub struct Cat2RuntimeOp<D: FloatBytes + Send + Sync + 'static> {
    a: usize,
    b: usize,
    _phantom: PhantomData<D>,
}

impl<D: FloatBytes + Send + Sync + 'static> RuntimeOp for Cat2RuntimeOp<D> {
    fn n_activation_inputs(&self) -> usize {
        2
    }
    fn param_shapes(&self, _i: &[&[usize]], _o: &[usize]) -> Vec<Vec<usize>> {
        Vec::new()
    }
    fn pack_args(
        &self,
        inputs: &[(RawPtr, &[usize])],
        _params: &[RawPtr],
        output: RawPtr,
        _output_shape: &[usize],
        _output_row_stride: i32,
        visitor: &mut dyn ArgVisitor,
    ) {
        assert_eq!(inputs[0].1[1], self.a, "cat2: first width");
        assert_eq!(inputs[1].1[1], self.b, "cat2: second width");
        visitor.visit_ptr(inputs[0].0);
        visitor.visit_ptr(inputs[1].0);
        visitor.visit_ptr(output);
        visitor.visit_i32(self.a as i32);
        visitor.visit_i32(self.b as i32);
    }
    fn grid(&self, output_shape: &[usize]) -> [u32; 3] {
        [
            (self.a + self.b).div_ceil(BLOCK as usize) as u32,
            output_shape[0] as u32,
            1,
        ]
    }
}
