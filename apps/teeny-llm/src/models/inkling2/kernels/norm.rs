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

//! Inkling-2's RMS norms, over rows of `R` elements.
//!
//! ```text
//! weighted:   y = w * x / sqrt(mean(x^2) + eps)
//! unweighted: y =     x / sqrt(mean(x^2) + eps)
//! ```
//!
//! **Not Gemma's.** [`crate::models::functiongemma::kernels::rmsnorm`] applies
//! `1 + w`, because Gemma stores the scale offset from one. `Inkling2LoopRMSNorm`
//! stores it directly (the parameter is initialised to ones, not zeros), so reusing
//! the Gemma kernel here would add one to every scale.
//!
//! The unweighted form (`Inkling2LoopUnweightedRMSNorm`) normalises the value
//! projection, where there is no learned scale at all. Grid: one program per row.

use teeny_core::dtype::{AddOffsets, Comparison, Float, Tensor};
use teeny_macros::kernel;
use teeny_triton::triton::*;

/// `y = w * x / sqrt(mean(x^2) + eps)`.
#[allow(clippy::multiple_bound_locations)]
#[kernel]
pub fn rmsnorm_forward<T: Triton, D: Float, const BLOCK: i32>(
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
    let inv = T::rsqrt(T::sum(x * x, Some(0), true) / n + T::full(&[BLOCK], D::from_f64(1e-6)));
    T::store(
        y_ptr.add_offsets(offs + row * R),
        x * inv * w,
        Some(m),
        &[],
        None,
        None,
    );
}

/// `y = x / sqrt(mean(x^2) + eps)`, with no learned scale.
#[allow(clippy::multiple_bound_locations)]
#[kernel]
pub fn unweighted_rmsnorm_forward<T: Triton, D: Float, const BLOCK: i32>(
    x_ptr: InPtr<T::Pointer<D>>,
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
    let n = T::cast::<i32, D>(T::zeros::<i32>(&[BLOCK]) + R, None, false);
    let inv = T::rsqrt(T::sum(x * x, Some(0), true) / n + T::full(&[BLOCK], D::from_f64(1e-6)));
    T::store(
        y_ptr.add_offsets(offs + row * R),
        x * inv,
        Some(m),
        &[],
        None,
        None,
    );
}

// ---------------------------------------------------------------------------
// Graph-level ops
// ---------------------------------------------------------------------------

use core::marker::PhantomData;
use std::any::Any;
use std::sync::Arc;
use teeny_core::device::program::ArgVisitor;
use teeny_core::dtype::FloatBytes;
use teeny_core::graph::{CustomOp, Shape};
use teeny_core::model::{RawPtr, RuntimeOp};

macro_rules! rmsnorm_op {
    ($op:ident, $rt:ident, $fwd:ident, $entry:literal, $name:literal, $weighted:literal) => {
        #[doc = concat!("Graph-level `", $name, "`, shape preserving over rows of `row`.")]
        pub struct $op<D: FloatBytes + Send + Sync + 'static> {
            /// Row length.
            pub row: usize,
            block: i32,
            _phantom: PhantomData<D>,
        }

        impl<D: FloatBytes + Send + Sync + 'static> $op<D> {
            #[doc = "A norm over rows of `row` elements."]
            pub fn new(row: usize) -> Self {
                Self {
                    row,
                    block: row.next_power_of_two() as i32,
                    _phantom: PhantomData,
                }
            }
        }

        impl<D: FloatBytes + Send + Sync + 'static> CustomOp for $op<D> {
            fn name(&self) -> &str {
                $name
            }
            fn infer_output_shape(&self, input_shapes: &[&Shape]) -> Shape {
                input_shapes[0].to_vec()
            }
            fn as_any(&self) -> &dyn Any {
                self
            }
            fn lower(&self) -> Option<(String, String, String, Arc<dyn RuntimeOp>)> {
                let kernel = $fwd::<D>::new(self.block);
                Some((
                    $entry.to_string(),
                    kernel.source,
                    "entry_point".to_string(),
                    Arc::new($rt::<D> {
                        row: self.row,
                        _phantom: PhantomData,
                    }),
                ))
            }
        }

        #[doc = concat!("Runtime dispatch for [`", stringify!($op), "`].")]
        pub struct $rt<D: FloatBytes + Send + Sync + 'static> {
            row: usize,
            _phantom: PhantomData<D>,
        }

        impl<D: FloatBytes + Send + Sync + 'static> RuntimeOp for $rt<D> {
            fn n_activation_inputs(&self) -> usize {
                1
            }
            fn param_shapes(&self, _i: &[&[usize]], _o: &[usize]) -> Vec<Vec<usize>> {
                if $weighted {
                    vec![vec![self.row]]
                } else {
                    Vec::new()
                }
            }
            fn param_names(&self) -> &'static [&'static str] {
                if $weighted { &["weight"] } else { &[] }
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
                if $weighted {
                    visitor.visit_ptr(params[0]);
                }
                visitor.visit_ptr(output);
                visitor.visit_i32(self.row as i32);
            }
            fn grid(&self, output_shape: &[usize]) -> [u32; 3] {
                [
                    (output_shape.iter().product::<usize>() / self.row) as u32,
                    1,
                    1,
                ]
            }
        }
    };
}

rmsnorm_op!(
    RmsNormOp,
    RmsNormRuntimeOp,
    RmsnormForward,
    "rmsnorm_forward",
    "llm.rmsnorm",
    true
);
rmsnorm_op!(
    UnweightedRmsNormOp,
    UnweightedRmsNormRuntimeOp,
    UnweightedRmsnormForward,
    "unweighted_rmsnorm_forward",
    "llm.unweighted_rmsnorm",
    false
);
