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

//! Inkling-2's softcap (`Inkling2LoopSoftcap`), elementwise:
//!
//! ```text
//! y = tanh(x / cap) * cap
//! ```
//!
//! The model uses three different caps: 4.0 on the MLP gate, 25.0 on the MLP up
//! projection, and 30.0 on the final logits. The reference computes it in fp32, so
//! the port runs it at the graph's dtype and relies on the caller using f32.
//!
//! `tanh(u)` is `2 / (1 + exp(-2u)) - 1`, as in Gemma's GELU. Grid: `ceil(N / BLOCK)`.

use teeny_core::dtype::{AddOffsets, Comparison, Float, Tensor};
use teeny_macros::kernel;
use teeny_triton::triton::*;

/// The softcap above.
#[allow(clippy::multiple_bound_locations)]
#[kernel]
pub fn softcap_forward<T: Triton, D: Float, const BLOCK: i32>(
    x_ptr: InPtr<T::Pointer<D>>,
    y_ptr: OutPtr<T::Pointer<D>>,
    N: i32,
    CAP: f32,
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
    let cap = T::full(&[BLOCK], D::from_f64(CAP as f64));
    let one = T::full(&[BLOCK], D::from_f64(1.0));
    let two = T::full(&[BLOCK], D::from_f64(2.0));
    let u = x / cap;
    let th = two / (one + T::exp(zeros - two * u)) - one;
    T::store(y_ptr.add_offsets(offs), th * cap, Some(m), &[], None, None);
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

/// Graph-level softcap, shape preserving.
pub struct SoftcapOp<D: FloatBytes + Send + Sync + 'static> {
    /// The cap. 4.0, 25.0 and 30.0 are the values the model ships with.
    pub cap: f32,
    _phantom: PhantomData<D>,
}

impl<D: FloatBytes + Send + Sync + 'static> SoftcapOp<D> {
    /// A softcap at `cap`, which must be positive.
    pub fn new(cap: f32) -> Self {
        assert!(cap > 0.0, "softcap: cap {cap} is not positive");
        Self {
            cap,
            _phantom: PhantomData,
        }
    }
}

impl<D: FloatBytes + Send + Sync + 'static> CustomOp for SoftcapOp<D> {
    fn name(&self) -> &str {
        "llm.softcap"
    }

    fn infer_output_shape(&self, input_shapes: &[&Shape]) -> Shape {
        input_shapes[0].to_vec()
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn lower(&self) -> Option<(String, String, String, Arc<dyn RuntimeOp>)> {
        let kernel = SoftcapForward::<D>::new(BLOCK);
        Some((
            "softcap_forward".to_string(),
            kernel.source,
            "entry_point".to_string(),
            Arc::new(SoftcapRuntimeOp::<D> {
                cap: self.cap,
                _phantom: PhantomData,
            }),
        ))
    }
}

/// Runtime dispatch for [`SoftcapOp`].
pub struct SoftcapRuntimeOp<D: FloatBytes + Send + Sync + 'static> {
    cap: f32,
    _phantom: PhantomData<D>,
}

impl<D: FloatBytes + Send + Sync + 'static> RuntimeOp for SoftcapRuntimeOp<D> {
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
        visitor.visit_f32(self.cap);
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
