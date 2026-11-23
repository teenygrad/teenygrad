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

//! Gemma's gated MLP activation, fused: `y = gelu_tanh(gate) * up`, with
//! `gelu_tanh(x) = 0.5 x (1 + tanh(sqrt(2/pi) (x + 0.044715 x^3)))` (PyTorch's
//! `gelu(approximate='tanh')`, Gemma's `gelu_pytorch_tanh`).
//!
//! `tanh(u)` is computed as `2 / (1 + exp(-2u)) - 1`. Grid: `ceil(N / BLOCK)`.

use teeny_core::dtype::{AddOffsets, Comparison, Float, Tensor};
use teeny_macros::kernel;
use teeny_triton::triton::*;

/// The fused activation above.
#[allow(clippy::multiple_bound_locations)]
#[kernel]
pub fn gemma_gelu_mul_forward<T: Triton, D: Float, const BLOCK: i32>(
    g_ptr: InPtr<T::Pointer<D>>,
    u_ptr: InPtr<T::Pointer<D>>,
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
    let u = T::load(
        u_ptr.add_offsets(offs),
        Some(m),
        Some(zeros),
        &[],
        None,
        None,
        None,
        false,
    );
    let one = T::full(&[BLOCK], D::from_f64(1.0));
    let two = T::full(&[BLOCK], D::from_f64(2.0));
    let half = T::full(&[BLOCK], D::from_f64(0.5));
    let c = T::full(&[BLOCK], D::from_f64(0.797_884_560_802_865_4));
    let a = T::full(&[BLOCK], D::from_f64(0.044715));
    let inner = c * (g + a * g * g * g);
    let th = two / (one + T::exp(zeros - two * inner)) - one;
    T::store(
        y_ptr.add_offsets(offs),
        half * g * (one + th) * u,
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

/// Graph-level `gelu_tanh(gate) * up`; inputs `gate`, `up` of one shape.
pub struct GemmaGeluMulOp<D: FloatBytes + Send + Sync + 'static> {
    _phantom: PhantomData<D>,
}

impl<D: FloatBytes + Send + Sync + 'static> Default for GemmaGeluMulOp<D> {
    fn default() -> Self {
        Self {
            _phantom: PhantomData,
        }
    }
}

impl<D: FloatBytes + Send + Sync + 'static> CustomOp for GemmaGeluMulOp<D> {
    fn name(&self) -> &str {
        "gemma.gelu_mul"
    }

    fn infer_output_shape(&self, input_shapes: &[&Shape]) -> Shape {
        assert_eq!(
            input_shapes[0], input_shapes[1],
            "gelu_mul: gate and up shapes"
        );
        input_shapes[0].clone()
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn lower(&self) -> Option<(String, String, String, Arc<dyn RuntimeOp>)> {
        let kernel = GemmaGeluMulForward::<D>::new(BLOCK);
        Some((
            "gemma_gelu_mul_forward".to_string(),
            kernel.source,
            "entry_point".to_string(),
            Arc::new(GemmaGeluMulRuntimeOp::<D> {
                _phantom: PhantomData,
            }),
        ))
    }
}

/// Runtime dispatch for [`GemmaGeluMulOp`].
pub struct GemmaGeluMulRuntimeOp<D: FloatBytes + Send + Sync + 'static> {
    _phantom: PhantomData<D>,
}

impl<D: FloatBytes + Send + Sync + 'static> RuntimeOp for GemmaGeluMulRuntimeOp<D> {
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
        let n: usize = output_shape.iter().product();
        [n.div_ceil(BLOCK as usize) as u32, 1, 1]
    }
}
