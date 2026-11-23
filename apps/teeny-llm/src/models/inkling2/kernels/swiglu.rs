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

//! Inkling-2's gated MLP activation (`Inkling2LoopSoftcappedSwiGLU`), fused:
//!
//! ```text
//! y = softcap(gate, gate_cap) * sigmoid(gate) * softcap(up, up_cap)
//! ```
//!
//! The expert's fused gate-and-up projection emits `[S, 2 * I]`; `chunk(2, -1)` in the
//! reference means the FIRST half is the gate and the second the up, so this reads
//! both halves of one row and emits `[S, I]`.
//!
//! The shape of it is worth stating, because it is not the usual SwiGLU: the sigmoid
//! sees the **raw** gate, while the softcaps bound only the linear gate factor and
//! `up`. Capping the gate before the sigmoid would squash the gate towards 0.5 and
//! change the activation everywhere, not just in the tails.
//!
//! Inkling-2-Mini uses 4.0 for the gate and 25.0 for the up. `tanh` and `sigmoid` are
//! both built from `exp`, as in Gemma's GELU. Grid: `ceil(I / BLOCK)` by `S`.

use teeny_core::dtype::{AddOffsets, Comparison, Float, Tensor};
use teeny_macros::kernel;
use teeny_triton::triton::*;

/// The gated activation above.
#[allow(clippy::multiple_bound_locations)]
#[kernel]
pub fn swiglu_forward<T: Triton, D: Float, const BLOCK: i32>(
    x_ptr: InPtr<T::Pointer<D>>,
    y_ptr: OutPtr<T::Pointer<D>>,
    I: i32,
    GATE_CAP: f32,
    UP_CAP: f32,
) where
    T: Triton<I32Tensor = <T as Triton>::Tensor<i32>>,
    T::I32Tensor: Tensor<i32, 1>,
    T::I32Tensor: core::ops::Add<i32, Output = T::I32Tensor>,
    T::I32Tensor: core::ops::Mul<i32, Output = T::I32Tensor>,
    T::I32Tensor: Comparison<i32, BoolTensor = T::BoolTensor>,
    T::Pointer<D>: AddOffsets<i32, 1, T::I32Tensor, Output = T::Tensor<T::Pointer<D>>>,
{
    let bid = T::program_id(Axis::X);
    let row = T::program_id(Axis::Y);
    let offs = T::arange(0, BLOCK) + bid * BLOCK;
    let m = offs.lt(I);
    let zeros = T::zeros::<D>(&[BLOCK]);

    // One row of the projection is [gate (I) | up (I)].
    let gate = T::load(
        x_ptr.add_offsets(offs + row * 2 * I),
        Some(m),
        Some(zeros),
        &[],
        None,
        None,
        None,
        false,
    );
    let up = T::load(
        x_ptr.add_offsets(offs + I + row * 2 * I),
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
    let gcap = T::full(&[BLOCK], D::from_f64(GATE_CAP as f64));
    let ucap = T::full(&[BLOCK], D::from_f64(UP_CAP as f64));

    // tanh(u) = 2 / (1 + exp(-2u)) - 1
    let gt = two / (one + T::exp(zeros - two * (gate / gcap))) - one;
    let ut = two / (one + T::exp(zeros - two * (up / ucap))) - one;
    // The sigmoid sees the raw gate, not the capped one.
    let sig = one / (one + T::exp(zeros - gate));

    T::store(
        y_ptr.add_offsets(offs + row * I),
        gt * gcap * sig * (ut * ucap),
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

const BLOCK: i32 = 256;

/// Graph-level softcapped SwiGLU: `[S, 2 * I]` to `[S, I]`.
pub struct SwigluOp<D: FloatBytes + Send + Sync + 'static> {
    /// Width of one half of the projection.
    pub intermediate: usize,
    /// Softcap on the gate. 4.0 in Inkling-2-Mini.
    pub gate_cap: f32,
    /// Softcap on the up projection. 25.0 in Inkling-2-Mini.
    pub up_cap: f32,
    _phantom: PhantomData<D>,
}

impl<D: FloatBytes + Send + Sync + 'static> SwigluOp<D> {
    /// A gated activation over `intermediate` lanes with the two caps.
    pub fn new(intermediate: usize, gate_cap: f32, up_cap: f32) -> Self {
        assert!(
            gate_cap > 0.0 && up_cap > 0.0,
            "swiglu: caps {gate_cap} and {up_cap} must be positive"
        );
        Self {
            intermediate,
            gate_cap,
            up_cap,
            _phantom: PhantomData,
        }
    }
}

impl<D: FloatBytes + Send + Sync + 'static> CustomOp for SwigluOp<D> {
    fn name(&self) -> &str {
        "llm.swiglu_softcap"
    }

    fn infer_output_shape(&self, input_shapes: &[&Shape]) -> Shape {
        vec![input_shapes[0][0], Some(self.intermediate)]
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn lower(&self) -> Option<(String, String, String, Arc<dyn RuntimeOp>)> {
        let kernel = SwigluForward::<D>::new(BLOCK);
        Some((
            "swiglu_forward".to_string(),
            kernel.source,
            "entry_point".to_string(),
            Arc::new(SwigluRuntimeOp::<D> {
                intermediate: self.intermediate,
                gate_cap: self.gate_cap,
                up_cap: self.up_cap,
                _phantom: PhantomData,
            }),
        ))
    }
}

/// Runtime dispatch for [`SwigluOp`].
pub struct SwigluRuntimeOp<D: FloatBytes + Send + Sync + 'static> {
    intermediate: usize,
    gate_cap: f32,
    up_cap: f32,
    _phantom: PhantomData<D>,
}

impl<D: FloatBytes + Send + Sync + 'static> RuntimeOp for SwigluRuntimeOp<D> {
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
        _output_shape: &[usize],
        _output_row_stride: i32,
        visitor: &mut dyn ArgVisitor,
    ) {
        assert_eq!(
            inputs[0].1[1],
            2 * self.intermediate,
            "swiglu: the projection must be gate and up together"
        );
        visitor.visit_ptr(inputs[0].0);
        visitor.visit_ptr(output);
        visitor.visit_i32(self.intermediate as i32);
        visitor.visit_f32(self.gate_cap);
        visitor.visit_f32(self.up_cap);
    }

    fn grid(&self, output_shape: &[usize]) -> [u32; 3] {
        [
            self.intermediate.div_ceil(BLOCK as usize) as u32,
            output_shape[0] as u32,
            1,
        ]
    }
}
