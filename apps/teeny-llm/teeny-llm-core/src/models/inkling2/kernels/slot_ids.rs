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

//! Re-packs the router's row so the down projection can be gathered with `top_k = 1`.
//!
//! ```text
//! out[t * top_k + s, 0] = 0
//! out[t * top_k + s, 1] = routed[t, top_k + s]
//! ```
//!
//! The router emits one row per token holding `top_k` weights then `top_k` ids. After
//! the first expert gather there is one row per (token, slot), so the down projection
//! wants one id per row, in the `[unused weight, id]` layout
//! [`super::expert_gather`] reads with `top_k = 1`.
//!
//! This existed as host arithmetic first. Doing it on the device is what lets a
//! decoder layer be a **single graph**: otherwise the host has to run the router,
//! re-pack, and run the rest, which is two dispatches per layer and 96 per token at
//! 48 effective layers.
//!
//! Grid: one program per `(token, slot)`.

use teeny_core::dtype::{AddOffsets, Comparison, Float, Tensor};
use teeny_macros::kernel;
use teeny_triton::triton::*;

/// The re-pack above.
#[allow(clippy::multiple_bound_locations)]
#[kernel]
pub fn slot_ids_forward<T: Triton, D: Float, const BLOCK: i32>(
    r_ptr: InPtr<T::Pointer<D>>,
    y_ptr: OutPtr<T::Pointer<D>>,
    TOP_K: i32,
) where
    T: Triton<I32Tensor = <T as Triton>::Tensor<i32>>,
    T::I32Tensor: Tensor<i32, 1>,
    T::I32Tensor: core::ops::Add<i32, Output = T::I32Tensor>,
    T::I32Tensor: core::ops::Mul<i32, Output = T::I32Tensor>,
    T::I32Tensor: Comparison<i32, BoolTensor = T::BoolTensor>,
    T::Pointer<D>: AddOffsets<i32, 1, T::I32Tensor, Output = T::Tensor<T::Pointer<D>>>,
{
    let pid = T::program_id(Axis::X);
    let token = pid / TOP_K;
    let slot = pid - token * TOP_K;
    let lane = T::arange(0, BLOCK);
    let zeros = T::zeros::<D>(&[BLOCK]);

    let id = T::load(
        r_ptr.add_offsets(T::zeros::<i32>(&[BLOCK]) + (token * 2 * TOP_K + TOP_K + slot)),
        None,
        Some(zeros),
        &[],
        None,
        None,
        None,
        false,
    );
    // Lane 0 is the unused weight slot, lane 1 the id.
    T::store(
        y_ptr.add_offsets(lane + pid * 2),
        T::where_(lane.eq(1), id, zeros),
        Some(lane.lt(2)),
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

/// Graph-level re-pack: `[T, 2 * top_k]` to `[T * top_k, 2]`.
pub struct SlotIdsOp<D: FloatBytes + Send + Sync + 'static> {
    /// Experts firing per token.
    pub top_k: usize,
    _phantom: PhantomData<D>,
}

impl<D: FloatBytes + Send + Sync + 'static> SlotIdsOp<D> {
    /// A re-pack for `top_k` slots per token.
    pub fn new(top_k: usize) -> Self {
        assert!(top_k > 0, "slot ids: top_k must be positive");
        Self {
            top_k,
            _phantom: PhantomData,
        }
    }
}

impl<D: FloatBytes + Send + Sync + 'static> CustomOp for SlotIdsOp<D> {
    fn name(&self) -> &str {
        "llm.slot_ids"
    }
    fn infer_output_shape(&self, input_shapes: &[&Shape]) -> Shape {
        vec![input_shapes[0][0].map(|t| t * self.top_k), Some(2)]
    }
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn lower(&self) -> Option<(String, String, String, Arc<dyn RuntimeOp>)> {
        let kernel = SlotIdsForward::<D>::new(2);
        Some((
            "slot_ids_forward".to_string(),
            kernel.source,
            "entry_point".to_string(),
            Arc::new(SlotIdsRuntimeOp::<D> {
                top_k: self.top_k,
                _phantom: PhantomData,
            }),
        ))
    }
}

/// Runtime dispatch for [`SlotIdsOp`].
pub struct SlotIdsRuntimeOp<D: FloatBytes + Send + Sync + 'static> {
    top_k: usize,
    _phantom: PhantomData<D>,
}

impl<D: FloatBytes + Send + Sync + 'static> RuntimeOp for SlotIdsRuntimeOp<D> {
    fn n_activation_inputs(&self) -> usize {
        1
    }

    /// One row per (token, slot): `top_k` times the router's rows.
    fn compute_concrete_output_shape(&self, input_shapes: &[&[usize]], _resolved: &[usize]) -> Vec<usize> {
        vec![input_shapes[0][0] * self.top_k, 2]
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
        assert_eq!(inputs[0].1[1], 2 * self.top_k, "slot ids: router row width");
        visitor.visit_ptr(inputs[0].0);
        visitor.visit_ptr(output);
        visitor.visit_i32(self.top_k as i32);
    }
    fn grid(&self, output_shape: &[usize]) -> [u32; 3] {
        [output_shape[0] as u32, 1, 1]
    }
}
