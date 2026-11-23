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

//! Inkling-2's top-k router (`Inkling2LoopTopKRouter`), one program per token.
//!
//! ```text
//! scores  = sigmoid(logits)                       logits = hidden @ gate_weight^T
//! indices = topk(scores + bias, k).indices        the bias steers the CHOICE
//! weights = scores[indices]                       but not the WEIGHT
//! weights = weights / (sum(weights) + 1e-20) * route_scale
//! ```
//!
//! The detail worth stating, because it would be invisible if wrong: the correction
//! bias is added only for picking the experts. The weights are gathered from the
//! unbiased scores, so a bias that changes which experts fire does not change how
//! much each contributes. Routing runs in fp32 in the reference, so the caller
//! should keep this at f32.
//!
//! `scoring_func` defaults to `"sigmoid"` and Inkling-2-Mini does not override it;
//! `route_scale` is 8.0 and `num_experts_per_tok` is 8 of 128.
//!
//! **Output packing.** A graph op has one output, so the row is `[2 * K]`: the K
//! weights, then the K expert ids as floats. Ids are exact in f32 well past the 128
//! experts here.
//!
//! The top-k is K rounds of argmax, each masking out the winner, which is cheap at
//! K=8 and avoids needing a sort. Grid: one program per token; `BLOCK` (a power of
//! two) must be at least the expert count.

use teeny_core::dtype::{AddOffsets, Comparison, Float, Tensor};
use teeny_macros::kernel;
use teeny_triton::triton::*;

/// The router above.
#[allow(clippy::multiple_bound_locations)]
#[kernel]
pub fn topk_router_forward<T: Triton, D: Float, const BLOCK: i32, const K: i32>(
    logits_ptr: InPtr<T::Pointer<D>>,
    bias_ptr: InPtr<T::Pointer<D>>,
    out_ptr: OutPtr<T::Pointer<D>>,
    EXPERTS: i32,
    SCALE: f32,
) where
    T: Triton<I32Tensor = <T as Triton>::Tensor<i32>>,
    T::I32Tensor: Tensor<i32, 1>,
    T::I32Tensor: core::ops::Add<i32, Output = T::I32Tensor>,
    T::I32Tensor: core::ops::Mul<i32, Output = T::I32Tensor>,
    T::I32Tensor: Comparison<i32, BoolTensor = T::BoolTensor>,
    T::Pointer<D>: AddOffsets<i32, 1, T::I32Tensor, Output = T::Tensor<T::Pointer<D>>>,
{
    let row = T::program_id(Axis::X);
    let lane = T::arange(0, BLOCK);
    let m = lane.lt(EXPERTS);
    let zeros = T::zeros::<D>(&[BLOCK]);

    let logits = T::load(
        logits_ptr.add_offsets(lane + row * EXPERTS),
        Some(m),
        Some(zeros),
        &[],
        None,
        None,
        None,
        false,
    );
    let bias = T::load(
        bias_ptr.add_offsets(lane),
        Some(m),
        Some(zeros),
        &[],
        None,
        None,
        None,
        false,
    );

    let one = T::full(&[BLOCK], D::from_f64(1.0));
    let scores = one / (one + T::exp(zeros - logits));

    // Lanes past the expert count, and already-chosen experts, are pushed below any
    // real score so argmax never returns them.
    let sink = T::full(&[BLOCK], D::from_f64(-1.0e30));
    let mut pool = T::where_(m, scores + bias, sink);

    let mut wvec = T::zeros::<D>(&[BLOCK]);
    let mut ivec = T::zeros::<D>(&[BLOCK]);
    let mut total = T::zeros::<D>(&[BLOCK]);

    // K is a const generic, so this unrolls.
    for i in 0..K {
        let idx = T::argmax(pool, 0, true, true);
        let chosen = T::eq(lane, idx);
        // The weight comes from the UNBIASED scores: a masked sum picks the one lane.
        let w = T::sum(T::where_(chosen, scores, zeros), Some(0), true);
        let slot = lane.eq(i);
        wvec = T::where_(slot, w, wvec);
        ivec = T::where_(slot, T::cast::<i32, D>(idx, None, false), ivec);
        total = total + w;
        pool = T::where_(chosen, sink, pool);
    }

    let eps = T::full(&[BLOCK], D::from_f64(1.0e-20));
    let scale = T::full(&[BLOCK], D::from_f64(SCALE as f64));
    let keep = lane.lt(K);
    T::store(
        out_ptr.add_offsets(lane + row * 2 * K),
        wvec / (total + eps) * scale,
        Some(keep),
        &[],
        None,
        None,
    );
    T::store(
        out_ptr.add_offsets(lane + K + row * 2 * K),
        ivec,
        Some(keep),
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

/// Graph-level top-k router: `[T, experts]` of logits to `[T, 2 * top_k]` of weights
/// followed by expert ids.
pub struct TopKRouterOp<D: FloatBytes + Send + Sync + 'static> {
    /// Experts to choose from.
    pub experts: usize,
    /// Experts that fire per token.
    pub top_k: usize,
    /// `route_scale`, which multiplies the normalised weights.
    pub scale: f32,
    block: i32,
    _phantom: PhantomData<D>,
}

impl<D: FloatBytes + Send + Sync + 'static> TopKRouterOp<D> {
    /// A router picking `top_k` of `experts`, scaling the weights by `scale`.
    pub fn new(experts: usize, top_k: usize, scale: f32) -> Self {
        assert!(
            top_k > 0 && top_k <= experts,
            "router: {top_k} of {experts} experts is not a valid choice"
        );
        Self {
            experts,
            top_k,
            scale,
            block: experts.next_power_of_two() as i32,
            _phantom: PhantomData,
        }
    }
}

impl<D: FloatBytes + Send + Sync + 'static> CustomOp for TopKRouterOp<D> {
    fn name(&self) -> &str {
        "llm.topk_router"
    }

    fn infer_output_shape(&self, input_shapes: &[&Shape]) -> Shape {
        vec![input_shapes[0][0], Some(2 * self.top_k)]
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn lower(&self) -> Option<(String, String, String, Arc<dyn RuntimeOp>)> {
        let kernel = TopkRouterForward::<D>::new(self.block, self.top_k as i32);
        Some((
            "topk_router_forward".to_string(),
            kernel.source,
            "entry_point".to_string(),
            Arc::new(TopKRouterRuntimeOp::<D> {
                experts: self.experts,
                scale: self.scale,
                _phantom: PhantomData,
            }),
        ))
    }
}

/// Runtime dispatch for [`TopKRouterOp`].
pub struct TopKRouterRuntimeOp<D: FloatBytes + Send + Sync + 'static> {
    experts: usize,
    scale: f32,
    _phantom: PhantomData<D>,
}

impl<D: FloatBytes + Send + Sync + 'static> RuntimeOp for TopKRouterRuntimeOp<D> {
    fn n_activation_inputs(&self) -> usize {
        1
    }

    fn param_shapes(&self, _input_shapes: &[&[usize]], _output_shape: &[usize]) -> Vec<Vec<usize>> {
        vec![vec![self.experts]]
    }

    fn param_names(&self) -> &'static [&'static str] {
        &["bias"]
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
        assert_eq!(inputs[0].1[1], self.experts, "router: logit width");
        visitor.visit_ptr(inputs[0].0);
        visitor.visit_ptr(params[0]);
        visitor.visit_ptr(output);
        visitor.visit_i32(self.experts as i32);
        visitor.visit_f32(self.scale);
    }

    fn grid(&self, output_shape: &[usize]) -> [u32; 3] {
        [output_shape[0] as u32, 1, 1]
    }
}
