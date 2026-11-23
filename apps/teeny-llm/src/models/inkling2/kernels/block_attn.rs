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

//! Inkling-2's block attention residual (`Inkling2LoopBlockAttentionResidual` and
//! `Inkling2LoopMessageAttentionResidual`): each token retrieves a message from the
//! block states accumulated so far, using **learned queries** shared by every token.
//!
//! ```text
//! out[t, q] = sum_s softmax_s(query[q] . states[t, s] * hidden^-0.5) states[t, s]
//! ```
//!
//! One head, `head_dim = hidden`, no mask and no causality: the `S` states of a token
//! are block boundaries it has already passed, not a sequence. `S` grows by one at
//! each block boundary, and the partial state of the current block is the last entry.
//!
//! The queries are a parameter of `[Q, hidden]`, the same at every position; the
//! checkpoint stores them centrally as `attn_res.attn_queries.{layer}` (`Q = 3`) and
//! `attn_res.mlp_queries.{layer}` (`Q = 1`).
//!
//! The caller normalises `states` first, with
//! [`super::norm::UnweightedRmsNormOp`] over rows of `hidden`: the reference norms
//! the stack once and uses it as both keys and values.
//!
//! Grid: one program per `(token, query)`; `BLOCK` must be at least `hidden`.

use teeny_core::dtype::{AddOffsets, Comparison, Float, Tensor};
use teeny_macros::kernel;
use teeny_triton::triton::*;

/// The block attention above.
#[allow(clippy::multiple_bound_locations)]
#[kernel]
pub fn block_attn_forward<T: Triton, D: Float, const BLOCK: i32>(
    s_ptr: InPtr<T::Pointer<D>>,
    q_ptr: InPtr<T::Pointer<D>>,
    y_ptr: OutPtr<T::Pointer<D>>,
    S: i32,
    H: i32,
    Q: i32,
    scale: f32,
) where
    T: Triton<I32Tensor = <T as Triton>::Tensor<i32>>,
    T::I32Tensor: Tensor<i32, 1>,
    T::I32Tensor: core::ops::Add<i32, Output = T::I32Tensor>,
    T::I32Tensor: core::ops::Mul<i32, Output = T::I32Tensor>,
    T::I32Tensor: Comparison<i32, BoolTensor = T::BoolTensor>,
    T::Pointer<D>: AddOffsets<i32, 1, T::I32Tensor, Output = T::Tensor<T::Pointer<D>>>,
{
    let pid = T::program_id(Axis::X);
    let token = pid / Q;
    let qi = pid - token * Q;

    let d = T::arange(0, BLOCK);
    let dm = d.lt(H);
    let zeros = T::zeros::<D>(&[BLOCK]);
    let q = T::load(
        q_ptr.add_offsets(d + qi * H),
        Some(dm),
        Some(zeros),
        &[],
        None,
        None,
        None,
        false,
    ) * T::full(&[BLOCK], D::from_f64(scale as f64));

    // Online softmax over the S states, as the attention kernel does over keys.
    let mut m_run = T::full(&[1], D::from_f64(-1.0e30));
    let mut l_run = T::zeros::<D>(&[1]);
    let mut acc = T::zeros::<D>(&[BLOCK]);
    for s in 0..S {
        let st = T::load(
            s_ptr.add_offsets(d + (token * S + s) * H),
            Some(dm),
            Some(zeros),
            &[],
            None,
            None,
            None,
            false,
        );
        let logit = T::sum(q * st, Some(0), true);
        let m_new = T::maximum(m_run, logit);
        let p = T::exp(logit - m_new);
        let alpha = T::exp(m_run - m_new);
        l_run = l_run * alpha + p;
        acc = acc * (alpha + zeros) + (p + zeros) * st;
        m_run = m_new;
    }
    T::store(
        y_ptr.add_offsets(d + pid * H),
        acc / (l_run + zeros),
        Some(dm),
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

/// Graph-level block attention: input `states` `[T * S, hidden]` (already normalised),
/// a `[Q, hidden]` query parameter, output `[T * Q, hidden]`.
pub struct BlockAttnOp<D: FloatBytes + Send + Sync + 'static> {
    /// Block states per token, including the current partial one.
    pub states: usize,
    /// Model width.
    pub hidden: usize,
    /// Learned queries.
    pub queries: usize,
    block: i32,
    _phantom: PhantomData<D>,
}

impl<D: FloatBytes + Send + Sync + 'static> BlockAttnOp<D> {
    /// Attention from `queries` learned queries over `states` states of `hidden` lanes.
    pub fn new(states: usize, hidden: usize, queries: usize) -> Self {
        assert!(
            states > 0 && queries > 0,
            "block attn: states and queries must be positive"
        );
        Self {
            states,
            hidden,
            queries,
            block: hidden.next_power_of_two() as i32,
            _phantom: PhantomData,
        }
    }
}

impl<D: FloatBytes + Send + Sync + 'static> CustomOp for BlockAttnOp<D> {
    fn name(&self) -> &str {
        "llm.block_attn"
    }
    fn infer_output_shape(&self, input_shapes: &[&Shape]) -> Shape {
        vec![
            input_shapes[0][0].map(|rows| rows / self.states * self.queries),
            Some(self.hidden),
        ]
    }
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn lower(&self) -> Option<(String, String, String, Arc<dyn RuntimeOp>)> {
        let kernel = BlockAttnForward::<D>::new(self.block);
        Some((
            "block_attn_forward".to_string(),
            kernel.source,
            "entry_point".to_string(),
            Arc::new(BlockAttnRuntimeOp::<D> {
                states: self.states,
                hidden: self.hidden,
                queries: self.queries,
                _phantom: PhantomData,
            }),
        ))
    }
}

/// Runtime dispatch for [`BlockAttnOp`].
pub struct BlockAttnRuntimeOp<D: FloatBytes + Send + Sync + 'static> {
    states: usize,
    hidden: usize,
    queries: usize,
    _phantom: PhantomData<D>,
}

impl<D: FloatBytes + Send + Sync + 'static> RuntimeOp for BlockAttnRuntimeOp<D> {
    fn n_activation_inputs(&self) -> usize {
        1
    }
    fn param_shapes(&self, _i: &[&[usize]], _o: &[usize]) -> Vec<Vec<usize>> {
        vec![vec![self.queries, self.hidden]]
    }
    fn param_names(&self) -> &'static [&'static str] {
        &["query"]
    }
    fn pack_args(
        &self,
        inputs: &[(RawPtr, &[usize])],
        params: &[RawPtr],
        output: RawPtr,
        output_shape: &[usize],
        _output_row_stride: i32,
        visitor: &mut dyn ArgVisitor,
    ) {
        assert_eq!(inputs[0].1[1], self.hidden, "block attn: state width");
        visitor.visit_ptr(inputs[0].0);
        visitor.visit_ptr(params[0]);
        visitor.visit_ptr(output);
        visitor.visit_i32(self.states as i32);
        visitor.visit_i32(self.hidden as i32);
        visitor.visit_i32(self.queries as i32);
        // The reference scales by hidden ** -0.5, the full model width, not by a
        // per-head dimension: this attention has one head as wide as the model.
        visitor.visit_f32((self.hidden as f32).powf(-0.5));
        let _ = output_shape;
    }
    fn grid(&self, output_shape: &[usize]) -> [u32; 3] {
        [output_shape[0] as u32, 1, 1]
    }
}
