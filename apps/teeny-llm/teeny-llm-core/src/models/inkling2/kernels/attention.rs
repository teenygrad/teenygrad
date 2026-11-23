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

//! Inkling-2's attention: grouped-query attention over a KV cache, with a relative
//! position bias added to the logits and a per-query **attention sink**.
//!
//! Same cache discipline as [`crate::kernels::attention`] (which this is derived
//! from): a step processes `T` new tokens at positions `P .. P + T`, keys `j < P` come
//! from the cache and the rest from this step. Two things differ.
//!
//! **The bias.** [`super::rel_bias`] produces `[T * H, KV]`, one trained
//! bias-vs-distance profile per query and head, already gathered onto key positions.
//! It is added to each logit before the softmax:
//!
//! ```text
//! s[t, h, j] = q[t, h] . k[j, kvh] * scale + bias[t * H + h, j]
//! ```
//!
//! **The sink.** Each (token, head) has a learned logit from `sink_proj` that takes
//! part in the softmax but carries **no value vector**, so it absorbs probability mass
//! and the attention weights deliberately sum to less than one. That is why the
//! reference cannot use SDPA or flash attention.
//!
//! The sink needs no special case in the online softmax: it is exactly a key whose
//! value is zero, so seeding the running maximum with it and the running sum with
//! `exp(sink - sink) = 1` is the whole implementation. Gemma's version starts from
//! `-1e30` and `0`, which is the same expression with no sink present.
//!
//! ```text
//! out[t, h] = sum_j softmax_j(s, including the valueless sink) v[j, kvh]
//! ```
//!
//! **Layout**: `q` `[T, H * HD]`; `k_new`, `v_new` `[T, KVH * HD]`; `k_cache`,
//! `v_cache` `[S, KVH * HD]` with rows `0 .. P` valid; `bias` `[T * H, KV]` with
//! `KV = P + T`; `sinks` `[T, H]`; output `[T, H * HD]`.

use teeny_core::dtype::{AddOffsets, Comparison, Float, Tensor};
use teeny_macros::kernel;
use teeny_triton::triton::*;

/// The attention above.
#[allow(clippy::multiple_bound_locations)]
#[allow(clippy::too_many_arguments)]
#[kernel]
pub fn sink_attention_forward<T: Triton, D: Float, const BLOCK: i32>(
    q_ptr: InPtr<T::Pointer<D>>,
    kn_ptr: InPtr<T::Pointer<D>>,
    vn_ptr: InPtr<T::Pointer<D>>,
    kc_ptr: InPtr<T::Pointer<D>>,
    vc_ptr: InPtr<T::Pointer<D>>,
    bias_ptr: InPtr<T::Pointer<D>>,
    sink_ptr: InPtr<T::Pointer<D>>,
    o_ptr: OutPtr<T::Pointer<D>>,
    H: i32,
    KV: i32,
    KVH: i32,
    HD: i32,
    P: i32,
    LO: i32,
    W: i32,
    scale: f32,
) where
    T: Triton<I32Tensor = <T as Triton>::Tensor<i32>>,
    T::I32Tensor: Tensor<i32, 1>,
    T::I32Tensor: core::ops::Add<i32, Output = T::I32Tensor>,
    T::I32Tensor: core::ops::Mul<i32, Output = T::I32Tensor>,
    T::I32Tensor: core::ops::Add<T::I32Tensor, Output = T::I32Tensor>,
    T::I32Tensor: Comparison<i32, BoolTensor = T::BoolTensor>,
    T::Pointer<D>: AddOffsets<i32, 1, T::I32Tensor, Output = T::Tensor<T::Pointer<D>>>,
{
    let pid = T::program_id(Axis::X);
    let t = pid / H;
    let h = pid % H;
    let kvh = h / (H / KVH);
    let qpos = P + t;
    let kv_row = KVH * HD;

    let d = T::arange(0, BLOCK);
    let dm = d.lt(HD);
    let zeros = T::zeros::<D>(&[BLOCK]);
    let q = T::load(
        q_ptr.add_offsets(d + (t * H + h) * HD),
        Some(dm),
        Some(zeros),
        &[],
        None,
        None,
        None,
        false,
    ) * T::full(&[BLOCK], D::from_f64(scale as f64));

    // Keys this query sees: causal (j <= qpos) and inside the window (qpos - j < W).
    // `LO` (from the host) is the first key query 0 sees; later queries of a prompt
    // chunk start up to `T - 1` keys later, and those keys are masked. Keys `j < P` come
    // from the cache, the rest from this step's `k_new` / `v_new`; the other load is
    // masked off. (One loop: teenyc's RISC-V codegen mishandles two in a row here.)
    let first = qpos - W;
    let ione = T::zeros::<i32>(&[1]);
    let izero = T::zeros::<i32>(&[BLOCK]);

    // Online softmax with a one-element running max and sum: one exp per key, not one
    // per lane.
    // The sink is a key with a zero value vector, so it simply seeds the running
    // maximum; its own weight is exp(sink - sink) = 1 in the running sum, and it
    // contributes nothing to `acc`. Everything below is unchanged by its presence.
    let zero1 = T::zeros::<D>(&[1]);
    let sink = T::load(
        sink_ptr.add_offsets(T::zeros::<i32>(&[1]) + (t * H + h)),
        None,
        Some(zero1),
        &[],
        None,
        None,
        None,
        false,
    );
    let mut m_run = sink;
    let mut l_run = T::full(&[1], D::from_f64(1.0));
    let mut acc = T::zeros::<D>(&[BLOCK]);
    // A loop from 0: for `LO..qpos + 1`, teenyc runs `qpos + 1` iterations starting at
    // `LO`, overrunning the query's own key once `LO > 0` (past the window).
    for jj in 0..(qpos + 1 - LO) {
        let j = LO + jj;
        let jt = izero + j;
        let from_cache = jt.lt(P);
        let from_new = jt.ge(P);
        let jc = T::minimum(jt, izero + (P - 1));
        let jn = T::maximum(jt + (-P), izero);
        let k = T::load(
            kc_ptr.add_offsets(d + jc * kv_row + kvh * HD),
            Some(from_cache),
            Some(zeros),
            &[],
            None,
            None,
            None,
            false,
        ) + T::load(
            kn_ptr.add_offsets(d + jn * kv_row + kvh * HD),
            Some(from_new),
            Some(zeros),
            &[],
            None,
            None,
            None,
            false,
        );
        let v = T::load(
            vc_ptr.add_offsets(d + jc * kv_row + kvh * HD),
            Some(from_cache),
            Some(zeros),
            &[],
            None,
            None,
            None,
            false,
        ) + T::load(
            vn_ptr.add_offsets(d + jn * kv_row + kvh * HD),
            Some(from_new),
            Some(zeros),
            &[],
            None,
            None,
            None,
            false,
        );
        let bj = T::load(
            bias_ptr.add_offsets(ione + ((t * H + h) * KV + j)),
            None,
            Some(zero1),
            &[],
            None,
            None,
            None,
            false,
        );
        let s = T::sum(q * k, Some(0), true) + bj;
        let seen = (ione + j).gt(first);
        let m_new = T::maximum(m_run, T::where_(seen, s, m_run));
        let p = T::where_(seen, T::exp(s - m_new), T::zeros::<D>(&[1]));
        let alpha = T::exp(m_run - m_new);
        l_run = l_run * alpha + p;
        acc = acc * (alpha + zeros) + (p + zeros) * v;
        m_run = m_new;
    }
    T::store(
        o_ptr.add_offsets(d + (t * H + h) * HD),
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

// The geometry and the shared past-length cell come from the common kernel: only
// the bias and the sink differ here.
pub use crate::kernels::attention::{AttentionSpec, PastLen};

/// Graph-level attention: inputs `q`, `k_new`, `v_new`, `k_cache`, `v_cache`,
/// `bias`, `sinks` (see the module docs); output shaped as `q`. No weights.
pub struct SinkAttentionOp<D: FloatBytes + Send + Sync + 'static> {
    spec: AttentionSpec,
    past: PastLen,
    _phantom: PhantomData<D>,
}

impl<D: FloatBytes + Send + Sync + 'static> SinkAttentionOp<D> {
    /// Creates the op, reading the past length from `past` at each launch.
    pub fn new(spec: AttentionSpec, past: PastLen) -> Self {
        assert!(
            spec.heads.is_multiple_of(spec.kv_heads),
            "attention: {} heads not a multiple of {} kv heads",
            spec.heads,
            spec.kv_heads
        );
        Self {
            spec,
            past,
            _phantom: PhantomData,
        }
    }
}

impl<D: FloatBytes + Send + Sync + 'static> CustomOp for SinkAttentionOp<D> {
    fn name(&self) -> &str {
        "llm.sink_attention"
    }

    fn infer_output_shape(&self, input_shapes: &[&Shape]) -> Shape {
        input_shapes[0].clone()
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn lower(&self) -> Option<(String, String, String, Arc<dyn RuntimeOp>)> {
        let block = self.spec.head_dim.next_power_of_two() as i32;
        let kernel = SinkAttentionForward::<D>::new(block);
        Some((
            "sink_attention_forward".to_string(),
            kernel.source,
            "entry_point".to_string(),
            Arc::new(SinkAttentionRuntimeOp::<D> {
                spec: self.spec,
                past: self.past.clone(),
                _phantom: PhantomData,
            }),
        ))
    }
}

/// Runtime dispatch for [`SinkAttentionOp`].
pub struct SinkAttentionRuntimeOp<D: FloatBytes + Send + Sync + 'static> {
    spec: AttentionSpec,
    past: PastLen,
    _phantom: PhantomData<D>,
}

impl<D: FloatBytes + Send + Sync + 'static> RuntimeOp for SinkAttentionRuntimeOp<D> {
    fn n_activation_inputs(&self) -> usize {
        7
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
        let s = &self.spec;
        let past = self.past.get();
        let cache_rows = inputs[3].1[0];
        assert!(
            past <= cache_rows,
            "attention: past length {past} exceeds the {cache_rows}-row KV cache"
        );
        let kv = inputs[5].1[1];
        assert_eq!(
            kv,
            past + output_shape[0],
            "attention: a bias of {kv} keys does not match {past} cached plus {} new",
            output_shape[0]
        );
        for (ptr, _) in &inputs[..7] {
            visitor.visit_ptr(*ptr);
        }
        visitor.visit_ptr(output);
        visitor.visit_i32(s.heads as i32);
        visitor.visit_i32(kv as i32);
        visitor.visit_i32(s.kv_heads as i32);
        visitor.visit_i32(s.head_dim as i32);
        visitor.visit_i32(past as i32);
        let window = s.window.map_or(i32::MAX / 2, |w| w as i32);
        assert!(
            output_shape[0] <= window as usize,
            "attention: a {}-token step is longer than the {window}-token window",
            output_shape[0]
        );
        // The first key query 0 (position `past`) sees.
        visitor.visit_i32((past as i64 - window as i64 + 1).max(0) as i32);
        visitor.visit_i32(window);
        visitor.visit_f32(s.scale);
    }

    fn grid(&self, output_shape: &[usize]) -> [u32; 3] {
        [(output_shape[0] * self.spec.heads) as u32, 1, 1]
    }
}
