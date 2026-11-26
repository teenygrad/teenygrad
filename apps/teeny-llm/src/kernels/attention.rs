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

//! Gemma 3 grouped-query attention over a KV cache, for prompt processing and
//! token-by-token decoding alike.
//!
//! A step processes `T` new tokens at positions `P .. P + T`, where `P` tokens are
//! already in the cache. Query `t` (position `P + t`) attends to key position `j`
//! when `j <= P + t` (causal) and, on sliding-window layers, `P + t - j < W`. Keys
//! `j < P` come from the cache, keys `j >= P` from this step's `k_new` / `v_new`.
//!
//! ```text
//! out[t, h] = sum_j softmax_j(q[t, h] . k[j, kvh] * scale) v[j, kvh],  kvh = h / (H / KVH)
//! ```
//!
//! `P` is a launch argument read from a shared [`PastLen`], which the KV cache keeps
//! at its length. Each query loops only over the keys it can see (causal, and inside the
//! window). The softmax is online, with a one-element running max and sum so each key
//! costs one `exp`; one program per `(t, h)`, lanes on `head_dim`.
//!
//! **Layout**: `q` `[T, H * HD]`; `k_new`, `v_new` `[T, KVH * HD]`; `k_cache`,
//! `v_cache` `[S, KVH * HD]` with rows `0 .. P` valid; output `[T, H * HD]`.

use teeny_core::dtype::{AddOffsets, Comparison, Float, Tensor};
use teeny_macros::kernel;
use teeny_triton::triton::*;

/// The attention above.
#[allow(clippy::multiple_bound_locations)]
#[allow(clippy::too_many_arguments)]
#[kernel]
pub fn gqa_attention_forward<T: Triton, D: Float, const BLOCK: i32>(
    q_ptr: InPtr<T::Pointer<D>>,
    kn_ptr: InPtr<T::Pointer<D>>,
    vn_ptr: InPtr<T::Pointer<D>>,
    kc_ptr: InPtr<T::Pointer<D>>,
    vc_ptr: InPtr<T::Pointer<D>>,
    o_ptr: OutPtr<T::Pointer<D>>,
    H: i32,
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
    let mut m_run = T::full(&[1], D::from_f64(-1.0e30));
    let mut l_run = T::zeros::<D>(&[1]);
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
        let s = T::sum(q * k, Some(0), true);
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

/// The number of positions already in the KV cache: the cache keeps it up to date, and
/// every attention op of a compiled step graph passes it to its kernel.
pub use teeny_kvcache::PastLen;

/// Attention geometry.
#[derive(Clone, Copy, Debug)]
pub struct AttentionSpec {
    /// Query heads.
    pub heads: usize,
    /// Key/value heads (`heads` a multiple of it).
    pub kv_heads: usize,
    /// Lanes per head.
    pub head_dim: usize,
    /// Sliding window, or `None` for full attention.
    pub window: Option<usize>,
    /// Logit scale (`query_pre_attn_scalar ** -0.5`).
    pub scale: f32,
}

/// Graph-level attention: inputs `q`, `k_new`, `v_new`, `k_cache`, `v_cache` (see
/// the module docs); output shaped as `q`. No weights.
pub struct GqaAttentionOp<D: FloatBytes + Send + Sync + 'static> {
    spec: AttentionSpec,
    past: PastLen,
    _phantom: PhantomData<D>,
}

impl<D: FloatBytes + Send + Sync + 'static> GqaAttentionOp<D> {
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

impl<D: FloatBytes + Send + Sync + 'static> CustomOp for GqaAttentionOp<D> {
    fn name(&self) -> &str {
        "llm.gqa_attention"
    }

    fn infer_output_shape(&self, input_shapes: &[&Shape]) -> Shape {
        input_shapes[0].clone()
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn lower(&self) -> Option<(String, String, String, Arc<dyn RuntimeOp>)> {
        let block = self.spec.head_dim.next_power_of_two() as i32;
        let kernel = GqaAttentionForward::<D>::new(block);
        Some((
            "gqa_attention_forward".to_string(),
            kernel.source,
            "entry_point".to_string(),
            Arc::new(GqaAttentionRuntimeOp::<D> {
                spec: self.spec,
                past: self.past.clone(),
                _phantom: PhantomData,
            }),
        ))
    }
}

/// Runtime dispatch for [`GqaAttentionOp`].
pub struct GqaAttentionRuntimeOp<D: FloatBytes + Send + Sync + 'static> {
    spec: AttentionSpec,
    past: PastLen,
    _phantom: PhantomData<D>,
}

impl<D: FloatBytes + Send + Sync + 'static> RuntimeOp for GqaAttentionRuntimeOp<D> {
    fn n_activation_inputs(&self) -> usize {
        5
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
        for (ptr, _) in &inputs[..5] {
            visitor.visit_ptr(*ptr);
        }
        visitor.visit_ptr(output);
        visitor.visit_i32(s.heads as i32);
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
