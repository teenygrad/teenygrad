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
//! `P` is a launch argument read from a shared [`PastLen`] the host sets before each
//! run, so the key loop covers only `P + T` keys however large the cache buffer is.
//! The softmax is online (running max and sum), one program per `(t, h)`, lanes on
//! `head_dim`.
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
pub fn gemma_attention_forward<T: Triton, D: Float, const BLOCK: i32>(
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
    TT: i32,
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
    let izero = T::zeros::<i32>(&[BLOCK]);
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

    let mut m_run = T::full(&[BLOCK], D::from_f64(-1.0e30));
    let mut l_run = T::zeros::<D>(&[BLOCK]);
    let mut acc = T::zeros::<D>(&[BLOCK]);
    for j in 0..P + TT {
        let jt = izero + j;
        // Key j from the cache (j < P) or this step's new keys (j >= P); the other
        // load is masked off and contributes zero.
        let from_cache = jt.lt(P);
        let from_new = jt.ge(P);
        let jc = T::minimum(jt, izero + (P - 1));
        let jc = T::maximum(jc, izero);
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
        let s = T::sum(q * k, Some(0), true) + zeros;
        // Causal (dist = qpos - j >= 0) and window (dist < W, i.e. W - 1 - dist >= 0),
        // folded into one comparison: min(dist, W - 1 - dist) >= 0.
        let dist = jt * (-1) + qpos;
        let valid = T::minimum(dist, dist * (-1) + (W - 1)).ge(0);
        let m_new = T::maximum(m_run, T::where_(valid, s, m_run));
        let p = T::where_(valid, T::exp(s - m_new), zeros);
        let alpha = T::exp(m_run - m_new);
        l_run = l_run * alpha + p;
        acc = acc * alpha + p * v;
        m_run = m_new;
    }
    T::store(
        o_ptr.add_offsets(d + (t * H + h) * HD),
        acc / l_run,
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
use std::sync::atomic::{AtomicUsize, Ordering};
use teeny_core::device::program::ArgVisitor;
use teeny_core::dtype::FloatBytes;
use teeny_core::graph::{CustomOp, Shape};
use teeny_core::model::{RawPtr, RuntimeOp};

/// The number of tokens already in the KV cache, shared between the host loop (which
/// sets it before each run) and every attention op of a compiled step graph (which
/// pass it to their kernels).
#[derive(Clone, Default, Debug)]
pub struct PastLen(Arc<AtomicUsize>);

impl PastLen {
    /// Sets the past length for the next run.
    pub fn set(&self, n: usize) {
        self.0.store(n, Ordering::SeqCst);
    }

    /// The current past length.
    pub fn get(&self) -> usize {
        self.0.load(Ordering::SeqCst)
    }
}

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
pub struct GemmaAttentionOp<D: FloatBytes + Send + Sync + 'static> {
    spec: AttentionSpec,
    past: PastLen,
    _phantom: PhantomData<D>,
}

impl<D: FloatBytes + Send + Sync + 'static> GemmaAttentionOp<D> {
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

impl<D: FloatBytes + Send + Sync + 'static> CustomOp for GemmaAttentionOp<D> {
    fn name(&self) -> &str {
        "gemma.attention"
    }

    fn infer_output_shape(&self, input_shapes: &[&Shape]) -> Shape {
        input_shapes[0].clone()
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn lower(&self) -> Option<(String, String, String, Arc<dyn RuntimeOp>)> {
        let block = self.spec.head_dim.next_power_of_two() as i32;
        let kernel = GemmaAttentionForward::<D>::new(block);
        Some((
            "gemma_attention_forward".to_string(),
            kernel.source,
            "entry_point".to_string(),
            Arc::new(GemmaAttentionRuntimeOp::<D> {
                spec: self.spec,
                past: self.past.clone(),
                _phantom: PhantomData,
            }),
        ))
    }
}

/// Runtime dispatch for [`GemmaAttentionOp`].
pub struct GemmaAttentionRuntimeOp<D: FloatBytes + Send + Sync + 'static> {
    spec: AttentionSpec,
    past: PastLen,
    _phantom: PhantomData<D>,
}

impl<D: FloatBytes + Send + Sync + 'static> RuntimeOp for GemmaAttentionRuntimeOp<D> {
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
        visitor.visit_i32(output_shape[0] as i32);
        visitor.visit_i32(s.window.map_or(i32::MAX / 2, |w| w as i32));
        visitor.visit_f32(s.scale);
    }

    fn grid(&self, output_shape: &[usize]) -> [u32; 3] {
        [(output_shape[0] * self.spec.heads) as u32, 1, 1]
    }
}
