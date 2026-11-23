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

//! Inkling-2's mixture-of-experts block as a teeny graph.
//!
//! **Scope.** This is the MoE half of a decoder layer, not yet a whole step graph.
//! The attention half is blocked on a kernel that does not exist: Inkling-2 adds a
//! relative position bias to the attention logits and gives every query a learned
//! *sink* logit that absorbs probability mass, so its attention weights do not sum to
//! one. [`crate::kernels::attention::AttentionSpec`] has neither, which is also why
//! the reference sets `_supports_sdpa = False`. See the module-level notes in
//! [`super`] for what remains.
//!
//! What this builds is the whole of `Inkling2LoopSparseMoeBlock`:
//!
//! ```text
//! logits   = x @ gate.weight^T                         [T, experts]
//! routed   = topk_router(logits, gate.bias)            [T, 2 * top_k]
//! gate_up  = expert_gather(x, w13, routed)             [T * top_k, 2 * inter]
//! act      = swiglu(gate_up)                           [T * top_k, inter]
//! down     = expert_gather(act, w2, slot_ids)          [T * top_k, hidden]
//! out      = scatter_add(down, routed)                 [T, hidden]
//! ```
//!
//! Weight slots are named by their `state_dict` keys, as the Gemma port does, so
//! checkpoint tensors load by name: `mlp.gate.weight`, `mlp.gate.bias`,
//! `mlp.experts.w13_weight`, `mlp.experts.w2_weight`.
//!
//! The second gather is the same op with `top_k = 1`: by then every row already *is*
//! one (token, slot) pair, so each needs exactly the expert it was routed to. The ids
//! for it are re-packed by [`slot_ids`], which is the one place the two gathers do
//! not simply chain.

use teeny_core::dtype::FloatBytes;
use teeny_core::graph::{CustomData, SymTensor};
use teeny_core::name_scope::name_scope;

use super::config::Inkling2TextConfig;
use super::kernels::{
    attention::{AttentionSpec, PastLen, SinkAttentionOp},
    block_attn::BlockAttnOp,
    cat2::Cat2Op,
    expert_gather::ExpertGatherOp,
    gated_mul::GatedMulOp,
    l2_clip::{L2ClipOp, ScaledL2ClipOp},
    mix::MessageMixOp,
    norm::{RmsNormOp, UnweightedRmsNormOp},
    rel_bias::RelBiasOp,
    router::TopKRouterOp,
    scatter_add::ExpertScatterAddOp,
    sconv::SconvOp,
    softcap::SoftcapOp,
    swiglu::SwigluOp,
};
use crate::kernels::linear::Linear4Op;

/// Block width for the linear kernel, as in the Gemma port.
const LINEAR_BLOCK: i32 = 16;

/// Records a bias-free linear in the current scope, naming its slot `weight`.
fn linear<D: FloatBytes + Send + Sync + 'static>(x: &SymTensor, k: usize, n: usize) -> SymTensor {
    x.record_custom(
        CustomData::new(Linear4Op::<D>::new(k, n, LINEAR_BLOCK)),
        &[],
        None,
    )
}

/// Re-packs the router's ids so the down projection can be gathered with `top_k = 1`.
///
/// The router emits one row per token holding `top_k` weights then `top_k` ids. After
/// the first gather there is one row per (token, slot), so the down projection wants
/// one id per row. This produces `[T * top_k, 2]`: an unused weight lane, then the id,
/// which is the layout [`ExpertGatherOp`] reads with `top_k = 1`.
///
/// It is host-side arithmetic on a tensor the host already has, so it is a plain
/// reshape of the routing decision rather than a kernel.
pub fn slot_ids(routed: &[f32], tokens: usize, top_k: usize) -> Vec<f32> {
    let mut out = vec![0.0; tokens * top_k * 2];
    for t in 0..tokens {
        for s in 0..top_k {
            out[(t * top_k + s) * 2 + 1] = routed[t * 2 * top_k + top_k + s];
        }
    }
    out
}

/// Records `Inkling2LoopSparseMoeBlock` on `x` of `[T, hidden]`, returning `[T, hidden]`.
///
/// `routed` is the router's packed `[T, 2 * top_k]` row and `ids` its re-packed form
/// from [`slot_ids`]; both are graph inputs because the second gather needs the ids in
/// a different layout from the one the router emits.
pub fn build_moe<D: FloatBytes + Send + Sync + 'static>(
    cfg: &Inkling2TextConfig,
    x: &SymTensor,
    routed: &SymTensor,
    ids: &SymTensor,
) -> SymTensor {
    let _m = name_scope("mlp");
    let (hid, inter) = (cfg.hidden_size, cfg.intermediate_size);
    let (e, k) = (cfg.n_routed_experts, cfg.num_experts_per_tok);

    let gate_up = {
        let _g = name_scope("experts");
        x.record_custom(
            CustomData::new(ExpertGatherOp::<D>::new(e, 2 * inter, hid, k)),
            &[routed],
            None,
        )
    };
    let act = gate_up.record_custom(
        CustomData::new(SwigluOp::<D>::new(
            inter,
            cfg.mlp_gate_softcap as f32,
            cfg.mlp_up_softcap as f32,
        )),
        &[],
        None,
    );
    let down = {
        let _g = name_scope("experts_down");
        act.record_custom(
            CustomData::new(ExpertGatherOp::<D>::new(e, hid, inter, 1)),
            &[ids],
            None,
        )
    };
    down.record_custom(
        CustomData::new(ExpertScatterAddOp::<D>::new(hid, k)),
        &[routed],
        None,
    )
}

/// Records the router on `x` of `[T, hidden]`, returning the packed `[T, 2 * top_k]`.
///
/// Kept separate from [`build_moe`] because the host needs the routing decision in
/// order to build the down projection's ids, so the two run as separate graphs for
/// now. Folding them together needs either a second graph output or a kernel that
/// re-packs on the device.
pub fn build_router<D: FloatBytes + Send + Sync + 'static>(
    cfg: &Inkling2TextConfig,
    x: &SymTensor,
) -> SymTensor {
    let _m = name_scope("mlp");
    // Both slots live under `mlp.gate`: the projection's `weight` and the router's
    // `bias`, matching the checkpoint's mlp.gate.weight and mlp.gate.bias.
    let _g = name_scope("gate");
    let logits = linear::<D>(x, cfg.hidden_size, cfg.n_routed_experts);
    logits.record_custom(
        CustomData::new(TopKRouterOp::<D>::new(
            cfg.n_routed_experts,
            cfg.num_experts_per_tok,
            cfg.route_scale as f32,
        )),
        &[],
        None,
    )
}

// ---------------------------------------------------------------------------
// Attention projections
// ---------------------------------------------------------------------------
//
// The slot names below are the CHECKPOINT's, not the reference implementation's.
// `thinkingmachines/Inkling-2-Mini` ships the original names, which transformers
// renames on load:
//
// | checkpoint                 | reference                          |
// |---|---|
// | `attn.wq_du.weight`        | `self_attn.q_proj.linear`          |
// | `attn.wk_dv.weight`        | `self_attn.k_proj.linear`          |
// | `attn.wv_dv.weight`        | `self_attn.v_proj.linear`          |
// | `attn.wo_ud.weight`        | `self_attn.o_proj.linear`          |
// | `attn.gate_attn.weight`    | `self_attn.o_proj.gate_proj`       |
// | `attn.wr_du.weight`        | `relative_position_bias.linear`    |
// | `attn.q_l2_clip.scale`     | `q_proj.norm.scale`                |
// | `attn_norm.weight`         | `input_layernorm`                  |
//
// Loading by the shipped names avoids a rename pass, and the weight shapes confirm
// the mapping: wq_du is [24 * 128, 2304], wk_dv and wv_dv are [8 * 128, 2304],
// wo_ud is [2304, 24 * 128], wr_du is [24 * 16, 2304].

/// Records the query projection: linear, then the scaled L2 clip, per head.
///
/// `Inkling2LoopQueryProjection` with `log_scaling = false`, which is what the
/// sliding layers use.
pub fn build_q_proj<D: FloatBytes + Send + Sync + 'static>(
    cfg: &Inkling2TextConfig,
    x: &SymTensor,
) -> SymTensor {
    let (nh, hd) = (cfg.num_attention_heads, cfg.head_dim);
    let q = {
        let _g = name_scope("wq_du");
        linear::<D>(x, cfg.hidden_size, nh * hd)
    };
    let _g = name_scope("q_l2_clip");
    q.record_custom(
        CustomData::new(ScaledL2ClipOp::<D>::new(
            nh,
            hd,
            cfg.qk_l2_clip_threshold as f32,
        )),
        &[],
        None,
    )
}

/// Records the key or value projection: linear, short convolution, then the norm.
///
/// The two differ only in that norm: keys take an **unscaled** L2 clip, values an
/// unweighted RMS norm. Neither has a learned scale, unlike the query's clip.
///
/// `x_hist` carries the convolution's `K - 1` rows of history, as
/// [`super::kernels::sconv`] expects, so the caller keeps the conv state the same way
/// it keeps the KV cache.
pub fn build_kv_proj<D: FloatBytes + Send + Sync + 'static>(
    cfg: &Inkling2TextConfig,
    x_hist: &SymTensor,
    is_key: bool,
) -> SymTensor {
    let (nkv, hd) = (cfg.num_key_value_heads, cfg.head_dim);
    let name = if is_key { "wk_dv" } else { "wv_dv" };
    let proj = {
        let _g = name_scope(name);
        linear::<D>(x_hist, cfg.hidden_size, nkv * hd)
    };
    let conv = {
        let _g = name_scope(if is_key { "k_sconv" } else { "v_sconv" });
        proj.record_custom(
            CustomData::new(SconvOp::<D>::new(nkv * hd, cfg.sconv_kernel_size)),
            &[],
            None,
        )
    };
    if is_key {
        conv.record_custom(
            CustomData::new(L2ClipOp::<D>::new(hd, cfg.qk_l2_clip_threshold as f32)),
            &[],
            None,
        )
    } else {
        conv.record_custom(
            CustomData::new(UnweightedRmsNormOp::<D>::new(hd)),
            &[],
            None,
        )
    }
}

/// Records the gated output projection.
///
/// `Inkling2LoopGatedOutputProjection`: the attention output is multiplied by a
/// sigmoid gate computed from the **layer input**, not from the attention output,
/// and only then projected back to the model width.
pub fn build_o_proj<D: FloatBytes + Send + Sync + 'static>(
    cfg: &Inkling2TextConfig,
    attn: &SymTensor,
    x: &SymTensor,
) -> SymTensor {
    let (nh, hd) = (cfg.num_attention_heads, cfg.head_dim);
    let gate = {
        let _g = name_scope("gate_attn");
        linear::<D>(x, cfg.hidden_size, nh * hd)
    };
    let gated = attn.record_custom(CustomData::new(GatedMulOp::<D>::default()), &[&gate], None);
    let _g = name_scope("wo_ud");
    linear::<D>(&gated, nh * hd, cfg.hidden_size)
}

/// Records the layer's RMS norm (`attn_norm` or `mlp_norm`).
pub fn build_norm<D: FloatBytes + Send + Sync + 'static>(
    name: &str,
    x: &SymTensor,
    width: usize,
) -> SymTensor {
    let _g = name_scope(name);
    x.record_custom(CustomData::new(RmsNormOp::<D>::new(width)), &[], None)
}

/// Records the relative position bias: linear, scaled clip, profile projection, then
/// the gather onto key positions.
pub fn build_rel_bias<D: FloatBytes + Send + Sync + 'static>(
    cfg: &Inkling2TextConfig,
    x: &SymTensor,
    past: usize,
    kv_len: usize,
) -> SymTensor {
    let (nh, dr) = (cfg.num_attention_heads, cfg.d_rel);
    let rel = {
        let _g = name_scope("wr_du");
        linear::<D>(x, cfg.hidden_size, nh * dr)
    };
    let rel = {
        let _g = name_scope("rel_l2_clip");
        rel.record_custom(
            CustomData::new(ScaledL2ClipOp::<D>::new(
                nh,
                dr,
                cfg.rel_l2_clip_threshold as f32,
            )),
            &[],
            None,
        )
    };
    let logits = {
        let _g = name_scope("rel_logits_proj");
        linear::<D>(&rel, dr, cfg.rel_extent)
    };
    logits.record_custom(
        CustomData::new(RelBiasOp::<D>::new(nh, cfg.rel_extent, past, kv_len)),
        &[],
        None,
    )
}

/// Records the attention half of a decoder layer: norm, the three projections, the
/// relative bias, attention with sinks, and the gated output projection.
///
/// `x` is the layer input `[T, hidden]`; `x_hist` is the same with the short
/// convolution's `K - 1` rows of history prepended. The remaining inputs are the KV
/// cache and this step's keys and values, as [`super::kernels::attention`] takes them.
#[allow(clippy::too_many_arguments)]
pub fn build_attention<D: FloatBytes + Send + Sync + 'static>(
    cfg: &Inkling2TextConfig,
    x: &SymTensor,
    x_hist: &SymTensor,
    k_cache: &SymTensor,
    v_cache: &SymTensor,
    past: usize,
    kv_len: usize,
    past_len: PastLen,
    window: Option<usize>,
) -> SymTensor {
    let _a = name_scope("attn");
    let (nh, nkv, hd) = (
        cfg.num_attention_heads,
        cfg.num_key_value_heads,
        cfg.head_dim,
    );
    let q = build_q_proj::<D>(cfg, x);
    let k = build_kv_proj::<D>(cfg, x_hist, true);
    let v = build_kv_proj::<D>(cfg, x_hist, false);
    let bias = build_rel_bias::<D>(cfg, x, past, kv_len);
    let sinks = {
        let _g = name_scope("attn_sink_proj");
        linear::<D>(x, cfg.hidden_size, nh)
    };
    let spec = AttentionSpec {
        heads: nh,
        kv_heads: nkv,
        head_dim: hd,
        window,
        scale: (hd as f32).powf(-0.5),
    };
    let attn = q.record_custom(
        CustomData::new(SinkAttentionOp::<D>::new(spec, past_len)),
        &[&k, &v, k_cache, v_cache, &bias, &sinks],
        None,
    );
    build_o_proj::<D>(cfg, &attn, x)
}

// ---------------------------------------------------------------------------
// The looped core's two mechanisms
// ---------------------------------------------------------------------------

/// Records `attn_res`: each token retrieves a message from its block states using
/// learned queries.
///
/// `states` is `[T * S, hidden]`, the `S` block states of each token with the current
/// partial block last, as the reference's `cat([block_states, partial], dim=-2)`
/// builds it. The states are normalised here, once, and used as both keys and values.
///
/// `name` picks the query slot: `attn_queries` before attention (three queries) or
/// `mlp_queries` before the MLP (one). The checkpoint stores them per layer under
/// `attn_res`.
pub fn build_attn_res<D: FloatBytes + Send + Sync + 'static>(
    cfg: &Inkling2TextConfig,
    name: &str,
    states: &SymTensor,
    n_states: usize,
    queries: usize,
) -> SymTensor {
    let _a = name_scope("attn_res");
    let normed = states.record_custom(
        CustomData::new(UnweightedRmsNormOp::<D>::new(cfg.hidden_size)),
        &[],
        None,
    );
    let _g = name_scope(name);
    normed.record_custom(
        CustomData::new(BlockAttnOp::<D>::new(n_states, cfg.hidden_size, queries)),
        &[],
        None,
    )
}

/// Records one `controllers.*` gate (`Inkling2LoopMessageGate`):
///
/// ```text
/// g      = down(cat(norm(hidden), norm(message)))
/// out    = (1 - tanh(forget(g))) * hidden + tanh(gain(g)) * message
/// ```
///
/// `which` is `attn`, `k`, `v` or `mlp`, matching the checkpoint's four controllers.
/// The gates are computed from the normalised inputs but applied to the raw ones.
///
/// `tanh` is [`super::kernels::softcap::SoftcapOp`] with a cap of 1, since
/// `tanh(x / 1) * 1` is `tanh(x)`; no separate kernel is needed for it.
pub fn build_controller<D: FloatBytes + Send + Sync + 'static>(
    cfg: &Inkling2TextConfig,
    which: &str,
    hidden: &SymTensor,
    message: &SymTensor,
) -> SymTensor {
    let _c = name_scope("controllers");
    let _w = name_scope(which);
    let hid = cfg.hidden_size;
    let rank = cfg.loop_config.gate_rank;

    let norm = |x: &SymTensor| {
        x.record_custom(
            CustomData::new(UnweightedRmsNormOp::<D>::new(hid)),
            &[],
            None,
        )
    };
    let joined = norm(hidden).record_custom(
        CustomData::new(Cat2Op::<D>::new(hid, hid)),
        &[&norm(message)],
        None,
    );
    let g = {
        let _d = name_scope("down");
        linear::<D>(&joined, 2 * hid, rank)
    };
    let tanh =
        |x: &SymTensor| x.record_custom(CustomData::new(SoftcapOp::<D>::new(1.0)), &[], None);
    let forget = {
        let _f = name_scope("forget");
        tanh(&linear::<D>(&g, rank, hid))
    };
    let gain = {
        let _g2 = name_scope("gain");
        tanh(&linear::<D>(&g, rank, hid))
    };
    hidden.record_custom(
        CustomData::new(MessageMixOp::<D>::default()),
        &[message, &forget, &gain],
        None,
    )
}
