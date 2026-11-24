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
use teeny_core::graph::{CustomData, Op, Shape, SymTensor};
use teeny_core::name_scope::name_scope;

use super::config::Inkling2TextConfig;
use super::kernels::{
    attention::{AttentionSpec, PastLen, SinkAttentionOp},
    block_attn::BlockAttnOp,
    cat2::Cat2Op,
    dyn_sconv::DynSconvOp,
    embed::EmbedOp,
    expert_gather::ExpertGatherOp,
    expert_gather_mxfp8::ExpertGatherMxfp8Op,
    expert_gather_nvfp4::ExpertGatherNvfp4Op,
    gated_mul::GatedMulOp,
    l2_clip::{L2ClipOp, ScaledL2ClipOp},
    mix::MessageMixOp,
    norm::{RmsNormOp, UnweightedRmsNormOp},
    rel_bias::RelBiasOp,
    router::TopKRouterOp,
    scale::ScaleOp,
    scatter_add::ExpertScatterAddOp,
    sconv::SconvOp,
    slot_ids::SlotIdsOp,
    softcap::SoftcapOp,
    swiglu::SwigluOp,
};
use super::runner::Quant;
use crate::kernels::last_row::LastRowOp;
use crate::kernels::linear::Linear4Op;

/// Block width for the linear kernel, as in the Gemma port.
const LINEAR_BLOCK: i32 = 16;

fn wrap(like: &SymTensor, node_id: usize, shape: Shape) -> SymTensor {
    SymTensor {
        node_id,
        graph: like.graph.clone(),
        dtype: like.dtype,
        shape,
    }
}

fn add(a: &SymTensor, b: &SymTensor) -> SymTensor {
    assert_eq!(a.shape, b.shape, "add: operand shapes differ");
    let shape = a.shape.clone();
    let node_id =
        a.graph
            .borrow_mut()
            .add_node(Op::Add, vec![a.node_id, b.node_id], a.dtype, shape.clone());
    wrap(a, node_id, shape)
}

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
/// `routed` is the router's packed `[T, 2 * top_k]` row, which [`build_router`]
/// produces in the same graph.
pub fn build_moe<D: FloatBytes + Send + Sync + 'static>(
    cfg: &Inkling2TextConfig,
    x: &SymTensor,
    routed: &SymTensor,
    quant: Option<Quant>,
) -> SymTensor {
    let _m = name_scope("mlp");
    // The down projection needs one id per (token, slot) row; the re-pack runs on the
    // device so the layer stays a single graph.
    let ids = &routed.record_custom(
        CustomData::new(SlotIdsOp::<D>::new(cfg.num_experts_per_tok)),
        &[],
        None,
    );
    let (hid, inter) = (cfg.hidden_size, cfg.intermediate_size);
    let (e, k) = (cfg.n_routed_experts, cfg.num_experts_per_tok);

    // `quant` picks the decoder; `None` keeps the plain f32 gather, which the tests
    // use and which no shipped layer needs.
    let gather = |n: usize, kk: usize, slots: usize, x: &SymTensor, r: &SymTensor| match quant {
        None => x.record_custom(
            CustomData::new(ExpertGatherOp::<D>::new(e, n, kk, slots)),
            &[r],
            None,
        ),
        Some(Quant::Nvfp4) => x.record_custom(
            CustomData::new(ExpertGatherNvfp4Op::<D>::new(e, n, kk, slots)),
            &[r],
            None,
        ),
        Some(Quant::Mxfp8) => x.record_custom(
            CustomData::new(ExpertGatherMxfp8Op::<D>::new(e, n, kk, slots)),
            &[r],
            None,
        ),
    };

    let gate_up = {
        let _g = name_scope("experts");
        gather(2 * inter, hid, k, x, routed)
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
        gather(hid, inter, 1, &act, ids)
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
/// `hist` carries the convolution's `K - 1` rows of state, which the caller keeps the
/// same way it keeps the KV cache. The convolution runs on the PROJECTED states, so
/// the history is of those, not of `x`.
pub fn build_kv_proj<D: FloatBytes + Send + Sync + 'static>(
    cfg: &Inkling2TextConfig,
    x: &SymTensor,
    hist: &SymTensor,
    is_key: bool,
    dynamic: Option<&SymTensor>,
) -> SymTensor {
    let (nkv, hd) = (cfg.num_key_value_heads, cfg.head_dim);
    let name = if is_key { "wk_dv" } else { "wv_dv" };
    let proj = {
        let _g = name_scope(name);
        linear::<D>(x, cfg.hidden_size, nkv * hd)
    };
    let conv = {
        let _g = name_scope(if is_key { "k_sconv" } else { "v_sconv" });
        match dynamic {
            // The last layer of each block projects its taps from the layer input.
            Some(src) => {
                let taps = {
                    let _w = name_scope("weight_proj");
                    linear::<D>(src, cfg.hidden_size, nkv * cfg.sconv_kernel_size)
                };
                hist.record_custom(
                    CustomData::new(DynSconvOp::<D>::new(nkv * hd, nkv, cfg.sconv_kernel_size)),
                    &[&proj, &taps],
                    None,
                )
            }
            None => hist.record_custom(
                CustomData::new(SconvOp::<D>::new(nkv * hd, cfg.sconv_kernel_size)),
                &[&proj],
                None,
            ),
        }
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
    past: PastLen,
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
        CustomData::new(RelBiasOp::<D>::new(nh, cfg.rel_extent, past)),
        &[],
        None,
    )
}

/// Records the attention half of a decoder layer: norm, the three projections, the
/// relative bias, attention with sinks, and the gated output projection.
///
/// The query, key and value inputs are separate: the looped core gates them
/// differently, and the checkpoint's three `attn_queries` produce three states. The
/// two `_hist` tensors are the key and value convolutions' state.
#[allow(clippy::too_many_arguments)]
pub fn build_attention<D: FloatBytes + Send + Sync + 'static>(
    cfg: &Inkling2TextConfig,
    q_in: &SymTensor,
    k_in: &SymTensor,
    v_in: &SymTensor,
    k_hist: &SymTensor,
    v_hist: &SymTensor,
    dynamic: Option<&SymTensor>,
    k_cache: &SymTensor,
    v_cache: &SymTensor,
    past_len: PastLen,
    window: Option<usize>,
) -> SymTensor {
    let _a = name_scope("attn");
    let (nh, nkv, hd) = (
        cfg.num_attention_heads,
        cfg.num_key_value_heads,
        cfg.head_dim,
    );
    let q = build_q_proj::<D>(cfg, q_in);
    let k = build_kv_proj::<D>(cfg, k_in, k_hist, true, dynamic);
    let v = build_kv_proj::<D>(cfg, v_in, v_hist, false, dynamic);
    // The bias and the sinks are conditioned on the query's hidden states.
    let bias = build_rel_bias::<D>(cfg, q_in, past_len.clone());
    let sinks = {
        let _g = name_scope("attn_sink_proj");
        linear::<D>(q_in, cfg.hidden_size, nh)
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
    build_o_proj::<D>(cfg, &attn, q_in)
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
    partial: &SymTensor,
    n_states: usize,
    queries: usize,
) -> Vec<SymTensor> {
    let _a = name_scope("attn_res");
    // The reference norms the stack once; normalising the committed states and the
    // partial separately is the same, since the norm is per row.
    let norm = |x: &SymTensor| {
        x.record_custom(
            CustomData::new(UnweightedRmsNormOp::<D>::new(cfg.hidden_size)),
            &[],
            None,
        )
    };
    let normed = norm(states);
    let normed_partial = norm(partial);
    // One op per query. They share the checkpoint's `[queries, hidden]` tensor, so
    // it loads into each of their slots; at 3 x 2304 floats that is ~27 KB, which is
    // cheaper than the alternatives (slicing the output needs a strided view the
    // graph has no op for).
    (0..queries)
        .map(|q| {
            let _g = name_scope(format!("{name}_{q}"));
            normed.record_custom(
                CustomData::new(BlockAttnOp::<D>::new(n_states, cfg.hidden_size, queries, q)),
                &[],
                None,
            )
        })
        .collect()
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

// ---------------------------------------------------------------------------
// The decoder layer
// ---------------------------------------------------------------------------

/// What a layer needs from the host that is not a tensor: the cache geometry and,
/// on the looped core's later passes, the messages from the previous pass.
pub struct LayerInputs<'a> {
    /// The layer's input, `[T, hidden]`.
    pub x: &'a SymTensor,
    /// The block states of each token, `[T * n_states, hidden]`, the current partial
    /// block last.
    pub block_states: &'a SymTensor,
    /// How many states that is per token.
    pub n_states: usize,
    /// The key convolution's `K - 1` rows of state, of the PROJECTED keys.
    pub k_hist: &'a SymTensor,
    /// As `k_hist`, for values.
    pub v_hist: &'a SymTensor,
    /// The attention output with the same history, for `attn_sconv`.
    pub attn_hist: &'a SymTensor,
    /// The MoE output with the same history, for `mlp_sconv`.
    pub mlp_hist: &'a SymTensor,
    /// Keys and values already cached.
    pub k_cache: &'a SymTensor,
    /// As `k_cache`.
    pub v_cache: &'a SymTensor,
    /// On passes after the first through the looped core, the previous pass's
    /// messages for the query, key, value and MLP inputs, in that order.
    pub messages: Option<[&'a SymTensor; 4]>,
}

/// Records one decoder layer.
///
/// ```text
/// block_residual = x
/// qh, kh, vh     = attn_res(block_states, x, attn_queries)   three learned queries
/// (looping)      = controllers.q/k/v(qh|kh|vh, message)
/// h              = attn(attn_norm(qh), attn_norm(kh), attn_norm(vh))
/// h              = attn_sconv(h)
/// residual       = h                       at a block boundary
///                = block_residual + h      otherwise
/// m              = attn_res(block_states, h, mlp_queries)    one learned query
/// (looping)      = controllers.mlp(m, message)
/// m              = mlp_sconv(moe(mlp_norm(m)))
/// out            = residual + m
/// ```
///
/// **The block boundary.** Every `n_layers_per_attn_res_block` layers the layer's
/// own input joins the block states instead of being added back as a residual. The
/// host appends it to `block_states` for the next layer; here the difference is only
/// whether `residual` is the attention output alone or the sum.
///
/// **Three queries, not one.** The checkpoint's `attn_res.attn_queries.{layer}` is
/// `[3, hidden]`, so the block attention yields three different states, one each for
/// the query, key and value projections. The reference's `num_queries = 1` form
/// feeds the same state to all three; the shipped weights say otherwise, and the
/// looped layer's separate q/k/v message gates agree with three.
#[allow(clippy::too_many_arguments)]
pub fn build_layer<D: FloatBytes + Send + Sync + 'static>(
    cfg: &Inkling2TextConfig,
    layer: usize,
    inputs: &LayerInputs<'_>,
    past_len: PastLen,
) -> SymTensor {
    let hid = cfg.hidden_size;
    let is_boundary = layer.is_multiple_of(cfg.n_layers_per_attn_res_block);
    // The LAST layer of each block convolves dynamically and norms q, k and v
    // separately; every other layer has a single attn_norm and a static convolution.
    let is_dynamic = (layer + 1).is_multiple_of(cfg.n_layers_per_attn_res_block);
    let window = if cfg.is_sliding(layer) {
        Some(cfg.sliding_window_size)
    } else {
        None
    };

    // Three states out of the block attention, one per projection.
    let qkv = build_attn_res::<D>(
        cfg,
        "attn_queries",
        inputs.block_states,
        inputs.x,
        inputs.n_states,
        3,
    );
    let gated: Vec<SymTensor> = match inputs.messages {
        None => qkv,
        Some(msgs) => ["q", "k", "v"]
            .iter()
            .enumerate()
            .map(|(i, which)| build_controller::<D>(cfg, which, &qkv[i], msgs[i]))
            .collect(),
    };
    let normed: Vec<SymTensor> = if is_dynamic {
        ["q_input_norm", "k_input_norm", "v_input_norm"]
            .iter()
            .enumerate()
            .map(|(i, n)| build_norm::<D>(n, &gated[i], hid))
            .collect()
    } else {
        gated
            .iter()
            .map(|g| build_norm::<D>("attn_norm", g, hid))
            .collect()
    };

    let attn = build_attention::<D>(
        cfg,
        &normed[0],
        &normed[1],
        &normed[2],
        inputs.k_hist,
        inputs.v_hist,
        // The dynamic taps are projected from the layer input, before any norm.
        if is_dynamic { Some(inputs.x) } else { None },
        inputs.k_cache,
        inputs.v_cache,
        past_len,
        window,
    );
    let h = {
        let _g = name_scope("attn_sconv");
        inputs.attn_hist.record_custom(
            CustomData::new(SconvOp::<D>::new(hid, cfg.sconv_kernel_size)),
            &[&attn],
            None,
        )
    };
    let residual = if is_boundary {
        h.clone()
    } else {
        add(inputs.x, &h)
    };

    let m = build_attn_res::<D>(
        cfg,
        "mlp_queries",
        inputs.block_states,
        &residual,
        inputs.n_states,
        1,
    )
    .remove(0);
    let m = match inputs.messages {
        None => m,
        Some(msgs) => build_controller::<D>(cfg, "mlp", &m, msgs[3]),
    };
    let m = build_norm::<D>("mlp_norm", &m, hid);
    let routed = build_router::<D>(cfg, &m);
    let moe = build_moe::<D>(cfg, &m, &routed, Some(super::runner::quant_of(cfg, layer)));
    let m = {
        let _g = name_scope("mlp_sconv");
        inputs.mlp_hist.record_custom(
            CustomData::new(SconvOp::<D>::new(hid, cfg.sconv_kernel_size)),
            &[&moe],
            None,
        )
    };
    add(&residual, &m)
}

// ---------------------------------------------------------------------------
// The two ends of the model
// ---------------------------------------------------------------------------

/// Records the embedding: a lookup, then `embed_norm`.
///
/// `Inkling2LoopNormedEmbedding` does not scale the lookup the way Gemma does; the
/// norm is a separate weight (`model.llm.embed_norm`).
pub fn build_embed<D: FloatBytes + Send + Sync + 'static>(
    cfg: &Inkling2TextConfig,
    ids: &SymTensor,
) -> SymTensor {
    let e = {
        let _g = name_scope("embed");
        ids.record_custom(
            CustomData::new(EmbedOp::<D>::new(cfg.vocab_size, cfg.hidden_size)),
            &[],
            None,
        )
    };
    build_norm::<D>("embed_norm", &e, cfg.hidden_size)
}

/// Records the head: final norm, the last token's row, the unembedding, the muP
/// width scaling, and the logit softcap.
///
/// Three details, all from the config rather than from the reference's defaults:
/// `logits_mup_width_multiplier` divides the logits, `final_logit_softcapping` caps
/// them, and only the LAST row is projected, since a step needs one distribution and
/// the vocabulary is 201024 wide.
pub fn build_head<D: FloatBytes + Send + Sync + 'static>(
    cfg: &Inkling2TextConfig,
    x: &SymTensor,
    mup_width_multiplier: f64,
) -> SymTensor {
    let x = build_norm::<D>("norm", x, cfg.hidden_size);
    let last = x.record_custom(CustomData::new(LastRowOp::<D>::default()), &[], None);
    let logits = {
        let _g = name_scope("unembed");
        linear::<D>(&last, cfg.hidden_size, cfg.vocab_size)
    };
    // muP: the trained width multiplier divides the logits before the cap.
    let scaled = logits.record_custom(
        CustomData::new(ScaleOp::<D>::new(1.0 / mup_width_multiplier as f32)),
        &[],
        None,
    );
    match cfg.final_logit_softcapping {
        Some(cap) => {
            scaled.record_custom(CustomData::new(SoftcapOp::<D>::new(cap as f32)), &[], None)
        }
        None => scaled,
    }
}
