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
    controller::{ControllerApplyOp, ControllerDownOp},
    dyn_sconv::DynSconvOp,
    embed::EmbedOp,
    expert_gather::ExpertGatherOp,
    expert_gemv_mxfp8::ExpertGemvMxfp8Op,
    expert_gemv_nvfp4::{ExpertGemvNvfp4Op, QuantRowsQ8Op},
    gated_mul::GatedMulOp,
    l2_clip::{L2ClipOp, ScaledL2ClipOp},
    mix::MessageMixOp,
    norm::{RmsNormOp, UnweightedRmsNormOp},
    regroup::RegroupOp,
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
use crate::kernels::linear::{LANES, Linear4Bf16Op, LinearLanesBf16Op, LinearLanesI8Op};

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

/// Records a bias-free linear in the current scope, on int8 weights (`weight_i8`).
///
/// Every linear weight in the checkpoint is bf16, so it is kept as bf16, two to a
/// word, and widened exactly in the kernel: half the bytes of f32 for the same
/// arithmetic, on projections that run at memory bandwidth. Output features are
/// the lanes, so a batch of rows unpacks each weight once. Decode reads every
/// weight once a token at memory bandwidth, so the weights are quantised to int8
/// per output row at load: half the bytes again. A projection too narrow
/// for whole blocks of lanes (24 outputs) keeps the lanes-along-`K` kernel and its
/// `weight_bf16` slot.
fn linear<D: FloatBytes + Send + Sync + 'static>(x: &SymTensor, k: usize, n: usize) -> SymTensor {
    // INKLING2_BF16_LINEARS keeps the exact bf16 weights, to compare against.
    let int8 = std::env::var_os("INKLING2_BF16_LINEARS").is_none();
    if int8 && n.is_multiple_of(LANES) && k.is_multiple_of(4) {
        x.record_custom(CustomData::new(LinearLanesI8Op::<D>::new(k, n)), &[], None)
    } else if n.is_multiple_of(LANES) {
        x.record_custom(CustomData::new(LinearLanesBf16Op::<D>::new(k, n)), &[], None)
    } else {
        x.record_custom(CustomData::new(Linear4Bf16Op::<D>::new(k, n, 16)), &[], None)
    }
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
        // Integer arithmetic: the activations are quantised to int8 per 16-element
        // block first, and the weights are exact int8 inside the projection.
        Some(Quant::Nvfp4) => x
            .record_custom(CustomData::new(QuantRowsQ8Op::<D>::new(kk)), &[], None)
            .record_custom(
                CustomData::new(ExpertGemvNvfp4Op::<D>::new(e, n, kk, slots)),
                &[r],
                None,
            ),
        // Rows as lanes, like the NVFP4 projection, in f32: an E4M3 value is not an
        // integer at any useful scale.
        Some(Quant::Mxfp8) => x.record_custom(
            CustomData::new(ExpertGemvMxfp8Op::<D>::new(e, n, kk, slots)),
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
) -> (SymTensor, SymTensor) {
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
    // The pre-convolution projection comes back too: it is what the convolution's
    // state carries into the next token, and the host cannot recover it from the output.
    let out = if is_key {
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
    };
    (proj, out)
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
    width: Option<usize>,
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
    // The projection is per head, so the matmul's rows are (token, head) pairs:
    // [T, heads * d_rel] regrouped to [T * heads, d_rel].
    let rel = rel.record_custom(CustomData::new(RegroupOp::<D>::new(nh)), &[], None);
    let logits = {
        let _g = name_scope("rel_logits_proj");
        linear::<D>(&rel, dr, cfg.rel_extent)
    };
    logits.record_custom(
        CustomData::new(RelBiasOp::<D>::with_width(nh, cfg.rel_extent, past, width)),
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
    dynamic: bool,
    k_cache: &SymTensor,
    v_cache: &SymTensor,
    past_len: PastLen,
    window: Option<usize>,
    bias_width: Option<usize>,
) -> Attention {
    let _a = name_scope("attn");
    let (nh, nkv, hd) = (
        cfg.num_attention_heads,
        cfg.num_key_value_heads,
        cfg.head_dim,
    );
    let q = build_q_proj::<D>(cfg, q_in);
    // A dynamic convolution's per-token taps are projected from the projection's own
    // input -- the normed key states for the key, the normed value states for the
    // value -- not from the layer input, and not from one shared tensor. The reference
    // passes `hidden_states` straight back into its own sconv.
    let (k_proj, k) = build_kv_proj::<D>(cfg, k_in, k_hist, true, dynamic.then_some(k_in));
    let (v_proj, v) = build_kv_proj::<D>(cfg, v_in, v_hist, false, dynamic.then_some(v_in));
    // The bias and the sinks are conditioned on the query's hidden states.
    let bias = build_rel_bias::<D>(cfg, q_in, past_len.clone(), bias_width);
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
    Attention {
        out: build_o_proj::<D>(cfg, &attn, q_in),
        k,
        v,
        k_proj,
        v_proj,
    }
}

/// What [`build_attention`] produces: its output, plus the tensors the host has to
/// carry from one token to the next.
///
/// `k` and `v` are post-convolution and post-norm, which is the form the cache holds,
/// so appending them is all the next token needs. `k_proj` and `v_proj` are the
/// *pre*-convolution projections, which are what each short convolution's state
/// window carries; neither can be recovered from the layer's output.
pub struct Attention {
    /// The attention block's output, `[T, hidden]`.
    pub out: SymTensor,
    /// Keys to append to the cache, `[T, kv_heads * head_dim]`.
    pub k: SymTensor,
    /// Values to append, as `k`.
    pub v: SymTensor,
    /// The key projection before its convolution, for the convolution's state.
    pub k_proj: SymTensor,
    /// As `k_proj`, for values.
    pub v_proj: SymTensor,
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
                &[&normed_partial],
                None,
            )
        })
        .collect()
}

/// Records the cross-pass message retrieval (`Inkling2LoopMessageAttentionResidual`):
/// the four messages one loop layer receives from the previous pass, in the reference's
/// order (query, key, value, mlp).
///
/// Like [`build_attn_res`] this is attention over the normalised stack of
/// `states` plus `partial`, but with queries that are shared by the whole core
/// rather than stored per layer. The reference holds them as one `[n_loop, 4, hidden]`
/// parameter; the checkpoint splits the same 48 vectors across two top-level tensors of
/// `[2 * n_loop, hidden]`:
///
/// ```text
/// middle_cycle_cross_pass_queries     row 2l     -> query    row 2l + 1 -> mlp
/// middle_cycle_cross_pass_kv_queries  row 2l     -> key      row 2l + 1 -> value
/// ```
///
/// The split by name is certain (one tensor is named for the keys and values); the
/// order **within** each pair is the one inference left, and nothing in the checkpoint
/// or the reference pins it down. Swapping query with mlp, or key with value, is the
/// first thing to try if a pass-2 comparison disagrees.
///
/// Each [`BlockAttnOp`] reads one row of the whole `[2 * n_loop, hidden]` tensor through
/// its `query_index`, so both tensors load unmodified and no slicing op is needed.
pub fn build_messages<D: FloatBytes + Send + Sync + 'static>(
    cfg: &Inkling2TextConfig,
    states: &SymTensor,
    partial: &SymTensor,
    n_states: usize,
    loop_layer: usize,
) -> [SymTensor; 4] {
    let _a = name_scope("cross_pass");
    let hid = cfg.hidden_size;
    let rows = 2 * (cfg.loop_config.core_end - cfg.loop_config.core_start);
    let norm = |x: &SymTensor| {
        x.record_custom(
            CustomData::new(UnweightedRmsNormOp::<D>::new(hid)),
            &[],
            None,
        )
    };
    let normed = norm(states);
    let normed_partial = norm(partial);

    // (checkpoint tensor, row) for query, key, value and mlp.
    let slots = [
        ("queries", 2 * loop_layer),
        ("kv_queries", 2 * loop_layer),
        ("kv_queries", 2 * loop_layer + 1),
        ("queries", 2 * loop_layer + 1),
    ];
    let built: Vec<SymTensor> = slots
        .iter()
        .enumerate()
        .map(|(i, (tensor, row))| {
            // The scope names the parameter slot, so each op gets its own; the name
            // carries which checkpoint tensor the loader must put there.
            let _g = name_scope(format!("{tensor}_{loop_layer}_{i}"));
            normed.record_custom(
                CustomData::new(BlockAttnOp::<D>::new(n_states, hid, rows, *row)),
                &[&normed_partial],
                None,
            )
        })
        .collect();
    let mut it = built.into_iter();
    [
        it.next().expect("query message"),
        it.next().expect("key message"),
        it.next().expect("value message"),
        it.next().expect("mlp message"),
    ]
}

/// Records every loop layer's messages in one graph.
///
/// All twelve share the same two inputs (the previous pass's trailing block states and
/// its final hidden states), so one graph serves the whole pass, and the same graph
/// serves every later pass: only the inputs differ.
///
/// The result is indexed by position in the core, not by model layer: entry `i` is for
/// layer `core_start + i`.
pub fn build_message_block<D: FloatBytes + Send + Sync + 'static>(
    cfg: &Inkling2TextConfig,
    states: &SymTensor,
    partial: &SymTensor,
    n_states: usize,
) -> Vec<[SymTensor; 4]> {
    let l = &cfg.loop_config;
    (0..l.core_end - l.core_start)
        .map(|i| build_messages::<D>(cfg, states, partial, n_states, i))
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
/// Computed by [`super::kernels::controller`] in two launches.
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

    // Two fused kernels rather than nine: see `kernels::controller`. Their weight
    // slots are `down.weight_bf16`, `forget.weight_bf16` and `gain.weight_bf16` under
    // this scope, the same checkpoint tensors the separate linears read.
    let g = hidden.record_custom(
        CustomData::new(ControllerDownOp::<D>::new(hid, rank)),
        &[message],
        None,
    );
    hidden.record_custom(
        CustomData::new(ControllerApplyOp::<D>::new(hid, rank)),
        &[message, &g],
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
    /// The block states the MLP's residual attends over, and how many per token.
    ///
    /// A boundary layer appends its own input to the stack *between* the attention
    /// and the MLP, so the two see different stacks: the attention the old one, the
    /// MLP one state deeper. Away from a boundary these are `block_states` and
    /// `n_states` again.
    pub mlp_block_states: &'a SymTensor,
    /// How many states `mlp_block_states` holds per token.
    pub mlp_n_states: usize,
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
    ///
    /// A looped-core layer should pass these on **every** pass, with `msg_scale` as the
    /// switch, so that one compiled graph serves all three. Giving `None` builds a layer
    /// with no controllers at all, which is right for a layer outside the core.
    pub messages: Option<[&'a SymTensor; 4]>,
    /// Pins the relative-bias width, for a graph reused across a whole sequence.
    ///
    /// The runtime resolves output shapes once, at compile time, so a bias sized from
    /// the cache length is only ever right at the length it was compiled at. The
    /// engine pins this to the cache capacity; a one-shot graph leaves it `None`.
    pub bias_width: Option<usize>,
    /// Blends the controller in: `0` leaves the layer input untouched (the first pass,
    /// which takes no messages) and `1` applies the gate in full.
    ///
    /// Present only with `messages`. It is a tensor rather than a constant because the
    /// alternative is a second compiled graph per core layer, and a second copy of that
    /// layer's 682 MB of expert weights with it -- 8.2 GB that does not fit beside
    /// everything else on a 31 GB board.
    pub msg_scale: Option<&'a SymTensor>,
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
) -> Layer {
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
    // The attention residual's query count varies: none on layer 0, three on the
    // dynamic layers (which norm q, k and v separately), one everywhere else. A
    // single query feeds the same state to all three projections.
    let n_queries = super::runner::attn_queries_of(cfg, layer);
    let qkv: Vec<SymTensor> = if n_queries == 0 {
        // Layer 0 has no block states to attend over and no queries in the
        // checkpoint, so it uses its own input.
        vec![inputs.x.clone(), inputs.x.clone(), inputs.x.clone()]
    } else {
        let got = build_attn_res::<D>(
            cfg,
            "attn_queries",
            inputs.block_states,
            inputs.x,
            inputs.n_states,
            n_queries,
        );
        if n_queries == 1 {
            vec![got[0].clone(), got[0].clone(), got[0].clone()]
        } else {
            got
        }
    };

    // `lerp(a, b, s)` as `(1 - s) * a + s * b`, which is exactly what the controller's
    // own combine kernel computes when both of its gates are the same tensor.
    // The combine takes four distinct inputs. Recording `s` twice would give the node
    // the same edge twice and it would arrive with three, so the second gate is a
    // separate node: an identity scale, one elementwise pass over `hidden` floats.
    let lerp = |a: &SymTensor, b: &SymTensor, s: &SymTensor| {
        let s2 = s.record_custom(CustomData::new(ScaleOp::<D>::new(1.0)), &[], None);
        a.record_custom(
            CustomData::new(MessageMixOp::<D>::default()),
            &[b, s, &s2],
            None,
        )
    };
    let gated: Vec<SymTensor> = match inputs.messages {
        None => qkv,
        // The checkpoint names the query controller `attn`, not `q`.
        Some(msgs) => ["attn", "k", "v"]
            .iter()
            .enumerate()
            .map(|(i, which)| {
                let g = build_controller::<D>(cfg, which, &qkv[i], msgs[i]);
                match inputs.msg_scale {
                    Some(s) => lerp(&qkv[i], &g, s),
                    None => g,
                }
            })
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
        is_dynamic,
        inputs.k_cache,
        inputs.v_cache,
        past_len,
        window,
        inputs.bias_width,
    );
    let h = {
        let _g = name_scope("attn_sconv");
        inputs.attn_hist.record_custom(
            CustomData::new(SconvOp::<D>::new(hid, cfg.sconv_kernel_size)),
            &[&attn.out],
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
        inputs.mlp_block_states,
        &residual,
        inputs.mlp_n_states,
        1,
    )
    .remove(0);
    let m = match inputs.messages {
        None => m,
        Some(msgs) => {
            let g = build_controller::<D>(cfg, "mlp", &m, msgs[3]);
            match inputs.msg_scale {
                Some(s) => lerp(&m, &g, s),
                None => g,
            }
        }
    };
    let m = build_norm::<D>("mlp_norm", &m, hid);
    let routed = build_router::<D>(cfg, &m);
    let moe = build_moe::<D>(cfg, &m, &routed, Some(super::runner::quant_of(cfg, layer)));
    let m_conv = {
        let _g = name_scope("mlp_sconv");
        inputs.mlp_hist.record_custom(
            CustomData::new(SconvOp::<D>::new(hid, cfg.sconv_kernel_size)),
            &[&moe],
            None,
        )
    };
    Layer {
        out: add(&residual, &m_conv),
        k: attn.k,
        v: attn.v,
        // The four convolution states, in the order State::conv holds them: the key
        // and value projections, then the attention and MoE outputs.
        conv_in: [attn.k_proj, attn.v_proj, attn.out, moe],
    }
}

/// What [`build_layer`] produces: the layer output, and the state the host carries to
/// the next token.
pub struct Layer {
    /// The layer's output, `[T, hidden]`.
    pub out: SymTensor,
    /// Keys to append to this layer's cache slot.
    pub k: SymTensor,
    /// Values to append.
    pub v: SymTensor,
    /// The inputs of the layer's four short convolutions, whose last `K - 1` rows
    /// become the next token's history, in `State::conv` order.
    pub conv_in: [SymTensor; 4],
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
    block_states: &SymTensor,
    n_states: usize,
    mup_width_multiplier: f64,
) -> SymTensor {
    // One last block-attention residual before the norm, with its own query:
    // `TextModel.forward` ends `hidden = self.final_input(block_states, hidden)[0]`.
    // Skipping it feeds the head a hidden state the model never produces.
    let x = build_attn_res::<D>(cfg, "output_query", block_states, x, n_states, 1).remove(0);
    let x = build_norm::<D>("norm", &x, cfg.hidden_size);
    // Every row the caller passes gets logits: decode passes one, verification all.
    let logits = {
        let _g = name_scope("unembed");
        linear::<D>(&x, cfg.hidden_size, cfg.vocab_size)
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
