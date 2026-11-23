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
    expert_gather::ExpertGatherOp, router::TopKRouterOp, scatter_add::ExpertScatterAddOp,
    swiglu::SwigluOp,
};
use crate::kernels::linear::Linear4Op;

/// Block width for the linear kernel, as in the Gemma port.
const LINEAR_BLOCK: i32 = 16;

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
    let op = Linear4Op::<D>::new(cfg.hidden_size, cfg.n_routed_experts, LINEAR_BLOCK);
    let logits = x.record_custom(CustomData::new(op), &[], None);
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
