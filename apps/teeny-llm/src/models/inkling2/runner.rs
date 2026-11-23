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

//! Host-side plan for running Inkling-2: which layers execute in what order, which
//! KV cache each uses, and how checkpoint tensors map onto graph slots.
//!
//! **Status: the plan and the name map. Not yet a loop that generates tokens**,
//! which is blocked on weight loading; see "Why it cannot load yet" below.
//!
//! # The order
//!
//! Inkling-2 does not run its 24 layers once. Layers `core_start .. core_end` run
//! `num_passes` times, so Mini executes **48 layers per token** from 24 distinct sets
//! of weights:
//!
//! ```text
//! layers 0..6      once
//! layers 6..18     pass 0
//! layers 6..18     pass 1, with messages from pass 0
//! layers 6..18     pass 2, with messages from pass 1
//! layers 18..24    once
//! ```
//!
//! Each pass of the core needs **its own KV cache and convolution state**: the same
//! layer attends three times over different hidden states. The reference calls this
//! the cache offset. So the cache has `num_hidden_layers + (num_passes - 1) *
//! core_layers` entries, 48 for Mini, not 24.
//!
//! Passes after the first restart from the hidden and block states as they were
//! *before* the core, and take messages built from the previous pass's output. The
//! core's three passes are therefore siblings reading one another's results, not a
//! chain.
//!
//! # Why it cannot load yet
//!
//! The graph is f32 and the checkpoint's experts are **NVFP4**: 4-bit values with an
//! E4M3 scale per 16 of them. Expanding them to f32 at load time would turn 20.2 GB
//! into roughly 160 GB, and to bf16 about 40 GB, against the 31 GB a K3 board has.
//! The model fits only while the experts stay packed.
//!
//! So the dequantisation belongs inside
//! [`super::kernels::expert_gather`]'s inner loop, unpacking a block into registers
//! just before the multiply-add, not in a load-time pass. The RVV kernel for that is
//! measured and working (see `experiments/nvfp4-riscv` in this repository: 1.48x
//! over scalar for the unpack, and 1.9x to 3.1x for the fused form on an X100), but
//! it is not wired into the expert gather here. That is the remaining work before a
//! token can be generated, and it is a change to one kernel rather than to the
//! model.

use super::config::Inkling2TextConfig;

/// How a layer's experts are quantised.
///
/// Inkling-2-Mini is not uniform: the last four layers hold `F8_E4M3` weights with an
/// `U8` E8M0 scale per 32, and the rest hold `U8` E2M1 nibbles with an `F8_E4M3`
/// scale per 16 plus a per-expert `amax` and a shared denominator.
/// `hf_quant_config.json` lists eight MXFP8 tensors, which is the four layers' two
/// expert matrices.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Quant {
    /// Layers 0 to 19.
    Nvfp4,
    /// Layers 20 to 23.
    Mxfp8,
}

/// Which quantisation layer `l` uses.
///
/// Decided by the presence of `_amax` in the checkpoint, which
/// `test_the_checkpoint_layer_shapes_are_what_the_port_assumes` pins to the last four
/// layers. There is nothing in `config.json` that says so.
pub fn quant_of(cfg: &Inkling2TextConfig, l: usize) -> Quant {
    if l + 4 >= cfg.num_hidden_layers {
        Quant::Mxfp8
    } else {
        Quant::Nvfp4
    }
}

/// One layer execution: a layer index, which pass it belongs to, and the cache it
/// uses.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Step {
    /// Which of the model's layers this runs.
    pub layer: usize,
    /// Which pass through the looped core, or `None` outside it.
    pub pass: Option<usize>,
    /// Which KV and convolution cache slot it uses.
    pub cache: usize,
    /// Whether the layer's input joins the block states afterwards.
    pub is_boundary: bool,
    /// Whether it takes messages from the previous pass.
    pub takes_messages: bool,
}

/// The whole order of execution for one step, in order.
pub fn plan(cfg: &Inkling2TextConfig) -> Vec<Step> {
    let l = &cfg.loop_config;
    let core: Vec<usize> = (l.core_start..l.core_end).collect();
    let n_core = core.len();
    let mut steps = Vec::new();

    let mut push = |layer: usize, pass: Option<usize>, cache: usize| {
        steps.push(Step {
            layer,
            pass,
            cache,
            is_boundary: layer.is_multiple_of(cfg.n_layers_per_attn_res_block),
            takes_messages: pass.is_some_and(|p| p > 0),
        });
    };

    for layer in 0..l.core_start {
        push(layer, None, layer);
    }
    for &layer in &core {
        push(layer, Some(0), layer);
    }
    for pass in 1..l.num_passes {
        // Each later pass has its own cache, appended after the model's own layers.
        let base = cfg.num_hidden_layers + (pass - 1) * n_core;
        for (i, &layer) in core.iter().enumerate() {
            push(layer, Some(pass), base + i);
        }
    }
    for layer in l.core_end..cfg.num_hidden_layers {
        push(layer, None, layer);
    }
    steps
}

/// How many KV and convolution cache slots the plan needs.
pub fn cache_slots(cfg: &Inkling2TextConfig) -> usize {
    let l = &cfg.loop_config;
    cfg.num_hidden_layers + (l.num_passes - 1) * (l.core_end - l.core_start)
}

/// The checkpoint tensor a graph slot loads from, for the slots whose names differ.
///
/// Most match exactly. These do not, because `Linear4Op` always names its parameter
/// `weight` and the expert gather likewise, so the graph writes a dot where the
/// checkpoint has an underscore, or a generic name where it has a specific one.
///
/// `layer` is needed because `attn_res`'s queries are stored per layer at the top
/// level rather than inside the layer.
pub fn checkpoint_name(slot: &str, layer: usize) -> String {
    let prefix = format!("model.llm.layers.{layer}.");
    let top = "model.llm.";
    match slot {
        // controllers: down.weight -> down_weight, and the same for forget and gain
        s if s.starts_with("controllers.") && s.ends_with(".weight") => {
            let body = s.trim_end_matches(".weight");
            let (head, last) = body.rsplit_once('.').expect("controllers.<which>.<part>");
            format!("{prefix}{head}.{last}_weight")
        }
        "attn.rel_logits_proj.weight" => format!("{prefix}attn.rel_logits_proj.proj"),
        // The dynamic convolutions of the last layer of each block: the projection's
        // slot gains a `.weight` the checkpoint does not have, and `static_w` is
        // stored under the convolution rather than beside it.
        "attn.k_sconv.weight_proj.weight" => format!("{prefix}attn.k_sconv.weight_proj"),
        "attn.v_sconv.weight_proj.weight" => format!("{prefix}attn.v_sconv.weight_proj"),
        "mlp.experts.weight" => format!("{prefix}mlp.experts.w13_weight"),
        "mlp.experts.scale" => format!("{prefix}mlp.experts.w13_weight_scale"),
        "mlp.experts_down.weight" => format!("{prefix}mlp.experts.w2_weight"),
        "mlp.experts_down.scale" => format!("{prefix}mlp.experts.w2_weight_scale"),
        // NVFP4's per-expert scale is not stored as one tensor: it is `amax` divided
        // by a denominator shared by the layer, so the loader computes it and this
        // name is the numerator.
        "mlp.experts.scale2" => format!("{prefix}mlp.experts.w13_weight_amax"),
        "mlp.experts_down.scale2" => format!("{prefix}mlp.experts.w2_weight_amax"),
        // attn_res queries live at the top level, one tensor per layer, and every
        // query of a layer reads the same one.
        s if s.starts_with("attn_res.attn_queries_") => {
            format!("{top}attn_res.attn_queries.{layer}")
        }
        s if s.starts_with("attn_res.mlp_queries_") => {
            format!("{top}attn_res.mlp_queries.{layer}")
        }
        other => format!("{prefix}{other}"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mini() -> Inkling2TextConfig {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures/inkling2-mini-config.json");
        super::super::Inkling2Config::load(&path)
            .expect("the shipped config")
            .text_config
    }

    #[test]
    fn test_the_plan_runs_48_layers_from_24() {
        let cfg = mini();
        let steps = plan(&cfg);
        assert_eq!(
            steps.len(),
            48,
            "24 layers with a 12-layer core run three times"
        );
        assert_eq!(steps.len(), cfg.effective_depth());
        let distinct: std::collections::BTreeSet<usize> = steps.iter().map(|s| s.layer).collect();
        assert_eq!(distinct.len(), 24, "from 24 distinct sets of weights");
    }

    #[test]
    fn test_every_pass_has_its_own_cache() {
        let cfg = mini();
        let steps = plan(&cfg);
        let slots: std::collections::BTreeSet<usize> = steps.iter().map(|s| s.cache).collect();
        assert_eq!(
            slots.len(),
            steps.len(),
            "no two executions share a cache slot"
        );
        assert_eq!(slots.len(), cache_slots(&cfg));
        assert_eq!(cache_slots(&cfg), 48);
    }

    #[test]
    fn test_only_later_passes_take_messages() {
        let cfg = mini();
        let steps = plan(&cfg);
        let core = cfg.loop_config.core_end - cfg.loop_config.core_start;
        let with = steps.iter().filter(|s| s.takes_messages).count();
        // Two of the three passes receive messages; the first has none to receive.
        assert_eq!(with, core * (cfg.loop_config.num_passes - 1));
        assert!(
            steps
                .iter()
                .all(|s| !s.takes_messages || s.pass.is_some_and(|p| p > 0)),
            "only the core's later passes take messages"
        );
    }

    #[test]
    fn test_the_order_is_pre_core_passes_post() {
        let cfg = mini();
        let steps = plan(&cfg);
        let l = &cfg.loop_config;
        assert_eq!(steps[0].layer, 0);
        assert_eq!(
            steps[l.core_start].layer, l.core_start,
            "the core starts after the pre-loop layers"
        );
        assert_eq!(steps[l.core_start].pass, Some(0));
        let core = l.core_end - l.core_start;
        assert_eq!(
            steps[l.core_start + core].pass,
            Some(1),
            "pass 1 follows pass 0"
        );
        assert_eq!(
            steps[l.core_start + core].layer,
            l.core_start,
            "and restarts at the core's first layer"
        );
        assert_eq!(
            steps.last().expect("a last step").layer,
            cfg.num_hidden_layers - 1
        );
        assert_eq!(steps.last().expect("a last step").pass, None);
    }

    #[test]
    fn test_boundaries_fall_every_block() {
        let cfg = mini();
        let steps = plan(&cfg);
        assert!(steps[0].is_boundary, "layer 0 opens the first block");
        for s in &steps {
            assert_eq!(
                s.is_boundary,
                s.layer % cfg.n_layers_per_attn_res_block == 0,
                "layer {} boundary",
                s.layer
            );
        }
    }

    #[test]
    fn test_the_name_map_matches_the_checkpoint() {
        for (slot, layer, want) in [
            (
                "controllers.attn.down.weight",
                7,
                "model.llm.layers.7.controllers.attn.down_weight",
            ),
            (
                "controllers.k.forget.weight",
                7,
                "model.llm.layers.7.controllers.k.forget_weight",
            ),
            (
                "attn.rel_logits_proj.weight",
                3,
                "model.llm.layers.3.attn.rel_logits_proj.proj",
            ),
            (
                "mlp.experts.weight",
                3,
                "model.llm.layers.3.mlp.experts.w13_weight",
            ),
            (
                "mlp.experts_down.weight",
                3,
                "model.llm.layers.3.mlp.experts.w2_weight",
            ),
            (
                "attn_res.attn_queries_2.query",
                5,
                "model.llm.attn_res.attn_queries.5",
            ),
            (
                "attn_res.mlp_queries_0.query",
                5,
                "model.llm.attn_res.mlp_queries.5",
            ),
            // Slots that already match keep their name, with the layer prefix.
            (
                "attn.wq_du.weight",
                9,
                "model.llm.layers.9.attn.wq_du.weight",
            ),
            ("attn_norm.weight", 9, "model.llm.layers.9.attn_norm.weight"),
            ("mlp.gate.bias", 9, "model.llm.layers.9.mlp.gate.bias"),
        ] {
            assert_eq!(checkpoint_name(slot, layer), want, "mapping {slot}");
        }
    }
}
