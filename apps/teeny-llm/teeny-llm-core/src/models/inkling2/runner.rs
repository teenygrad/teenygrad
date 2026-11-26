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

/// How many learned queries a layer's attention residual has.
///
/// Not uniform, and not stated anywhere in `config.json`: the checkpoint holds
/// `[3, hidden]` for layers 5, 11, 17 and 23 and `[hidden]` for the rest, with
/// nothing at all for layer 0. Those four are exactly the layers whose convolutions
/// are dynamic and whose q, k and v are normed separately, so three queries give
/// them three distinct states; every other layer feeds one state to all three
/// projections, which is the reference's `num_queries = 1` form.
pub fn attn_queries_of(cfg: &Inkling2TextConfig, layer: usize) -> usize {
    if layer == 0 {
        0
    } else if (layer + 1).is_multiple_of(cfg.n_layers_per_attn_res_block) {
        3
    } else {
        1
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
        // The `_rows` slots are the integer NVFP4 projection's: the same tensors,
        // interleaved by row at load time.
        "mlp.experts.weight" | "mlp.experts.weight_rows" => {
            format!("{prefix}mlp.experts.w13_weight")
        }
        "mlp.experts.scale"
        | "mlp.experts.folded_scale"
        | "mlp.experts.folded_scale_rows"
        | "mlp.experts.scale_rows" => {
            format!("{prefix}mlp.experts.w13_weight_scale")
        }
        "mlp.experts_down.weight" | "mlp.experts_down.weight_rows" => {
            format!("{prefix}mlp.experts.w2_weight")
        }
        "mlp.experts_down.scale"
        | "mlp.experts_down.folded_scale"
        | "mlp.experts_down.folded_scale_rows"
        | "mlp.experts_down.scale_rows" => {
            format!("{prefix}mlp.experts.w2_weight_scale")
        }
        // NVFP4's per-expert scale is not stored as one tensor: it is `amax` divided
        // by a denominator shared by the layer, so the loader computes it and this
        // name is the numerator.
        "mlp.experts.scale2" => format!("{prefix}mlp.experts.w13_weight_amax"),
        "mlp.experts_down.scale2" => format!("{prefix}mlp.experts.w2_weight_amax"),
        // attn_res queries live at the top level, one tensor per layer, and every
        // query of a layer reads the same one.
        // The attention queries are offset by one: layer 0 has no attention
        // residual at all (the reference builds it only `if layer_idx > 0`), so the
        // checkpoint's 23 entries cover layers 1..=23 while the 24 mlp_queries cover
        // every layer. Indexing them by layer leaves the last layer with nothing.
        s if s.starts_with("attn_res.attn_queries_") => {
            format!("{top}attn_res.attn_queries.{}", layer.saturating_sub(1))
        }
        s if s.starts_with("attn_res.mlp_queries_") => {
            format!("{top}attn_res.mlp_queries.{layer}")
        }
        // The two ends of the model sit beside the layers, not inside one. They are
        // only ever built once, so the layer they are resolved against is arbitrary.
        "embed.weight" => format!("{top}embed.weight"),
        "embed_norm.weight" => format!("{top}embed_norm.weight"),
        "norm.weight" => format!("{top}norm.weight"),
        "unembed.weight" => format!("{top}unembed.weight"),
        // The cross-pass queries are shared by the whole core, not stored per layer:
        // two top-level tensors of [2 * n_loop, hidden], each read a row at a time.
        // The head's own block-attention query, stored centrally like the others.
        s if s.starts_with("attn_res.output_query_") => format!("{top}attn_res.output_query"),
        s if s.starts_with("cross_pass.kv_queries") => {
            format!("{top}middle_cycle_cross_pass_kv_queries")
        }
        s if s.starts_with("cross_pass.queries") => {
            format!("{top}middle_cycle_cross_pass_queries")
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
            // The attention queries are offset by one: layer 0 has none, so the
            // checkpoint's 23 entries cover layers 1..=23.
            (
                "attn_res.attn_queries_2.query",
                5,
                "model.llm.attn_res.attn_queries.4",
            ),
            (
                "attn_res.attn_queries_0.query",
                23,
                "model.llm.attn_res.attn_queries.22",
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

// ---------------------------------------------------------------------------
// Host-side state for one sequence
// ---------------------------------------------------------------------------

/// Everything a step carries between layers and between tokens.
///
/// A step graph per layer rather than one graph for the whole model: the looped core
/// restarts from saved states and the block states grow at boundaries, so the shapes
/// change as the stack is walked. Keeping the layer as the unit of compilation means
/// one graph per distinct layer shape instead of one enormous graph, and the host
/// carries the tensors.
pub struct State {
    /// Keys per cache slot, `capacity` rows of `kv_heads * head_dim`.
    pub k: Vec<Vec<f32>>,
    /// Values, as `k`.
    pub v: Vec<Vec<f32>>,
    /// How many rows of each cache are valid.
    pub len: usize,
    /// The four short-convolution states per cache slot, each `K - 1` rows: the key
    /// and value projections, then the layer's own two.
    pub conv: Vec<[Vec<f32>; 4]>,
    /// Block states accumulated at boundaries, `[T * n, hidden]` per token.
    pub blocks: Vec<Vec<f32>>,
    hidden: usize,
    kv_row: usize,
    conv_rows: usize,
}

impl State {
    /// Empty state for a sequence of at most `capacity` tokens.
    pub fn new(cfg: &Inkling2TextConfig, capacity: usize) -> Self {
        let slots = cache_slots(cfg);
        let kv_row = cfg.num_key_value_heads * cfg.head_dim;
        let conv_rows = cfg.sconv_kernel_size - 1;
        Self {
            k: vec![vec![0.0; capacity * kv_row]; slots],
            v: vec![vec![0.0; capacity * kv_row]; slots],
            len: 0,
            conv: (0..slots)
                .map(|_| {
                    [
                        vec![0.0; conv_rows * kv_row],
                        vec![0.0; conv_rows * kv_row],
                        vec![0.0; conv_rows * cfg.hidden_size],
                        vec![0.0; conv_rows * cfg.hidden_size],
                    ]
                })
                .collect(),
            blocks: Vec::new(),
            hidden: cfg.hidden_size,
            kv_row,
            conv_rows,
        }
    }

    /// Appends a step's keys and values to cache slot `slot`.
    pub fn append_kv(&mut self, slot: usize, k_new: &[f32], v_new: &[f32]) -> anyhow::Result<()> {
        let rows = k_new.len() / self.kv_row;
        anyhow::ensure!(
            k_new.len() == v_new.len() && k_new.len().is_multiple_of(self.kv_row),
            "append_kv: {} keys and {} values are not whole rows of {}",
            k_new.len(),
            v_new.len(),
            self.kv_row
        );
        let at = self.len * self.kv_row;
        anyhow::ensure!(
            at + k_new.len() <= self.k[slot].len(),
            "append_kv: slot {slot} holds {} rows, cannot take {} more at {}",
            self.k[slot].len() / self.kv_row,
            rows,
            self.len
        );
        self.k[slot][at..at + k_new.len()].copy_from_slice(k_new);
        self.v[slot][at..at + v_new.len()].copy_from_slice(v_new);
        Ok(())
    }

    /// Replaces a convolution state with the last `K - 1` rows of `x`.
    ///
    /// Shorter input keeps the older rows, which is what a one-token step needs.
    pub fn roll_conv(&mut self, slot: usize, which: usize, x: &[f32], width: usize) {
        let state = &mut self.conv[slot][which];
        let keep = self.conv_rows * width;
        if x.len() >= keep {
            state.copy_from_slice(&x[x.len() - keep..]);
        } else {
            state.rotate_left(x.len());
            let at = keep - x.len();
            state[at..].copy_from_slice(x);
        }
    }

    /// Appends one block state per token, which a boundary layer does with its input.
    pub fn push_block(&mut self, x: &[f32]) {
        self.blocks.push(x.to_vec());
    }

    /// The block states as the layer graph wants them: `[T * n, hidden]`, a token's
    /// states together.
    pub fn block_input(&self, tokens: usize) -> Vec<f32> {
        let n = self.blocks.len();
        let mut out = vec![0.0; tokens * n * self.hidden];
        for (s, block) in self.blocks.iter().enumerate() {
            for t in 0..tokens {
                let to = (t * n + s) * self.hidden;
                let from = t * self.hidden;
                out[to..to + self.hidden].copy_from_slice(&block[from..from + self.hidden]);
            }
        }
        out
    }

    /// How many block states each token carries.
    pub fn n_blocks(&self) -> usize {
        self.blocks.len()
    }

    /// Forgets the block states, which every pass through the core does.
    pub fn reset_blocks(&mut self, to: Vec<Vec<f32>>) {
        self.blocks = to;
    }
}

/// The id with the largest logit.
pub fn argmax(logits: &[f32]) -> u32 {
    let mut best = 0usize;
    for (i, v) in logits.iter().enumerate() {
        if v > &logits[best] {
            best = i;
        }
    }
    best as u32
}

#[cfg(test)]
mod state_tests {
    use super::*;

    fn cfg() -> Inkling2TextConfig {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures/inkling2-mini-config.json");
        super::super::Inkling2Config::load(&path)
            .expect("config")
            .text_config
    }

    #[test]
    fn test_state_has_a_cache_per_effective_layer() {
        let c = cfg();
        let s = State::new(&c, 16);
        assert_eq!(
            s.k.len(),
            48,
            "a cache per executed layer, not per weight set"
        );
        assert_eq!(s.k.len(), cache_slots(&c));
        assert_eq!(s.conv.len(), s.k.len());
        // Four convolutions per layer: the key and value projections, then the
        // layer's own attn_sconv and mlp_sconv.
        assert_eq!(s.conv[0].len(), 4);
        assert_eq!(
            s.conv[0][0].len(),
            (c.sconv_kernel_size - 1) * c.num_key_value_heads * c.head_dim
        );
        assert_eq!(
            s.conv[0][2].len(),
            (c.sconv_kernel_size - 1) * c.hidden_size
        );
    }

    #[test]
    fn test_appending_keys_respects_the_cursor() {
        let mut c = cfg();
        c.num_key_value_heads = 2;
        c.head_dim = 2;
        let row = c.num_key_value_heads * c.head_dim;
        let mut s = State::new(&c, 4);
        s.append_kv(0, &[1.0; 8], &[2.0; 8]).expect("two rows");
        s.len = 2;
        s.append_kv(0, &[3.0; 4], &[4.0; 4]).expect("one more");
        assert_eq!(&s.k[0][..row], &[1.0; 4], "the first row stays");
        assert_eq!(
            &s.k[0][2 * row..3 * row],
            &[3.0; 4],
            "the third lands at the cursor"
        );
        s.len = 4;
        assert!(
            s.append_kv(0, &[5.0; 4], &[5.0; 4]).is_err(),
            "past capacity must error"
        );
    }

    #[test]
    fn test_rolling_a_conv_state_keeps_the_newest_rows() {
        let mut c = cfg();
        c.hidden_size = 2;
        c.sconv_kernel_size = 4; // three rows of state
        let mut s = State::new(&c, 8);
        // A long step replaces the state with its last three rows.
        let long: Vec<f32> = (1..=10).map(|i| i as f32).collect(); // five rows of two
        s.roll_conv(0, 2, &long, 2);
        assert_eq!(s.conv[0][2], vec![5.0, 6.0, 7.0, 8.0, 9.0, 10.0]);
        // A one-row step shifts the window by one.
        s.roll_conv(0, 2, &[11.0, 12.0], 2);
        assert_eq!(s.conv[0][2], vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0]);
    }

    #[test]
    fn test_block_states_are_laid_out_per_token() {
        let mut c = cfg();
        c.hidden_size = 2;
        let mut s = State::new(&c, 8);
        // Two tokens, two block states: [t0s0, t0s1, t1s0, t1s1].
        s.push_block(&[1.0, 1.0, 2.0, 2.0]); // state 0 for tokens 0 and 1
        s.push_block(&[3.0, 3.0, 4.0, 4.0]); // state 1
        assert_eq!(s.n_blocks(), 2);
        assert_eq!(
            s.block_input(2),
            vec![1.0, 1.0, 3.0, 3.0, 2.0, 2.0, 4.0, 4.0],
            "a token's states must be adjacent, as block_attn indexes them"
        );
    }

    #[test]
    fn test_argmax_takes_the_first_of_equals() {
        assert_eq!(argmax(&[0.1, 0.9, 0.3]), 1);
        assert_eq!(argmax(&[1.0, 1.0]), 0, "ties go left, as the router's does");
        assert_eq!(argmax(&[-5.0, -1.0, -3.0]), 1);
    }
}
