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

//! The subset of `Inkling2LoopForConditionalGeneration`'s `config.json` the port uses.
//!
//! Inkling-2-Mini differs from Gemma in four ways that reach into the step graph, so
//! they are parsed and validated here rather than assumed:
//!
//! - **Looped depth.** Layers `core_start..core_end` run `num_passes` times with the
//!   same weights, gated by a rank-`gate_rank` projection. Depth is therefore not
//!   `num_hidden_layers`.
//! - **Mixture of experts.** `num_experts_per_tok` of `n_routed_experts` fire per
//!   token, per layer. The expert weights ship as NVFP4 (and MXFP8 for a few layers).
//! - **No RoPE.** Position enters through a relative scheme of width `d_rel` over
//!   `rel_extent`, plus short convolutions of width `sconv_kernel_size`.
//! - **Softcapping**, which the Gemma port rejects outright, is used here on the MLP
//!   gate, the MLP up projection, and the final logits.

use std::path::Path;

use serde::Deserialize;

/// The looped-core settings: `core_start..core_end` repeated `num_passes` times.
#[derive(Clone, Debug, Deserialize)]
pub struct LoopConfig {
    /// First layer of the repeated core.
    pub core_start: usize,
    /// One past the last layer of the repeated core.
    pub core_end: usize,
    /// How many times the core runs. 1 would mean an ordinary stack.
    pub num_passes: usize,
    /// Rank of the gate that mixes each pass's output.
    pub gate_rank: usize,
}

/// Inkling-2 text hyperparameters, from `config.json`'s `text_config`.
#[derive(Clone, Debug, Deserialize)]
pub struct Inkling2TextConfig {
    /// Model width.
    pub hidden_size: usize,
    /// Width of one expert's MLP.
    pub intermediate_size: usize,
    /// Distinct decoder layers, before looping.
    pub num_hidden_layers: usize,
    /// Query heads.
    pub num_attention_heads: usize,
    /// Key/value heads.
    pub num_key_value_heads: usize,
    /// Lanes per head.
    pub head_dim: usize,
    /// Vocabulary size, padded.
    pub vocab_size: usize,
    /// Vocabulary size actually used by the tokenizer.
    pub unpadded_vocab_size: usize,
    /// RMSNorm epsilon.
    pub rms_norm_eps: f64,
    /// Width of the relative positional embedding.
    pub d_rel: usize,
    /// How far the relative scheme reaches before it is clamped.
    pub rel_extent: usize,
    /// Width of the short convolutions on the attention branch.
    pub sconv_kernel_size: usize,
    /// Sliding window, for the layers listed in `local_layer_ids`.
    pub sliding_window_size: usize,
    /// Which layers use the sliding window; the rest attend globally.
    pub local_layer_ids: Vec<usize>,
    /// Experts per layer.
    pub n_routed_experts: usize,
    /// Experts that fire per token.
    pub num_experts_per_tok: usize,
    /// Always-on experts. Zero for Inkling-2-Mini, unlike Inkling-1.
    pub n_shared_experts: usize,
    /// Router logits are multiplied by this.
    pub route_scale: f64,
    /// Whether the router has a bias term.
    pub use_gate_bias: bool,
    /// Softcap on the MLP gate.
    pub mlp_gate_softcap: f64,
    /// Softcap on the MLP up projection.
    pub mlp_up_softcap: f64,
    /// Softcap on the output logits.
    pub final_logit_softcapping: Option<f64>,
    /// L2 clip threshold on queries.
    pub qk_l2_clip_threshold: f64,
    /// L2 clip threshold on the relative term.
    pub rel_l2_clip_threshold: f64,
    /// Layers per attention-residual block.
    pub n_layers_per_attn_res_block: usize,
    /// The looped core.
    pub loop_config: LoopConfig,
}

/// The top level of `config.json`.
#[derive(Clone, Debug, Deserialize)]
pub struct Inkling2Config {
    /// Text-model settings.
    pub text_config: Inkling2TextConfig,
    /// Token id that ends generation.
    pub eos_token_id: u32,
}

impl Inkling2Config {
    /// Reads `config.json`, rejecting anything the port does not implement.
    pub fn load(path: &Path) -> anyhow::Result<Self> {
        let text = std::fs::read_to_string(path)
            .map_err(|e| anyhow::anyhow!("reading Inkling-2 config {}: {e}", path.display()))?;
        let cfg: Inkling2Config = serde_json::from_str(&text)
            .map_err(|e| anyhow::anyhow!("parsing Inkling-2 config {}: {e}", path.display()))?;
        cfg.text_config.validate()?;
        Ok(cfg)
    }
}

impl Inkling2TextConfig {
    /// Rejects configurations outside what the step graph handles.
    pub fn validate(&self) -> anyhow::Result<()> {
        let l = &self.loop_config;
        anyhow::ensure!(
            l.core_start < l.core_end && l.core_end <= self.num_hidden_layers,
            "loop core {}..{} is not inside {} layers",
            l.core_start,
            l.core_end,
            self.num_hidden_layers
        );
        anyhow::ensure!(
            l.num_passes >= 1,
            "num_passes {} is not positive",
            l.num_passes
        );
        anyhow::ensure!(
            self.num_experts_per_tok <= self.n_routed_experts,
            "{} experts fire per token but only {} exist",
            self.num_experts_per_tok,
            self.n_routed_experts
        );
        anyhow::ensure!(
            self.local_layer_ids
                .iter()
                .all(|&l| l < self.num_hidden_layers),
            "local_layer_ids names a layer outside the {} the model has",
            self.num_hidden_layers
        );
        anyhow::ensure!(
            self.num_attention_heads % self.num_key_value_heads == 0,
            "{} query heads do not group evenly over {} kv heads",
            self.num_attention_heads,
            self.num_key_value_heads
        );
        anyhow::ensure!(
            self.unpadded_vocab_size <= self.vocab_size,
            "unpadded vocab {} exceeds padded {}",
            self.unpadded_vocab_size,
            self.vocab_size
        );
        anyhow::ensure!(
            (self.rms_norm_eps - 1e-6).abs() < 1e-12,
            "rms_norm_eps {} is not the kernel's 1e-6",
            self.rms_norm_eps
        );
        Ok(())
    }

    /// Whether layer `l` uses the sliding window.
    pub fn is_sliding(&self, l: usize) -> bool {
        self.local_layer_ids.contains(&l)
    }

    /// Layers actually executed per token, counting the looped core once per pass.
    pub fn effective_depth(&self) -> usize {
        let l = &self.loop_config;
        self.num_hidden_layers + (l.core_end - l.core_start) * (l.num_passes - 1)
    }
}
