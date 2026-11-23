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

//! The subset of Hugging Face's `Gemma3TextConfig` (`config.json`) the port uses.

use std::path::Path;

use serde::Deserialize;

/// Gemma 3 text-model hyperparameters, as in `config.json`.
#[derive(Clone, Debug, Deserialize)]
pub struct GemmaConfig {
    /// Model width.
    pub hidden_size: usize,
    /// MLP width.
    pub intermediate_size: usize,
    /// Decoder layers.
    pub num_hidden_layers: usize,
    /// Query heads.
    pub num_attention_heads: usize,
    /// Key/value heads.
    pub num_key_value_heads: usize,
    /// Lanes per head.
    pub head_dim: usize,
    /// Vocabulary size.
    pub vocab_size: usize,
    /// RMSNorm epsilon (the kernel uses Gemma's 1e-6).
    pub rms_norm_eps: f64,
    /// RoPE base of the full-attention layers.
    pub rope_theta: f64,
    /// RoPE base of the sliding-window layers.
    pub rope_local_base_freq: f64,
    /// Sliding-window size.
    pub sliding_window: usize,
    /// `"sliding_attention"` or `"full_attention"`, per layer.
    pub layer_types: Vec<String>,
    /// Attention logits are scaled by `query_pre_attn_scalar ** -0.5`.
    pub query_pre_attn_scalar: f64,
    /// Token ids that end generation.
    #[serde(default)]
    pub eos_token_id: EosIds,
    /// Must be `None`: the port implements no softcapping.
    pub attn_logit_softcapping: Option<f64>,
    /// Must be `None`: the port implements no softcapping.
    pub final_logit_softcapping: Option<f64>,
}

/// `eos_token_id`, which `config.json` gives as one id or a list.
#[derive(Clone, Debug, Default, Deserialize)]
#[serde(untagged)]
pub enum EosIds {
    /// No end-of-sequence id.
    #[default]
    None,
    /// One id.
    One(u32),
    /// Several ids.
    Many(Vec<u32>),
}

impl EosIds {
    /// The ids as a list.
    pub fn ids(&self) -> Vec<u32> {
        match self {
            EosIds::None => Vec::new(),
            EosIds::One(i) => vec![*i],
            EosIds::Many(v) => v.clone(),
        }
    }
}

impl GemmaConfig {
    /// Reads `config.json`, rejecting features the port does not implement.
    pub fn load(path: &Path) -> anyhow::Result<Self> {
        let text = std::fs::read_to_string(path)
            .map_err(|e| anyhow::anyhow!("reading Gemma config {}: {e}", path.display()))?;
        let cfg: GemmaConfig = serde_json::from_str(&text)
            .map_err(|e| anyhow::anyhow!("parsing Gemma config {}: {e}", path.display()))?;
        anyhow::ensure!(
            cfg.attn_logit_softcapping.is_none() && cfg.final_logit_softcapping.is_none(),
            "logit softcapping is not implemented (config sets it)"
        );
        anyhow::ensure!(
            cfg.layer_types.len() == cfg.num_hidden_layers,
            "config has {} layer_types for {} layers",
            cfg.layer_types.len(),
            cfg.num_hidden_layers
        );
        anyhow::ensure!(
            (cfg.rms_norm_eps - 1e-6).abs() < 1e-12,
            "rms_norm_eps {} is not the kernel's 1e-6",
            cfg.rms_norm_eps
        );
        Ok(cfg)
    }

    /// Whether layer `l` uses the sliding window.
    pub fn is_sliding(&self, l: usize) -> bool {
        self.layer_types[l] == "sliding_attention"
    }
}
