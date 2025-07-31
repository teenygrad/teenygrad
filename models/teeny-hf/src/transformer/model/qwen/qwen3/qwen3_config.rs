/*
 * Copyright (c) 2025 Teenygrad. All rights reserved.
 *
 * This program is free software: you can redistribute it and/or modify it
 * under the terms of the GNU General Public License as published by the
 * Free Software Foundation, either version 3 of the License, or (at your
 * option) any later version.
 *
 * This program is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program. If not, see <https://www.gnu.org/licenses/>.
 */

use std::path::Path;
use std::str::FromStr;
use std::{collections::HashMap, vec};

use serde::{Deserialize, Serialize};

use crate::transformer::config::model_config::{Architecture, HiddenAct, ModelType, TorchDtype};

use crate::error::{Error, Result};
use crate::transformer::model::qwen::qwen3::qwen3_model::Qwen3AttentionType;

#[derive(Debug, Copy, Serialize, Deserialize, PartialEq, Clone)]
pub enum RopeType {
    #[serde(rename = "default")]
    Default,

    #[serde(rename = "linear")]
    Linear,

    #[serde(rename = "dynamic")]
    Dynamic,

    #[serde(rename = "yarn")]
    Yarn,

    #[serde(rename = "longrope")]
    LongRope,

    #[serde(rename = "llama")]
    Llama,
}

#[derive(Debug, Serialize, Deserialize, PartialEq, Clone)]
pub struct RopeScaling {
    pub rope_type: RopeType,
}

#[derive(Debug, Serialize, Deserialize, PartialEq, Clone)]
pub enum Attention {
    #[serde(rename = "flex_attention")]
    FlexAttention,

    #[serde(rename = "flash_attention_2")]
    FlashAttention2,
}

impl Default for Attention {
    fn default() -> Self {
        Attention::FlashAttention2
    }
}

#[derive(Debug, Serialize, Deserialize, PartialEq, Clone)]
pub struct Qwen3Config {
    pub architectures: Vec<Architecture>,
    pub attention_bias: bool,
    pub attention_dropout: f32,
    pub bos_token_id: usize,
    pub eos_token_id: usize,
    #[serde(default)]
    pub head_dim: Option<usize>,
    pub hidden_act: HiddenAct,
    pub hidden_size: usize,
    pub initializer_range: f32,
    pub intermediate_size: usize,
    pub max_position_embeddings: usize,
    pub max_window_layers: usize,
    pub model_type: ModelType,
    pub num_attention_heads: usize,
    pub num_hidden_layers: usize,
    pub num_key_value_heads: Option<usize>,
    pub rms_norm_eps: f32,
    pub rope_scaling: Option<RopeScaling>,
    pub rope_theta: f32,
    pub sliding_window: Option<f32>,
    pub tie_word_embeddings: bool,
    pub torch_dtype: TorchDtype,
    pub transformers_version: String,
    pub use_cache: bool,
    pub use_sliding_window: bool,
    pub vocab_size: usize,

    #[serde(default)]
    pub attn_implementation: Attention,

    #[serde(default)]
    pub keys_to_ignore_at_inference: Vec<String>,

    #[serde(default)]
    pub base_model_tp_plan: HashMap<String, String>,

    #[serde(default)]
    pub base_model_pp_plan: HashMap<String, [Vec<String>; 2]>,

    #[serde(default)]
    pub layer_types: Vec<Qwen3AttentionType>,

    #[serde(default)]
    pub pad_token_id: Option<usize>,

    #[serde(default)]
    pub partial_rotary_factor: Option<f32>,
}

impl Qwen3Config {
    pub fn from_pretrained(model_id: &str, cache_dir: &Path) -> Result<Self> {
        let config_path = cache_dir.join(model_id).join("config.json");
        let config_str = std::fs::read_to_string(config_path).map_err(Error::IoError)?;
        let config = Self::from_str(&config_str)?;

        Ok(config)
    }
}

impl FromStr for Qwen3Config {
    type Err = Error;

    fn from_str(config_str: &str) -> std::result::Result<Self, Self::Err> {
        let mut config: Self = serde_json::from_str(config_str).map_err(Error::ConfigParseError)?;

        config.keys_to_ignore_at_inference = vec!["past_key_values".to_string()];

        config.base_model_tp_plan = serde_json::from_str(
            r#"
        {
            "layers.*.self_attn.q_proj": "colwise",
            "layers.*.self_attn.k_proj": "colwise",
            "layers.*.self_attn.v_proj": "colwise",
            "layers.*.self_attn.o_proj": "rowwise"
        }"#,
        )?;

        config.base_model_pp_plan = serde_json::from_str(
            r#"
        {
            "embed_tokens": [["input_ids"], ["inputs_embeds"]],
            "layers": [["hidden_states", "attention_mask"], ["hidden_states"]],
            "norm": [["hidden_states"], ["hidden_states"]]
        }"#,
        )?;

        if config.num_key_value_heads.is_none() {
            config.num_key_value_heads = Some(config.num_attention_heads);
        }

        if config.rope_scaling.is_some() {
            unimplemented!("AXM FIXME: rope_scaling is not supported");
        }

        if config.layer_types.is_empty() {
            config.layer_types = (0..config.num_hidden_layers)
                .map(|i| {
                    if config.sliding_window.is_some() && i >= config.max_window_layers {
                        Qwen3AttentionType::SlidingAttention
                    } else {
                        Qwen3AttentionType::FullAttention
                    }
                })
                .collect();
        }

        Ok(config)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_qwen3_config() {
        let config_str = r#"
        {
          "architectures": [
            "Qwen3ForCausalLM"
          ],
          "attention_bias": false,
          "attention_dropout": 0.0,
          "bos_token_id": 151643,
          "eos_token_id": 151645,
          "head_dim": 128,
          "hidden_act": "silu",
          "hidden_size": 2048,
          "initializer_range": 0.02,
          "intermediate_size": 6144,
          "max_position_embeddings": 40960,
          "max_window_layers": 28,
          "model_type": "qwen3",
          "num_attention_heads": 16,
          "num_hidden_layers": 28,
          "num_key_value_heads": 8,
          "rms_norm_eps": 1e-06,
          "rope_scaling": null,
          "rope_theta": 1000000,
          "sliding_window": null,
          "tie_word_embeddings": true,
          "torch_dtype": "bfloat16",
          "transformers_version": "4.51.0",
          "use_cache": true,
          "use_sliding_window": false,
          "vocab_size": 151936
        }
        "#;

        let config = Qwen3Config::from_str(config_str).unwrap();
        assert_eq!(config.model_type, ModelType::Qwen3);
    }
}
