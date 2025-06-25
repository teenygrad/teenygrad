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

use teeny_core::TeenyModule;
use teeny_core::nn::{Embedding, Linear};
use teeny_torch::{torch_nn_embedding, torch_nn_linear, torch_ones};
use teeny_triton::tensor::{DenseTensor, DynamicShape};
use teeny_triton::types::F32;

use crate::transformer::activations::get_activation;
use crate::transformer::model::qwen::qwen2::qwen2_config::IQwen2Config;

use crate::error::{Result, TeenyHFError};
use crate::transformer::util::rope_util::compute_default_rope_parameters;

pub struct Qwen2Model {
    pub vocab_size: usize,
    pub padding_idx: Option<usize>,
    pub embed_tokens: Box<dyn Embedding>,
    pub layers: Vec<Box<dyn TeenyModule<Err = TeenyHFError>>>,
    pub norm: Qwen2RMSNorm,
    pub rotary_emb: Qwen2RotaryEmbedding,
    pub gradient_checkpointing: bool,
    pub has_sliding_layers: bool,
}

impl Qwen2Model {
    pub fn new(config: &impl IQwen2Config) -> Result<Self> {
        Ok(Qwen2Model {
            vocab_size: config.vocab_size(),
            padding_idx: config.pad_token_id(),
            embed_tokens: torch_nn_embedding(
                config.vocab_size(),
                config.hidden_size(),
                config.pad_token_id(),
            ),
            layers: (0..config.num_hidden_layers())
                .map(|layer_idx| Qwen2DecoderLayer::new(config, layer_idx))
                .map(|layer| Box::new(layer) as Box<dyn TeenyModule<Err = TeenyHFError>>)
                .collect::<Vec<_>>(),
            norm: Qwen2RMSNorm::new(config.hidden_size(), config.rms_norm_eps()),
            rotary_emb: Qwen2RotaryEmbedding::new(config),
            gradient_checkpointing: false,
            has_sliding_layers: config
                .layer_types()
                .contains(&"sliding_attention".to_string()),
        })
    }
}

impl TeenyModule for Qwen2Model {
    type Err = TeenyHFError;

    fn forward(&self, _model_inputs: &[u32]) -> Result<Vec<u32>> {
        todo!()
    }
}

pub struct Qwen2DecoderLayer {
    pub hidden_size: usize,
    pub self_attn: Qwen2Attention,
    pub mlp: Qwen2MLP,
    pub input_layernorm: Qwen2RMSNorm,
    pub post_attention_layernorm: Qwen2RMSNorm,
    pub attention_type: String,
}

impl Qwen2DecoderLayer {
    pub fn new(config: &impl IQwen2Config, layer_idx: usize) -> Self {
        Self {
            hidden_size: config.hidden_size(),
            self_attn: Qwen2Attention::new(config, layer_idx),
            mlp: Qwen2MLP::new(config),
            input_layernorm: Qwen2RMSNorm::new(config.hidden_size(), config.rms_norm_eps()),
            post_attention_layernorm: Qwen2RMSNorm::new(
                config.hidden_size(),
                config.rms_norm_eps(),
            ),
            attention_type: config.layer_types()[layer_idx].clone(),
        }
    }
}

impl TeenyModule for Qwen2DecoderLayer {
    type Err = TeenyHFError;

    fn forward(&self, _model_inputs: &[u32]) -> Result<Vec<u32>> {
        todo!()
    }
}

pub struct Qwen2RMSNorm {
    pub weight: DenseTensor<DynamicShape, F32>,
    pub variance_epsilon: f32,
}

impl Qwen2RMSNorm {
    pub fn new(hidden_size: usize, eps: f32) -> Self {
        Self {
            weight: torch_ones(&[hidden_size]),
            variance_epsilon: eps,
        }
    }
}

impl TeenyModule for Qwen2RMSNorm {
    type Err = TeenyHFError;

    fn forward(&self, _model_inputs: &[u32]) -> Result<Vec<u32>> {
        todo!()
    }
}

pub struct Qwen2RotaryEmbedding {
    pub max_seq_len_cached: usize,
    pub original_max_seq_len: usize,
    pub attention_scaling: f32,
    pub original_inv_freq: DenseTensor<DynamicShape, F32>,
}

impl Qwen2RotaryEmbedding {
    pub fn new(config: &impl IQwen2Config) -> Self {
        let (inv_freq, attention_scaling) = compute_default_rope_parameters(config);

        Self {
            max_seq_len_cached: config.max_position_embeddings(),
            original_max_seq_len: config.max_position_embeddings(),
            original_inv_freq: inv_freq,
            attention_scaling,
        }
    }
}

impl TeenyModule for Qwen2RotaryEmbedding {
    type Err = TeenyHFError;

    fn forward(&self, _model_inputs: &[u32]) -> Result<Vec<u32>> {
        todo!()
    }
}

pub struct Qwen2Attention {
    pub layer_idx: usize,
    pub head_dim: usize,
    pub num_key_value_groups: usize,
    pub scaling: f32,
    pub attention_dropout: f32,
    pub is_causal: bool,
    pub q_proj: Box<dyn Linear>,
    pub k_proj: Box<dyn Linear>,
    pub v_proj: Box<dyn Linear>,
    pub o_proj: Box<dyn Linear>,
    pub sliding_window: Option<f32>,
}

impl Qwen2Attention {
    pub fn new(config: &impl IQwen2Config, layer_idx: usize) -> Self {
        assert!(config.num_key_value_heads().is_some());

        let num_key_value_heads = config.num_key_value_heads().unwrap();
        let head_dim = config
            .head_dim()
            .unwrap_or(config.hidden_size() / config.num_attention_heads());

        Self {
            layer_idx,
            head_dim,
            num_key_value_groups: config.num_attention_heads() / num_key_value_heads,
            scaling: (head_dim as f32).powf(-0.5),
            attention_dropout: config.attention_dropout(),
            is_causal: true,
            q_proj: torch_nn_linear(
                config.hidden_size(),
                config.num_attention_heads() * head_dim,
                true,
            ),
            k_proj: torch_nn_linear(config.hidden_size(), num_key_value_heads * head_dim, true),
            v_proj: torch_nn_linear(config.hidden_size(), num_key_value_heads * head_dim, true),
            o_proj: torch_nn_linear(
                config.num_attention_heads() * head_dim,
                config.hidden_size(),
                false,
            ),
            sliding_window: if config.layer_types()[layer_idx] == "sliding_attention" {
                config.sliding_window()
            } else {
                None
            },
        }
    }
}

pub struct Qwen2MLP {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub gate_proj: Box<dyn Linear>,
    pub up_proj: Box<dyn Linear>,
    pub down_proj: Box<dyn Linear>,
    pub act_fn: Box<dyn TeenyModule<Err = TeenyHFError>>,
}

impl Qwen2MLP {
    pub fn new(config: &impl IQwen2Config) -> Self {
        Self {
            hidden_size: config.hidden_size(),
            intermediate_size: config.intermediate_size(),
            gate_proj: torch_nn_linear(config.hidden_size(), config.intermediate_size(), false),
            up_proj: torch_nn_linear(config.hidden_size(), config.intermediate_size(), false),
            down_proj: torch_nn_linear(config.intermediate_size(), config.hidden_size(), false),
            act_fn: get_activation(config.hidden_act()),
        }
    }
}

impl TeenyModule for Qwen2MLP {
    type Err = TeenyHFError;

    fn forward(&self, _model_inputs: &[u32]) -> std::result::Result<Vec<u32>, Self::Err> {
        todo!()
    }
}
