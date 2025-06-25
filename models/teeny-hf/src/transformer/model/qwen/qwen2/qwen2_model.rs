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
use teeny_core::nn::Embedding;
use teeny_torch::{torch_nn_embedding, torch_ones};
use teeny_triton::tensor::{DenseTensor, DynamicShape};
use teeny_triton::types::F32;

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
}

impl Qwen2DecoderLayer {
    pub fn new(_config: &impl IQwen2Config, layer_idx: usize) -> Self {
        //    super().__init__()
        //     self.hidden_size = config.hidden_size

        //     self.self_attn = Qwen2Attention(config=config, layer_idx=layer_idx)

        //     self.mlp = Qwen2MLP(config)
        //     self.input_layernorm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        //     self.post_attention_layernorm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        //     self.attention_type = config.layer_types[layer_idx]
        Self {
            hidden_size: _config.hidden_size(),
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

pub struct Qwen2Attention {}

impl Qwen2Attention {
    pub fn new(_config: &impl IQwen2Config, _layer_idx: usize) -> Self {
        todo!()
    }
}
