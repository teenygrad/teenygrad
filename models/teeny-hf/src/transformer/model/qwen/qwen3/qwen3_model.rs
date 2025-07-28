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
use std::sync::Arc;

use derive_builder::Builder;
use ndarray::Array;
use teeny_cache::Cache;
use teeny_core::dtype::Dtype;
use teeny_core::graph::{self, NodeRef};
use teeny_core::nn::Module;
use teeny_core::nn::embedding::EmbeddingBuilder;
use teeny_core::nn::{embedding::Embedding, linear::Linear};
use teeny_core::safetensors::SafeTensors;
use teeny_core::tensor::{FloatTensor, LongTensor};

use crate::transformer::activations::get_activation;

use crate::error::{Error, Result};
use crate::transformer::model::qwen::qwen3::qwen3_config::Qwen3Config;
use crate::transformer::util::rope_util::compute_default_rope_parameters;

pub struct Qwen3Model<'data, N: Dtype> {
    pub vocab_size: usize,
    pub padding_idx: Option<usize>,
    pub embed_tokens: Embedding<N>,
    pub layers: Vec<Qwen3DecoderLayer<'data>>,
    pub norm: Qwen3RMSNorm<'data>,
    pub rotary_emb: Qwen3RotaryEmbedding<'data>,
    pub gradient_checkpointing: bool,
    pub has_sliding_layers: bool,
}

#[derive(Debug, Builder, Clone)]
pub struct QwenModelInputs<'data> {
    pub input_ids: Option<LongTensor<'data>>,
    pub attention_mask: Option<FloatTensor<'data, f32>>,
    pub position_ids: Option<LongTensor<'data>>,
    pub past_key_values: Option<Cache>,
    pub inputs_embeds: Option<FloatTensor<'data, f32>>,
    pub use_cache: Option<bool>,
    pub cache_position: Option<LongTensor<'data>>,
}

impl<'data, N: Dtype> Qwen3Model<'data, N> {
    pub fn from_pretrained<T: SafeTensors<'data>>(
        config: &Qwen3Config,
        _cache_dir: &Path,
        safetensors: &'data T,
    ) -> Result<Self> {
        let weights = safetensors.tensor("model.embed_tokens.weight")?;
        let shape = weights.shape();
        let data = N::from_bytes(weights.data());
        let weights = Array::from_shape_vec((shape[0], shape[1]), data)?;

        Ok(Qwen3Model {
            vocab_size: config.vocab_size,
            padding_idx: config.pad_token_id,
            embed_tokens: EmbeddingBuilder::default()
                .num_embeddings(config.vocab_size)
                .embedding_dim(config.hidden_size)
                .padding_idx(config.pad_token_id)
                .weight(weights.into_dyn())
                .build()
                .map_err(|e| Error::BuilderError(Arc::new(e)))?,
            layers: (0..config.num_hidden_layers)
                .map(|layer_idx| Qwen3DecoderLayer::from_pretrained(config, layer_idx, safetensors))
                .collect::<Result<Vec<_>>>()?,
            norm: Qwen3RMSNorm::from_pretrained(
                "model.norm.weight",
                safetensors,
                config.rms_norm_eps,
            )?,
            rotary_emb: Qwen3RotaryEmbedding::new(config),
            gradient_checkpointing: false,
            has_sliding_layers: config
                .layer_types
                .contains(&"sliding_attention".to_string()),
        })
    }
}

impl<'data, N: Dtype> Module<'data, N, QwenModelInputs<'data>, NodeRef<'data, N>>
    for Qwen3Model<'data, N>
{
    fn forward(
        &self,
        QwenModelInputs {
            input_ids,
            inputs_embeds,
            ..
        }: QwenModelInputs<'data>,
    ) -> Result<NodeRef<'data, N>> {
        if input_ids.is_none() ^ inputs_embeds.is_some() {
            return Err(Error::ModelError(
                "Only one of input_ids and inputs_embeds must be provided.".to_string(),
            )
            .into());
        }

        let _inputs_embeds = inputs_embeds.ok_or(|| self.embed_tokens.forward(input_ids.unwrap()));

        todo!()
    }

    fn parameters(&self) -> Vec<NodeRef<'data, N>> {
        todo!()
    }
}

pub struct Qwen3DecoderLayer<'data> {
    pub hidden_size: usize,
    pub self_attn: Qwen3Attention<'data>,
    pub mlp: Qwen3MLP<'data>,
    pub input_layernorm: Qwen3RMSNorm<'data>,
    pub post_attention_layernorm: Qwen3RMSNorm<'data>,
    pub attention_type: String,
}

impl<'data> Qwen3DecoderLayer<'data> {
    pub fn from_pretrained<T: SafeTensors<'data>>(
        config: &Qwen3Config,
        layer_idx: usize,
        safetensors: &'data T,
    ) -> Result<Self> {
        Ok(Self {
            hidden_size: config.hidden_size,
            self_attn: Qwen3Attention::from_pretrained(config, layer_idx, safetensors)?,
            mlp: Qwen3MLP::from_pretrained(config, layer_idx, safetensors)?,
            input_layernorm: Qwen3RMSNorm::from_pretrained(
                &format!("model.layers.{layer_idx}.input_layernorm.weight"),
                safetensors,
                config.rms_norm_eps,
            )?,
            post_attention_layernorm: Qwen3RMSNorm::from_pretrained(
                &format!("model.layers.{layer_idx}.post_attention_layernorm.weight"),
                safetensors,
                config.rms_norm_eps,
            )?,
            attention_type: config.layer_types[layer_idx].clone(),
        })
    }
}

impl<'data> Module<'data, f32, NodeRef<'data, f32>, NodeRef<'data, f32>>
    for Qwen3DecoderLayer<'data>
{
    fn forward(&self, _model_inputs: NodeRef<'data, f32>) -> Result<NodeRef<'data, f32>> {
        todo!()
    }

    fn parameters(&self) -> Vec<teeny_core::graph::NodeRef<'data, f32>> {
        todo!()
    }
}

pub struct Qwen3RMSNorm<'data> {
    pub weight: NodeRef<'data, f32>,
    pub variance_epsilon: f32,
}

impl<'data> Qwen3RMSNorm<'data> {
    pub fn from_pretrained<T: SafeTensors<'data>>(
        name: &str,
        safetensors: &'data T,
        rms_norm_eps: f32,
    ) -> Result<Self> {
        Ok(Self {
            weight: graph::safetensor_with_name(name, safetensors)?,
            variance_epsilon: rms_norm_eps,
        })
    }
}

impl<'data> Module<'data, f32, NodeRef<'data, f32>, NodeRef<'data, f32>> for Qwen3RMSNorm<'data> {
    fn forward(&self, _model_inputs: NodeRef<'data, f32>) -> Result<NodeRef<'data, f32>> {
        todo!()
    }

    fn parameters(&self) -> Vec<NodeRef<'data, f32>> {
        todo!()
    }
}

pub struct Qwen3RotaryEmbedding<'data> {
    pub max_seq_len_cached: usize,
    pub original_max_seq_len: usize,
    pub attention_scaling: NodeRef<'data, f32>,
    pub original_inv_freq: NodeRef<'data, f32>,
}

impl<'data> Qwen3RotaryEmbedding<'data> {
    pub fn new(config: &Qwen3Config) -> Self {
        let (inv_freq, attention_scaling) = compute_default_rope_parameters(config);

        Self {
            max_seq_len_cached: config.max_position_embeddings,
            original_max_seq_len: config.max_position_embeddings,
            original_inv_freq: inv_freq,
            attention_scaling,
        }
    }
}

impl<'data> Module<'data, f32, NodeRef<'data, f32>, NodeRef<'data, f32>>
    for Qwen3RotaryEmbedding<'data>
{
    fn forward(&self, _model_inputs: NodeRef<'data, f32>) -> Result<NodeRef<'data, f32>> {
        todo!()
    }

    fn parameters(&self) -> Vec<NodeRef<'data, f32>> {
        todo!()
    }
}

pub struct Qwen3Attention<'data> {
    pub layer_idx: usize,
    pub head_dim: usize,
    pub num_key_value_groups: usize,
    pub scaling: f32,
    pub attention_dropout: f32,
    pub is_causal: bool,
    pub q_proj: Linear<'data, f32>,
    pub k_proj: Linear<'data, f32>,
    pub v_proj: Linear<'data, f32>,
    pub o_proj: Linear<'data, f32>,
    pub q_norm: Qwen3RMSNorm<'data>,
    pub k_norm: Qwen3RMSNorm<'data>,
    pub sliding_window: Option<f32>,
}

impl<'data> Qwen3Attention<'data> {
    pub fn from_pretrained<T: SafeTensors<'data>>(
        config: &Qwen3Config,
        layer_idx: usize,
        safetensors: &'data T,
    ) -> Result<Self> {
        let head_dim = config
            .head_dim
            .unwrap_or(config.hidden_size / config.num_attention_heads);
        let num_key_value_heads = config.num_key_value_heads.ok_or(Error::ModelError(
            "num_key_value_heads is required for Qwen3Attention.".to_string(),
        ))?;

        Ok(Self {
            layer_idx,
            head_dim,
            num_key_value_groups: config.num_attention_heads / num_key_value_heads,
            scaling: (head_dim as f32).powf(-0.5),
            attention_dropout: config.attention_dropout,
            is_causal: true,
            q_proj: Linear::from_pretrained(
                &format!("model.layers.{layer_idx}.self_attn.q_proj"),
                false,
                safetensors,
            )?,
            k_proj: Linear::from_pretrained(
                &format!("model.layers.{layer_idx}.self_attn.k_proj"),
                false,
                safetensors,
            )?,
            v_proj: Linear::from_pretrained(
                &format!("model.layers.{layer_idx}.self_attn.v_proj"),
                false,
                safetensors,
            )?,
            o_proj: Linear::from_pretrained(
                &format!("model.layers.{layer_idx}.self_attn.o_proj"),
                false,
                safetensors,
            )?,
            q_norm: Qwen3RMSNorm::from_pretrained(
                &format!("model.layers.{layer_idx}.self_attn.q_norm.weight"),
                safetensors,
                config.rms_norm_eps,
            )?,
            k_norm: Qwen3RMSNorm::from_pretrained(
                &format!("model.layers.{layer_idx}.self_attn.k_norm.weight"),
                safetensors,
                config.rms_norm_eps,
            )?,
            sliding_window: if config.layer_types[layer_idx] == "sliding_attention" {
                config.sliding_window
            } else {
                None
            },
        })
    }
}

pub struct Qwen3MLP<'data> {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub gate_proj: Linear<'data, f32>,
    pub down_proj: Linear<'data, f32>,
    pub up_proj: Linear<'data, f32>,
    pub act_fn: Box<dyn Module<'data, f32, NodeRef<'data, f32>, NodeRef<'data, f32>>>,
}

impl<'data> Qwen3MLP<'data> {
    pub fn from_pretrained<T: SafeTensors<'data>>(
        config: &Qwen3Config,
        layer_idx: usize,
        safetensors: &'data T,
    ) -> Result<Self> {
        let activation = get_activation(&config.hidden_act)?;

        Ok(Self {
            hidden_size: config.hidden_size,
            intermediate_size: config.intermediate_size,
            gate_proj: Linear::from_pretrained(
                &format!("model.layers.{layer_idx}.mlp.gate_proj"),
                false,
                safetensors,
            )?,
            down_proj: Linear::from_pretrained(
                &format!("model.layers.{layer_idx}.mlp.down_proj"),
                false,
                safetensors,
            )?,
            up_proj: Linear::from_pretrained(
                &format!("model.layers.{layer_idx}.mlp.up_proj"),
                false,
                safetensors,
            )?,
            act_fn: activation,
        })
    }
}

impl<'data> Module<'data, f32, NodeRef<'data, f32>, NodeRef<'data, f32>> for Qwen3MLP<'data> {
    fn forward(&self, _model_inputs: NodeRef<'data, f32>) -> Result<NodeRef<'data, f32>> {
        todo!()
    }

    fn parameters(&self) -> Vec<NodeRef<'data, f32>> {
        todo!()
    }
}
