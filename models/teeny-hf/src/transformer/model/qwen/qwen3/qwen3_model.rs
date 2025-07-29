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

use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

use derive_builder::Builder;
use ndarray::Array;
use serde::{Deserialize, Serialize};
use teeny_cache::DynamicCache;
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

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Qwen3AttentionType {
    #[serde(rename = "full_attention")]
    FullAttention,

    #[serde(rename = "sliding_attention")]
    SlidingAttention,
}

pub struct Qwen3Model<'data> {
    pub vocab_size: usize,
    pub padding_idx: Option<usize>,
    pub embed_tokens: Embedding<N>,
    pub layers: Vec<Qwen3DecoderLayer<'data>>,
    pub norm: Qwen3RMSNorm<'data>,
    pub rotary_emb: Qwen3RotaryEmbedding<'data>,
    pub gradient_checkpointing: bool,
    pub has_sliding_layers: bool,
    pub num_hidden_layers: usize,
}

#[derive(Debug, Builder, Clone)]
pub struct QwenModelInputs<'data> {
    pub input_ids: Option<LongTensor<'data>>,
    pub attention_mask: Option<FloatTensor<'data>>,
    pub position_ids: Option<LongTensor<'data>>,
    pub past_key_values: Option<DynamicCache>,
    pub inputs_embeds: Option<FloatTensor<'data>>,
    pub use_cache: bool,
    pub cache_position: Option<LongTensor<'data>>,
}

impl<'data> Qwen3Model<'data> {
    pub fn from_pretrained<T: SafeTensors<'data>>(
        config: &Qwen3Config,
        _cache_dir: &Path,
        safetensors: &'data T,
    ) -> Result<Self> {
        let weights = safetensors.tensor("model.embed_tokens.weight")?;
        let shape = weights.shape();
        assert_eq!(shape[0], config.vocab_size);
        assert_eq!(shape[1], config.hidden_size);
        let data = N::from_bytes(weights.data());
        let weights = Array::from_shape_vec((shape[0], shape[1]), data)?;

        Ok(Qwen3Model {
            vocab_size: config.vocab_size,
            padding_idx: config.pad_token_id,
            num_hidden_layers: config.num_hidden_layers,
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
                .contains(&Qwen3AttentionType::SlidingAttention),
        })
    }

    fn create_causal_mask(&self) -> NodeRef<'data> {
        todo!()
    }

    fn create_sliding_window_causal_mask(&self) -> NodeRef<'data> {
        todo!()
    }
}

pub struct Qwen3ModelOutput<'data> {
    pub hidden_states: NodeRef<'data>,
    pub past_key_values: Option<DynamicCache>,
}

impl<'data> Module<'data, N, QwenModelInputs<'data>, Qwen3ModelOutput<'data>>
    for Qwen3Model<'data>
{
    fn forward(
        &self,
        QwenModelInputs {
            input_ids,
            inputs_embeds,
            use_cache,
            past_key_values,
            cache_position,
            position_ids,
            attention_mask,
            ..
        }: QwenModelInputs<'data>,
    ) -> Result<Qwen3ModelOutput<'data>> {
        if input_ids.is_none() ^ inputs_embeds.is_some() {
            return Err(Error::ModelError(
                "Only one of input_ids and inputs_embeds must be provided.".to_string(),
            )
            .into());
        }

        let inputs_embeds = match inputs_embeds {
            Some(embeds) => embeds,
            None => self.embed_tokens.forward(input_ids.unwrap())?,
        };

        let past_key_values = if use_cache && past_key_values.is_none() {
            Some(DynamicCache::new())
        } else {
            past_key_values
        };

        let cache_position = match cache_position {
            Some(position) => position,
            None => {
                let past_seen_tokens = match &past_key_values {
                    Some(cache) => cache.get_sequence_length(),
                    None => 0,
                };

                let embeds_shape = inputs_embeds.shape()?;
                graph::arange(past_seen_tokens, past_seen_tokens + embeds_shape.dims[1], 1)
            }
        };

        let position_ids = match position_ids {
            Some(ids) => ids,
            None => graph::unsqueeze(cache_position.clone(), 0),
        };

        let mut causal_mask_mapping = HashMap::<Qwen3AttentionType, NodeRef<'data>>::new();
        if let Some(_mask) = attention_mask {
            causal_mask_mapping
                .insert(Qwen3AttentionType::FullAttention, self.create_causal_mask());
            // todo create attention mask
        }

        if self.has_sliding_layers {
            causal_mask_mapping.insert(
                Qwen3AttentionType::SlidingAttention,
                self.create_sliding_window_causal_mask(),
            );
        }

        let mut hidden_states = inputs_embeds;
        let position_embeddings = self
            .rotary_emb
            .forward((hidden_states.clone(), position_ids.clone()))?;

        for layer in self.layers[..self.num_hidden_layers].iter() {
            let layer_inputs = LayerInputs {
                hidden_states,
                attention_mask: causal_mask_mapping[&layer.attention_type].clone(),
                position_ids: position_ids.clone(),
                past_key_values: past_key_values.clone(),
                use_cache,
                cache_position: cache_position.clone(),
                position_embeddings: position_embeddings.clone(),
            };

            hidden_states = layer.forward(&layer_inputs)?;
        }

        hidden_states = self.norm.forward(hidden_states)?;

        Ok(Qwen3ModelOutput {
            hidden_states,
            past_key_values: if use_cache { past_key_values } else { None },
        })
    }

    fn parameters(&self) -> Vec<NodeRef<'data>> {
        todo!()
    }
}

pub struct Qwen3DecoderLayer<'data> {
    pub hidden_size: usize,
    pub self_attn: Qwen3Attention<'data>,
    pub mlp: Qwen3MLP<'data>,
    pub input_layernorm: Qwen3RMSNorm<'data>,
    pub post_attention_layernorm: Qwen3RMSNorm<'data>,
    pub attention_type: Qwen3AttentionType,
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

struct LayerInputs<'data> {
    hidden_states: NodeRef<'data>,
    attention_mask: NodeRef<'data>,
    position_ids: NodeRef<'data, usize>,
    past_key_values: Option<DynamicCache>,
    use_cache: bool,
    cache_position: NodeRef<'data, usize>,
    position_embeddings: (NodeRef<'data>, NodeRef<'data>),
}

impl<'data> Module<'data, N, &LayerInputs<'data>, NodeRef<'data>>
    for Qwen3DecoderLayer<'data>
{
    fn forward(&self, model_inputs: &LayerInputs<'data>) -> Result<NodeRef<'data>> {
        let residual = model_inputs.hidden_states.clone();
        let hidden_states = self
            .input_layernorm
            .forward(model_inputs.hidden_states.clone())?;

        let attention_inputs = Qwen3AttentionInputs {
            hidden_states,
            attention_mask: model_inputs.attention_mask.clone(),
            position_ids: model_inputs.position_ids.clone(),
            past_key_values: model_inputs.past_key_values.clone(),
            use_cache: model_inputs.use_cache,
            cache_position: model_inputs.cache_position.clone(),
            position_embeddings: model_inputs.position_embeddings.clone(),
        };

        let (hidden_states, _) = self.self_attn.forward(&attention_inputs)?;
        let hidden_states = residual + hidden_states;

        let residual = hidden_states.clone();
        let hidden_states = self.post_attention_layernorm.forward(hidden_states)?;
        let hidden_states = self.mlp.forward(hidden_states)?;

        Ok(residual + hidden_states)
    }

    fn parameters(&self) -> Vec<NodeRef<'data>> {
        todo!()
    }
}

pub struct Qwen3RMSNorm<'data> {
    pub weight: NodeRef<'data>,
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

impl<'data> Module<'data, N, NodeRef<'data>, NodeRef<'data>> for Qwen3RMSNorm<'data> {
    fn forward(&self, hidden_states: NodeRef<'data>) -> Result<NodeRef<'data>> {
        // let input_dtype = hidden_states.dtype();
        // let hidden_states = hidden_states.to(torch.float32);
        // let variance = hidden_states.pow(2).mean(-1, keepdim=True)
        // let hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        // return self.weight * hidden_states.to(input_dtype)

        todo!()
    }

    fn parameters(&self) -> Vec<NodeRef<'data>> {
        todo!()
    }
}

pub struct Qwen3RotaryEmbedding<'data> {
    pub max_seq_len_cached: usize,
    pub original_max_seq_len: usize,
    pub attention_scaling: NodeRef<'data>,
    pub original_inv_freq: NodeRef<'data>,
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

impl<'data>
    Module<
        'data,
        N,
        (NodeRef<'data>, NodeRef<'data, usize>),
        (NodeRef<'data>, NodeRef<'data>),
    > for Qwen3RotaryEmbedding<'data>
{
    fn forward(
        &self,
        (_hidden_states, _position_ids): (NodeRef<'data>, NodeRef<'data, usize>),
    ) -> Result<(NodeRef<'data>, NodeRef<'data>)> {
        //    @torch.no_grad()
        // @dynamic_rope_update  # power user: used with advanced RoPE types (e.g. dynamic rope)
        // def forward(self, x, position_ids):
        //     inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(x.device)
        //     position_ids_expanded = position_ids[:, None, :].float()

        //     device_type = x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu"
        //     with torch.autocast(device_type=device_type, enabled=False):  # Force float32
        //         freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
        //         emb = torch.cat((freqs, freqs), dim=-1)
        //         cos = emb.cos() * self.attention_scaling
        //         sin = emb.sin() * self.attention_scaling

        //     return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)
        todo!()
    }

    fn parameters(&self) -> Vec<NodeRef<'data>> {
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
    pub q_proj: Linear<'data>,
    pub k_proj: Linear<'data>,
    pub v_proj: Linear<'data>,
    pub o_proj: Linear<'data>,
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
            sliding_window: if config.layer_types[layer_idx] == Qwen3AttentionType::SlidingAttention
            {
                config.sliding_window
            } else {
                None
            },
        })
    }
}

type Qwen3AttentionInputs<'data> = LayerInputs<'data>;

impl<'data>
    Module<'data, N, &Qwen3AttentionInputs<'data>, (NodeRef<'data>, NodeRef<'data>)>
    for Qwen3Attention<'data>
{
    fn forward(
        &self,
        _model_inputs: &Qwen3AttentionInputs<'data>,
    ) -> Result<(NodeRef<'data>, NodeRef<'data>)> {
        //     def forward(
        //     self,
        //     hidden_states: torch.Tensor,
        //     position_embeddings: tuple[torch.Tensor, torch.Tensor],
        //     attention_mask: Optional[torch.Tensor],
        //     past_key_value: Optional[Cache] = None,
        //     cache_position: Optional[torch.LongTensor] = None,
        //     **kwargs: Unpack[FlashAttentionKwargs],
        // ) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[tuple[torch.Tensor]]]:
        //     input_shape = hidden_states.shape[:-1]
        //     hidden_shape = (*input_shape, -1, self.head_dim)

        //     query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        //     key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        //     value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        //     cos, sin = position_embeddings
        //     query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        //     if past_key_value is not None:
        //         # sin and cos are specific to RoPE models; cache_position needed for the static cache
        //         cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        //         key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        //     attention_interface: Callable = eager_attention_forward
        //     if self.config._attn_implementation != "eager":
        //         attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        //     attn_output, attn_weights = attention_interface(
        //         self,
        //         query_states,
        //         key_states,
        //         value_states,
        //         attention_mask,
        //         dropout=0.0 if not self.training else self.attention_dropout,
        //         scaling=self.scaling,
        //         sliding_window=self.sliding_window,  # diff with Llama
        //         **kwargs,
        //     )

        //     attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        //     attn_output = self.o_proj(attn_output)
        //     return attn_output, attn_weights
        todo!()
    }

    fn parameters(&self) -> Vec<NodeRef<'data>> {
        todo!()
    }
}
pub struct Qwen3MLP<'data> {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub gate_proj: Linear<'data>,
    pub down_proj: Linear<'data>,
    pub up_proj: Linear<'data>,
    pub act_fn: Box<dyn Module<'data, N, NodeRef<'data>, NodeRef<'data>>>,
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

impl<'data> Module<'data, N, NodeRef<'data>, NodeRef<'data>> for Qwen3MLP<'data> {
    fn forward(&self, _model_inputs: NodeRef<'data>) -> Result<NodeRef<'data>> {
        //  def forward(self, x):
        //   down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        //   return down_proj
        todo!()
    }

    fn parameters(&self) -> Vec<NodeRef<'data>> {
        todo!()
    }
}
