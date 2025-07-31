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
use serde::{Deserialize, Serialize};
use teeny_cache::DynamicCache;
use teeny_core::dtype::{DtypeEnum, Value};
use teeny_core::graph::ops::Op;
use teeny_core::graph::{self, NodeRef};
use teeny_core::nn::Module;
use teeny_core::nn::embedding::EmbeddingBuilder;
use teeny_core::nn::{embedding::Embedding, linear::Linear};
use teeny_core::num::bf16::bf16;
use teeny_core::safetensors::SafeTensors;
use teeny_core::tensor::{FloatTensor, LongTensor};

use crate::transformer::activations::get_activation;

use crate::error::{Error, Result};
use crate::transformer::model::qwen::qwen3::qwen3_config::{Attention, Qwen3Config, RopeType};
use crate::transformer::util::rope_util::compute_default_rope_parameters;

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Qwen3AttentionType {
    #[serde(rename = "full_attention")]
    FullAttention,

    #[serde(rename = "sliding_attention")]
    SlidingAttention,
}

type MaskFunction = fn(isize, isize, isize, isize) -> bool;

pub struct Qwen3Model<'data> {
    pub vocab_size: usize,
    pub padding_idx: Option<usize>,
    pub embed_tokens: Embedding<'data>,
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
        let weights_safetensor = safetensors.tensor("model.embed_tokens.weight")?;
        let weights = match weights_safetensor.dtype() {
            DtypeEnum::F32 => graph::tensor_f32(weights_safetensor.into_array::<f32>()?),
            DtypeEnum::Bf16 => graph::tensor_bf16(weights_safetensor.into_array::<bf16>()?),
            _ => todo!(),
        };

        Ok(Qwen3Model {
            vocab_size: config.vocab_size,
            padding_idx: config.pad_token_id,
            num_hidden_layers: config.num_hidden_layers,
            embed_tokens: EmbeddingBuilder::default()
                .num_embeddings(config.vocab_size)
                .embedding_dim(config.hidden_size)
                .padding_idx(config.pad_token_id)
                .weight(weights)
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

    fn create_causal_mask(
        &self,
        config: &Qwen3Config,
        input_embeds: NodeRef<'data>,
        attention_mask: Option<NodeRef<'data>>,
        cache_position: NodeRef<'data>,
        past_key_values: Option<DynamicCache>,
        position_ids: Option<NodeRef<'data>>,
        or_mask_function: Option<MaskFunction>,
        and_mask_function: Option<MaskFunction>,
    ) -> Result<Option<NodeRef<'data>>> {
        let layer_idx = past_key_values
            .as_ref()
            .map(|cache| cache.get_sequence_length())
            .unwrap_or(0);

        let (early_exit, attention_mask, packed_sequence_mask, kv_length, kv_offset) =
            Self::preprocess_mask_arguments(
                config,
                &input_embeds,
                &attention_mask,
                &cache_position,
                &past_key_values,
                &position_ids,
                layer_idx,
            );

        if early_exit {
            return Ok(attention_mask);
        }

        let batch_size = input_embeds.shape()?.dims()[0];
        let dtype = input_embeds.dtype();
        let mut mask_factory_function: MaskFunction = Self::causal_mask_function;
        let mask_interface = match config.attn_implementation {
            Attention::FlexAttention => Self::flash_attention_mask,
            Attention::FlashAttention2 => Self::flex_attention_mask,
        };

        let mut allow_is_causal_skip = past_key_values
            .map(|cache| !cache.is_compileable())
            .unwrap_or(true);

        if let Some(packed_sequence_mask) = packed_sequence_mask {
            mask_factory_function = and_masks(
                mask_factory_function,
                packed_sequence_mask_function(packed_sequence_mask),
            );
            allow_is_causal_skip = false;
        }

        if let Some(mask_function) = or_mask_function {
            mask_factory_function = or_masks(mask_factory_function, mask_function);
            allow_is_causal_skip = false;
        }

        if let Some(mask_function) = and_mask_function {
            mask_factory_function = and_masks(mask_factory_function, mask_function);
            allow_is_causal_skip = false;
        }

        let causal_mask = mask_interface(
            &config,
            batch_size,
            cache_position,
            kv_length,
            kv_offset,
            mask_factory_function,
            attention_mask,
            allow_is_causal_skip,
            dtype,
        );

        Ok(causal_mask)
    }

    fn create_sliding_window_causal_mask(&self) -> NodeRef<'data> {
        todo!()
    }

    fn preprocess_mask_arguments(
        _config: &Qwen3Config,
        _input_embeds: &NodeRef<'data>,
        _attention_mask: &Option<NodeRef<'data>>,
        _cache_position: &NodeRef<'data>,
        _past_key_values: &Option<DynamicCache>,
        _position_ids: &Option<NodeRef<'data>>,
        _layer_idx: usize,
    ) -> (
        bool,
        Option<NodeRef<'data>>,
        Option<NodeRef<'data>>,
        Option<NodeRef<'data>>,
        Option<NodeRef<'data>>,
    ) {
        todo!()
    }

    fn causal_mask_function(
        _batch_idx: isize,
        _head_idx: isize,
        q_idx: isize,
        kv_idx: isize,
    ) -> bool {
        kv_idx <= q_idx
    }

    fn flash_attention_mask(
        _batch_size: isize,
        _cache_position: NodeRef<'data>,
        _kv_length: isize,
        _kv_offset: isize,
        _mask_function: Option<fn(isize, isize, isize, isize) -> bool>,
        _attention_mask: Option<NodeRef<'data>>,
    ) -> Option<NodeRef<'data>> {
        let _mask_function = _mask_function.unwrap_or(Self::causal_mask_function);
        todo!()
    }

    //     def flash_attention_mask(
    //     batch_size: int,
    //     cache_position: torch.Tensor,
    //     kv_length: int,
    //     kv_offset: int = 0,
    //     mask_function: Callable = causal_mask_function,
    //     attention_mask: Optional[torch.Tensor] = None,
    //     **kwargs,
    // ):
    //     """
    //     Create the attention mask necesary to use FA2. Since FA2 is un-padded by definition, here we simply return
    //     `None` if the mask is fully causal, or we return the 2D mask which will then be used to extract the seq_lens.
    //     We just slice it in case of sliding window.

    //     Args:
    //         batch_size (`int`):
    //             The batch size of the input sequence.
    //         cache_position (`torch.Tensor`):
    //             A tensor of shape (query_length,) indicating the current indices of the input sequence elements.
    //         kv_length (`int`):
    //             The size that the key and value states will have during the attention computation.
    //         kv_offset (`int`, optional):
    //             An optional offset to indicate at which first position the key and values states will refer to.
    //         mask_function (`Callable`):
    //             The mask factory function describing the mask pattern.
    //         attention_mask (`torch.Tensor`, optional):
    //             The 2D attention mask corresponding to padded tokens of shape (batch_size, number_of_seen_tokens+q_length)
    //     """
    //     if attention_mask is not None:
    //         # Here we need to slice from the right if using sliding or chunked (for full attention, this is equivalent to doing nothing)
    //         attention_mask = attention_mask[:, -kv_length:]
    //         # We only return an actual mask if there is at least 1 padding token, otherwise we return `None` and use `is_causal` in FA2
    //         # (note that the attention_mask is a boolean dtype here)
    //         if attention_mask.all():
    //             attention_mask = None

    //     return attention_mask

    fn flex_attention_mask(
        _batch_size: isize,
        _cache_position: NodeRef<'data>,
        _kv_length: isize,
        _kv_offset: isize,
        _mask_function: Option<fn(isize, isize, isize, isize) -> bool>,
        _attention_mask: Option<NodeRef<'data>>,
    ) -> Option<NodeRef<'data>> {
        todo!()
    }

    // def flex_attention_mask(
    //     batch_size: int,
    //     cache_position: torch.Tensor,
    //     kv_length: int,
    //     kv_offset: int = 0,
    //     mask_function: Callable = causal_mask_function,
    //     attention_mask: Optional[torch.Tensor] = None,
    //     **kwargs,
    // ) -> BlockMask:
    //     """
    //     Create a 4D block mask which is a compressed representation of the full 4D block causal mask. BlockMask is essential
    //     for performant computation of flex attention. See: https://pytorch.org/blog/flexattention/

    //     Args:
    //         batch_size (`int`):
    //             The batch size of the input sequence.
    //         cache_position (`torch.Tensor`):
    //             A tensor of shape (query_length,) indicating the current indices of the input sequence elements.
    //         kv_length (`int`):
    //             The size that the key and value states will have during the attention computation.
    //         kv_offset (`int`, optional):
    //             An optional offset to indicate at which first position the key and values states will refer to.
    //         mask_function (`Callable`):
    //             The mask factory function describing the mask pattern.
    //         attention_mask (`torch.Tensor`, optional):
    //             The 2D attention mask corresponding to padded tokens of shape (batch_size, number_of_seen_tokens+q_length)
    //     """
    //     q_length, q_offset = cache_position.shape[0], cache_position[0]

    //     # Potentially add the padding 2D mask
    //     if attention_mask is not None:
    //         # Older torch (2.5.x) cannot handle sequences not in multiples of 128 (default block size)
    //         # Hence we pad to multiples of this as a minimum to ensure this
    //         pad_len = ((attention_mask.shape[1] // flex_default_block_size) + 1) * flex_default_block_size
    //         pad_len = pad_len - attention_mask.shape[1]
    //         if not _is_torch_greater_or_equal_than_2_6 and pad_len > 0:
    //             attention_mask = torch.nn.functional.pad(attention_mask, value=0, pad=(0, pad_len))

    //         padding_mask = prepare_padding_mask(attention_mask, kv_length, kv_offset, _slice=False)
    //         mask_function = and_masks(mask_function, padding_mask_function(padding_mask))

    //     # Add the offsets on top (because flex interface only allows length, not start and end indices)
    //     mask_function = add_offsets_to_mask_function(mask_function, q_offset, kv_offset)

    //     # Finally create the block mask
    //     block_mask = create_block_mask(
    //         mask_mod=mask_function,
    //         B=batch_size,
    //         H=None,
    //         Q_LEN=q_length,
    //         KV_LEN=kv_length,
    //         device=cache_position.device,
    //         _compile=_is_torch_greater_or_equal_than_2_6,
    //     )
    //     return block_mask
}

pub struct Qwen3ModelOutput<'data> {
    pub hidden_states: NodeRef<'data>,
    pub past_key_values: Option<DynamicCache>,
}

impl<'data> Module<'data, QwenModelInputs<'data>, Qwen3ModelOutput<'data>> for Qwen3Model<'data> {
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
                graph::arange(
                    Value::Usize(past_seen_tokens),
                    Value::Usize(past_seen_tokens + embeds_shape.dims()[1]),
                    Value::Usize(1),
                )
            }
        };

        let position_ids = match position_ids {
            Some(ids) => ids,
            None => cache_position.clone().unsqueeze(0),
        };

        let mut causal_mask_mapping = HashMap::<Qwen3AttentionType, NodeRef<'data>>::new();
        if let Some(_mask) = attention_mask {
            causal_mask_mapping
                .insert(Qwen3AttentionType::FullAttention, self.create_causal_mask());
        }

        if self.has_sliding_layers {
            causal_mask_mapping.insert(
                Qwen3AttentionType::SlidingAttention,
                self.create_sliding_window_causal_mask(),
            );
        }

        let mut hidden_states = inputs_embeds;
        let position_embeddings = self.rotary_emb.forward(position_ids.clone())?;

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
    position_ids: NodeRef<'data>,
    past_key_values: Option<DynamicCache>,
    use_cache: bool,
    cache_position: NodeRef<'data>,
    position_embeddings: (NodeRef<'data>, NodeRef<'data>),
}

impl<'data> Module<'data, &LayerInputs<'data>, NodeRef<'data>> for Qwen3DecoderLayer<'data> {
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

impl<'data> Module<'data, NodeRef<'data>, NodeRef<'data>> for Qwen3RMSNorm<'data> {
    fn forward(&self, hidden_states: NodeRef<'data>) -> Result<NodeRef<'data>> {
        let input_dtype = hidden_states.dtype();
        let hidden_states = hidden_states.to_dtype(DtypeEnum::F32);
        let variance = graph::pow(hidden_states.clone(), graph::scalar(Value::F32(2.0)));
        let variance = graph::mean(variance, Some(-1));
        let hidden_states = hidden_states
            * graph::rsqrt(variance + graph::scalar(Value::F32(self.variance_epsilon)));
        let hidden_states = hidden_states.to_dtype(input_dtype);
        Ok(self.weight.clone() * hidden_states)
    }

    fn parameters(&self) -> Vec<NodeRef<'data>> {
        todo!()
    }
}

type RopeInitFn<'data> =
    Box<dyn Fn(&Qwen3Config, Option<usize>) -> (NodeRef<'data>, NodeRef<'data>)>;

pub struct Qwen3RotaryEmbedding<'data> {
    pub rope_type: RopeType,
    pub max_seq_len_cached: usize,
    pub original_max_seq_len: usize,
    pub attention_scaling: NodeRef<'data>,
    pub inv_freq: NodeRef<'data>,
    pub original_inv_freq: NodeRef<'data>,
    pub rope_init_fn: RopeInitFn<'data>,
}

impl<'data> Qwen3RotaryEmbedding<'data> {
    pub fn new(config: &Qwen3Config) -> Self {
        let rope_type = config
            .rope_scaling
            .as_ref()
            .map(|rope_scaling| rope_scaling.rope_type)
            .unwrap_or(RopeType::Default);

        let rope_init_fn = match rope_type {
            RopeType::Default => compute_default_rope_parameters,
            _ => unimplemented!(),
        };

        let (inv_freq, attention_scaling) = rope_init_fn(config, None);

        Self {
            max_seq_len_cached: config.max_position_embeddings,
            original_max_seq_len: config.max_position_embeddings,
            inv_freq: inv_freq.clone(),
            original_inv_freq: inv_freq,
            attention_scaling,
            rope_type,
            rope_init_fn: Box::new(rope_init_fn),
        }
    }
}

impl<'data> Module<'data, NodeRef<'data>, (NodeRef<'data>, NodeRef<'data>)>
    for Qwen3RotaryEmbedding<'data>
{
    fn forward(&self, position_ids: NodeRef<'data>) -> Result<(NodeRef<'data>, NodeRef<'data>)> {
        let position_ids_shape = position_ids.shape()?;
        let batch_size = position_ids_shape.dims()[0].try_into().unwrap();

        let inv_freq_expanded = self
            .inv_freq
            .clone()
            .unsqueeze(0)
            .unsqueeze(1)
            .to_dtype(DtypeEnum::F32)
            .expand(&[batch_size, -1, 1]);

        let position_ids_expanded = position_ids
            .clone()
            .unsqueeze(1)
            .unsqueeze(-1)
            .to_dtype(DtypeEnum::F32);

        let freqs = (inv_freq_expanded.dot(&position_ids_expanded)).transpose(&[1, 2]);
        let emb = graph::cat(&[freqs.clone(), freqs.clone()], -1);
        let cos = graph::cos(emb.clone()).dot(&self.attention_scaling);
        let sin = graph::sin(emb).dot(&self.attention_scaling);

        Ok((cos, sin))
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

impl<'data> Module<'data, &Qwen3AttentionInputs<'data>, (NodeRef<'data>, NodeRef<'data>)>
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
    pub act_fn: Box<dyn Module<'data, NodeRef<'data>, NodeRef<'data>>>,
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

impl<'data> Module<'data, NodeRef<'data>, NodeRef<'data>> for Qwen3MLP<'data> {
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
