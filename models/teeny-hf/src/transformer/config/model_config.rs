/*
 * Copyright (c) 2026 Teenygrad.
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

use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize, PartialEq, Clone)]
pub enum Architecture {
    Qwen3ForCausalLM,
}

#[derive(Debug, Serialize, Deserialize, PartialEq, Clone)]
pub enum ModelType {
    #[serde(rename = "qwen3")]
    Qwen3,
}

#[derive(Debug, Serialize, Deserialize, PartialEq, Clone)]
pub enum HiddenAct {
    #[serde(rename = "silu")]
    Silu,
}

#[derive(Debug, Serialize, Deserialize, PartialEq, Clone)]
pub enum TorchDtype {
    #[serde(rename = "bfloat16")]
    Bfloat16,
}

pub trait IPretrainedConfig {
    fn architectures(&self) -> &[Architecture];
    fn attention_bias(&self) -> bool;
    fn attention_dropout(&self) -> f32;
    fn bos_token_id(&self) -> usize;
    fn eos_token_id(&self) -> usize;
    fn head_dim(&self) -> Option<usize>;
    fn hidden_act(&self) -> HiddenAct;
    fn hidden_size(&self) -> usize;
    fn initializer_range(&self) -> f32;
    fn intermediate_size(&self) -> usize;
    fn max_position_embeddings(&self) -> usize;
    fn max_window_layers(&self) -> usize;
    fn model_type(&self) -> ModelType;
    fn num_attention_heads(&self) -> usize;
    fn num_hidden_layers(&self) -> usize;
    fn num_key_value_heads(&self) -> Option<usize>;
    fn rms_norm_eps(&self) -> f32;
    fn rope_scaling(&self) -> Option<f32>;
    fn rope_theta(&self) -> usize;
    fn sliding_window(&self) -> Option<f32>;
    fn tie_word_embeddings(&self) -> bool;
    fn torch_dtype(&self) -> TorchDtype;
    fn transformers_version(&self) -> String;
    fn use_cache(&self) -> bool;
    fn use_sliding_window(&self) -> bool;
    fn vocab_size(&self) -> usize;
    fn partial_rotary_factor(&self) -> Option<f32>;
}
