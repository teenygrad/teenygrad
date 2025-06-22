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

use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize, PartialEq)]
pub enum Architecture {
    Qwen3ForCausalLM,
}

#[derive(Debug, Serialize, Deserialize, PartialEq)]
pub enum ModelType {
    #[serde(rename = "qwen3")]
    Qwen3,
}

#[derive(Debug, Serialize, Deserialize, PartialEq)]
pub enum HiddenAct {
    #[serde(rename = "silu")]
    Silu,
}

#[derive(Debug, Serialize, Deserialize, PartialEq)]
pub enum TorchDtype {
    #[serde(rename = "bfloat16")]
    Bfloat16,
}

pub trait ModelConfig {
    fn architectures(&self) -> Vec<Architecture>;
    fn attention_bias(&self) -> bool;
    fn attention_dropout(&self) -> f32;
    fn bos_token_id(&self) -> usize;
    fn eos_token_id(&self) -> usize;
    fn head_dim(&self) -> usize;
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
}
