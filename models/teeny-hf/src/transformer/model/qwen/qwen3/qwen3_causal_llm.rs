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

// use teeny_core::{TeenyModule, nn::Linear};
// use teeny_torch::torch_nn_linear;

// use super::qwen3_config::Qwen3Config;

// use crate::{
//     error::{Result, TeenyHFError},
//     transformer::model::qwen::qwen2::qwen2_model::Qwen2Model,
// };

// pub struct Qwen3ForCausalLM {
//     pub model: Qwen2Model,
//     pub vocab_size: usize,
//     pub lm_head: Box<dyn Linear>,
// }

// impl Qwen3ForCausalLM {
//     pub fn new(config: &Qwen3Config) -> Result<Self> {
//         Ok(Self {
//             model: Qwen2Model::new(config)?,
//             vocab_size: config.vocab_size,
//             lm_head: torch_nn_linear(config.hidden_size, config.vocab_size, false),
//         })
//     }

//     pub fn generate(&self, model_inputs: &[u32], _max_new_tokens: usize) -> Result<Vec<u32>> {
//         self.forward(model_inputs)
//     }
// }

// impl TeenyModule for Qwen3ForCausalLM {
//     type Err = TeenyHFError;

//     fn forward(&self, model_inputs: &[u32]) -> Result<Vec<u32>> {
//         self.model.forward(model_inputs)
//     }
// }
