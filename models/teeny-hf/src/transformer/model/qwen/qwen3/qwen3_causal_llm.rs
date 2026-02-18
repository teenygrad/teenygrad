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

// use std::path::Path;

// use teeny_core::{
//     graph::NodeRef,
//     nn::{Module, linear::Linear},
//     safetensors::SafeTensors,
// };

// use super::qwen3_config::Qwen3Config;

// use crate::{
//     error::Result,
//     transformer::model::qwen::qwen3::qwen3_model::{Qwen3Model, QwenModelInputs},
// };

// pub struct Qwen3ForCausalLM<'data> {
//     pub model: Qwen3Model<'data>,
//     pub vocab_size: usize,
//     pub lm_head: Linear<'data>,
// }

// impl<'data> Qwen3ForCausalLM<'data> {
//     pub fn from_pretrained<T: SafeTensors<'data>>(
//         config: &Qwen3Config,
//         cache_dir: &Path,
//         safetensors: &'data T,
//     ) -> Result<Self> {
//         Ok(Self {
//             model: Qwen3Model::from_pretrained(config, cache_dir, safetensors)?,
//             vocab_size: config.vocab_size,
//             lm_head: Linear::from_pretrained("lm_head", false, safetensors)?,
//         })
//     }

//     pub fn generate(
//         &mut self,
//         model_inputs: QwenModelInputs<'data>,
//         _max_new_tokens: usize,
//     ) -> Result<NodeRef<'data>> {
//         self.forward(model_inputs)
//     }
// }

// impl<'data> Module<'data, QwenModelInputs<'data>, NodeRef<'data>> for Qwen3ForCausalLM<'data> {
//     fn forward(&mut self, model_inputs: QwenModelInputs<'data>) -> Result<NodeRef<'data>> {
//         let hidden_states = self.model.forward(model_inputs)?;
//         self.lm_head.forward(hidden_states.hidden_states)
//     }

//     fn parameters(&self) -> Vec<NodeRef<'data>> {
//         todo!()
//     }
// }
