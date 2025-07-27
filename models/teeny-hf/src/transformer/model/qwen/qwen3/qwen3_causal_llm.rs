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

use teeny_core::{
    graph::NodeRef,
    nn::{Module, linear::Linear},
    safetensors::SafeTensors,
};

use super::qwen3_config::Qwen3Config;

use crate::{
    error::Result,
    transformer::model::qwen::{
        qwen2::qwen2_model::{Qwen2Model, QwenModelInputs},
        qwen3::qwen3_model::Qwen3Model,
    },
};

pub struct Qwen3ForCausalLM {
    pub model: Qwen2Model,
    pub vocab_size: usize,
    pub lm_head: Linear<f32>,
}

impl Qwen3ForCausalLM {
    pub fn from_pretrained<'data, T: SafeTensors<'data>>(
        config: &Qwen3Config,
        cache_dir: &Path,
        _safetensors: &T,
    ) -> Result<Self> {
        Ok(Self {
            model: Qwen3Model::from_pretrained(config, cache_dir)?,
            vocab_size: config.vocab_size,
            lm_head: Linear::new("lm_head", config.hidden_size, config.vocab_size, false)?,
        })
    }

    pub fn generate(
        &self,
        model_inputs: QwenModelInputs,
        _max_new_tokens: usize,
    ) -> Result<NodeRef<f32>> {
        self.forward(model_inputs)
    }
}

impl Module<f32, QwenModelInputs, NodeRef<f32>> for Qwen3ForCausalLM {
    fn forward(&self, model_inputs: QwenModelInputs) -> Result<NodeRef<f32>> {
        let hidden_states = self.model.forward(model_inputs)?;
        self.lm_head.forward(hidden_states)
    }

    fn parameters(&self) -> Vec<NodeRef<f32>> {
        todo!()
    }
}
