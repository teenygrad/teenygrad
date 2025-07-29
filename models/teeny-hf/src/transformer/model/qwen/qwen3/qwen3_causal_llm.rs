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
    dtype::Dtype,
    graph::NodeRef,
    nn::{Module, linear::Linear},
    safetensors::SafeTensors,
};

use super::qwen3_config::Qwen3Config;

use crate::{
    error::Result,
    transformer::model::qwen::qwen3::qwen3_model::{Qwen3Model, QwenModelInputs},
};

pub struct Qwen3ForCausalLM<'data, N: Dtype> {
    pub model: Qwen3Model<'data, N>,
    pub vocab_size: usize,
    pub lm_head: Linear<'data, N>,
}

impl<'data, N: Dtype> Qwen3ForCausalLM<'data, N> {
    pub fn from_pretrained<T: SafeTensors<'data>>(
        config: Qwen3Config,
        cache_dir: &Path,
        safetensors: &'data T,
    ) -> Result<Self> {
        Ok(Self {
            model: Qwen3Model::from_pretrained(&config, cache_dir, safetensors)?,
            vocab_size: config.vocab_size,
            lm_head: Linear::from_pretrained("lm_head", false, safetensors)?,
        })
    }

    pub fn generate(
        &self,
        model_inputs: QwenModelInputs<'data, N>,
        _max_new_tokens: usize,
    ) -> Result<NodeRef<'data, N>> {
        self.forward(model_inputs)
    }
}

impl<'data, N: Dtype> Module<'data, N, QwenModelInputs<'data, N>, NodeRef<'data, N>>
    for Qwen3ForCausalLM<'data, N>
{
    fn forward(&self, model_inputs: QwenModelInputs<'data, N>) -> Result<NodeRef<'data, N>> {
        let hidden_states = self.model.forward(model_inputs)?;
        self.lm_head.forward(hidden_states.hidden_states)
    }

    fn parameters(&self) -> Vec<NodeRef<'data, N>> {
        todo!()
    }
}
