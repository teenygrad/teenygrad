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
    transformer::model::qwen::qwen3::qwen3_model::{Qwen3Model, QwenModelInputs},
};

pub struct Qwen3ForCausalLM<'data> {
    pub model: Qwen3Model<'data>,
    pub vocab_size: usize,
    pub lm_head: Linear<'data>,
}

impl<'data> Qwen3ForCausalLM<'data> {
    pub fn from_pretrained<T: SafeTensors<'data>>(
        config: &Qwen3Config,
        cache_dir: &Path,
        safetensors: &'data T,
    ) -> Result<Self> {
        Ok(Self {
            model: Qwen3Model::from_pretrained(config, cache_dir, safetensors)?,
            vocab_size: config.vocab_size,
            lm_head: Linear::from_pretrained("lm_head", false, safetensors)?,
        })
    }

    pub fn generate(
        &self,
        model_inputs: QwenModelInputs<'data>,
        _max_new_tokens: usize,
    ) -> Result<NodeRef<'data>> {
        self.forward(model_inputs)
    }
}

impl<'data> Module<'data, QwenModelInputs<'data>, NodeRef<'data>> for Qwen3ForCausalLM<'data> {
    fn forward(&self, model_inputs: QwenModelInputs<'data>) -> Result<NodeRef<'data>> {
        let hidden_states = self.model.forward(model_inputs)?;
        self.lm_head.forward(hidden_states.hidden_states)
    }

    fn parameters(&self) -> Vec<NodeRef<'data>> {
        todo!()
    }
}
