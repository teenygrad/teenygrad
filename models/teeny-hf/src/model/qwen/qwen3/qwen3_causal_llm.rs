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

use super::qwen3_config::Qwen3Config;

use crate::{error::Result, model::Model};

pub struct Qwen3ForCausalLM {}

impl Model for Qwen3ForCausalLM {
    fn generate(&self, _model_inputs: &[u32], _max_new_tokens: usize) -> Vec<u32> {
        todo!()
    }
}

impl Qwen3ForCausalLM {
    pub fn new(_model_id: &str, _cache_dir: &str, _config: Qwen3Config) -> Result<Self> {
        Ok(Self {})
    }
}
