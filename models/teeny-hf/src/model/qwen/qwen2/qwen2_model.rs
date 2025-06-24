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

use teeny_core::TeenyModel;

use crate::model::qwen::qwen3::qwen3_config::Qwen3Config;

use crate::error::{Result, TeenyHFError};

pub struct Qwen2Model {}

impl Qwen2Model {
    pub fn new(_config: &Qwen3Config) -> Result<Self> {
        Ok(Self {})
    }
}

impl TeenyModel for Qwen2Model {
    type Err = TeenyHFError;

    fn forward(&self, _model_inputs: &[u32]) -> Result<Vec<u32>> {
        todo!()
    }
}
