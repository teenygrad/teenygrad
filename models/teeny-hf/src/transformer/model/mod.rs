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

use teeny_core::nn::Module;

use crate::error::{Error, Result};
use crate::transformer::config::model_config::Architecture;
use crate::transformer::model::qwen::qwen3::qwen3_causal_llm::Qwen3ForCausalLM;
use crate::transformer::model::qwen::qwen3::qwen3_config::Qwen3Config;

pub mod qwen;

pub fn from_pretrained(
    model_id: &str,
    cache_dir: &str,
) -> Result<Box<dyn Module<f32, Err = Error>>> {
    let config = Qwen3Config::from_pretrained(model_id, cache_dir)?;

    match config.architectures[0] {
        Architecture::Qwen3ForCausalLM => {
            let model = Qwen3ForCausalLM::new(&config)?;
            Ok(Box::new(model))
        }
    }
}
