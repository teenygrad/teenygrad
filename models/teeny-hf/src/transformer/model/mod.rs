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

use teeny_core::graph::NodeRef;
use teeny_core::nn::Module;
use teeny_data::safetensors::{FileSafeTensors, SafeTensorsMmaps};

use crate::error::{Error, Result};
use crate::transformer::config::model_config::Architecture;
use crate::transformer::model::qwen::qwen2::qwen2_model::QwenModelInputs;
use crate::transformer::model::qwen::qwen3::qwen3_causal_llm::Qwen3ForCausalLM;
use crate::transformer::model::qwen::qwen3::qwen3_config::Qwen3Config;

pub mod qwen;

pub fn from_pretrained(
    model_id: &str,
    cache_dir: &Path,
) -> Result<Box<dyn Module<f32, QwenModelInputs, NodeRef<f32>, Err = Error>>> {
    let config = Qwen3Config::from_pretrained(model_id, cache_dir)?;

    match config.architectures[0] {
        Architecture::Qwen3ForCausalLM => {
            let model_dir = cache_dir.join(model_id);

            let mmaps = SafeTensorsMmaps::from_pretrained(&model_dir)?;
            let safetensors = FileSafeTensors::from_pretrained(&mmaps)?;

            let _model = Qwen3ForCausalLM::from_pretrained(&config, cache_dir, &safetensors)?;
            todo!()
            // Ok(Box::new(model))
        }
    }
}
