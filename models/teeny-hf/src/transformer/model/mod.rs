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
use std::sync::Arc;

use ndarray::Array1;
use teeny_core::dtype::Dtype;
use teeny_core::graph::tensor_usize;
use teeny_core::nn::Module;
use teeny_core::safetensors::SafeTensors;
use teeny_core::value::Value;
use teeny_data::safetensors::{FileSafeTensors, SafeTensorsMmaps};
use teeny_nlp::tokenizer::Message;

use crate::error::{Error, Result};
use crate::transformer::config::model_config::Architecture;
use crate::transformer::model::qwen::qwen3::qwen3_causal_llm::Qwen3ForCausalLM;
use crate::transformer::model::qwen::qwen3::qwen3_config::Qwen3Config;
use crate::transformer::model::qwen::qwen3::qwen3_model::QwenModelInputsBuilder;
use crate::transformer::tokenizer::tokenizer_config::TokenizerConfig;
use crate::transformer::util::template;
use crate::transformer::{self, tokenizer};

pub mod qwen;

pub fn run_qwen3<N: Dtype>(model_id: &str, cache_dir: &Path) -> Result<()> {
    let model_dir = cache_dir.join(model_id);
    let mmaps = SafeTensorsMmaps::from_pretrained(&model_dir)?;
    let safetensors = FileSafeTensors::from_pretrained(&mmaps)?;

    let tokenizer_config = TokenizerConfig::from_pretrained(model_id, cache_dir)?;
    let tokenizer = tokenizer::from_pretrained(model_id, cache_dir)?;
    let mut model = transformer::model::from_pretrained(model_id, cache_dir, &safetensors)?;

    let prompt = "Give me a short introduction to large language model.";
    let messages = [Message::new("user", prompt)];

    let text = template::apply_chat_template(&tokenizer_config.chat_template, &messages, &[]);
    let encoded_inputs = tokenizer
        .encode(text, false)
        .map_err(Error::TokenizerError)?;
    let encoded_ids = tensor_usize(
        Array1::from(
            encoded_inputs
                .get_ids()
                .iter()
                .map(|x| *x as usize)
                .collect::<Vec<_>>(),
        )
        .into_dyn(),
    );

    let model_inputs = QwenModelInputsBuilder::default()
        .input_ids(Some(encoded_ids))
        .build()
        .map_err(|e| Error::BuilderError(Arc::new(e)))?;

    let generated_ids = model
        .forward(model_inputs)?
        .realize()?
        .iter()
        .map(|x| match x {
            Value::F32(x) => *x as u32,
            _ => unreachable!(),
        })
        .collect::<Vec<_>>();

    let thinking_content = tokenizer
        .decode(&generated_ids[..], false)
        .map_err(Error::TokenizerError)?;

    let content = tokenizer
        .decode(&generated_ids[thinking_content.len()..], false)
        .map_err(Error::TokenizerError)?;

    println!("thinking content: {thinking_content}");
    println!("content: {content}");

    Ok(())
}

fn from_pretrained<'data, T: SafeTensors<'data>>(
    model_id: &str,
    cache_dir: &Path,
    safetensors: &'data T,
) -> Result<Qwen3ForCausalLM<'data>> {
    let config = Qwen3Config::from_pretrained(model_id, cache_dir)?;

    match config.architectures[0] {
        Architecture::Qwen3ForCausalLM => {
            let model = Qwen3ForCausalLM::from_pretrained(&config, cache_dir, safetensors)?;
            Ok(model)
        }
    }
}
