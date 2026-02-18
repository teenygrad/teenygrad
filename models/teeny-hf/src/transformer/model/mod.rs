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
use crate::transformer::tokenizer::tokenizer_config::TokenizerConfig;
use crate::transformer::util::template;
use crate::transformer::{self, tokenizer};

pub mod qwen;

pub fn run_qwen3<N: Dtype>(model_id: &str, cache_dir: &Path) -> Result<()> {
    // let model_dir = cache_dir.join(model_id);
    // let mmaps = SafeTensorsMmaps::from_pretrained(&model_dir)?;
    // let safetensors = FileSafeTensors::from_pretrained(&mmaps)?;

    // let tokenizer_config = TokenizerConfig::from_pretrained(model_id, cache_dir)?;
    // let tokenizer = tokenizer::from_pretrained(model_id, cache_dir)?;
    // let mut model = transformer::model::from_pretrained(model_id, cache_dir, &safetensors)?;

    // let prompt = "Give me a short introduction to large language model.";
    // let messages = [Message::new("user", prompt)];

    // let text = template::apply_chat_template(&tokenizer_config.chat_template, &messages, &[]);
    // let encoded_inputs = tokenizer
    //     .encode(text, false)
    //     .map_err(Error::TokenizerError)?;
    // let encoded_ids = tensor_usize(
    //     Array1::from(
    //         encoded_inputs
    //             .get_ids()
    //             .iter()
    //             .map(|x| *x as usize)
    //             .collect::<Vec<_>>(),
    //     )
    //     .into_dyn(),
    // );

    // let model_inputs = QwenModelInputsBuilder::default()
    //     .input_ids(Some(encoded_ids))
    //     .build()
    //     .map_err(|e| Error::BuilderError(Arc::new(e)))?;

    // let generated_ids = model
    //     .forward(model_inputs)?
    //     .realize()?
    //     .iter()
    //     .map(|x| match x {
    //         Value::F32(x) => *x as u32,
    //         _ => unreachable!(),
    //     })
    //     .collect::<Vec<_>>();

    // let thinking_content = tokenizer
    //     .decode(&generated_ids[..], false)
    //     .map_err(Error::TokenizerError)?;

    // let content = tokenizer
    //     .decode(&generated_ids[thinking_content.len()..], false)
    //     .map_err(Error::TokenizerError)?;

    // println!("thinking content: {thinking_content}");
    // println!("content: {content}");

    Ok(())
}

// fn from_pretrained<'data, T: SafeTensors<'data>>(
//     model_id: &str,
//     cache_dir: &Path,
//     safetensors: &'data T,
// ) -> Result<Qwen3ForCausalLM<'data>> {
//     let config = Qwen3Config::from_pretrained(model_id, cache_dir)?;

//     match config.architectures[0] {
//         Architecture::Qwen3ForCausalLM => {
//             let model = Qwen3ForCausalLM::from_pretrained(&config, cache_dir, safetensors)?;
//             Ok(model)
//         }
//     }
// }
