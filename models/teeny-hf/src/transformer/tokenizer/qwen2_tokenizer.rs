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

use teeny_nlp::tokenizer::{Message, Tokenizer};

use crate::transformer::tokenizer::tokenizer_config::TokenizerConfig;

use crate::error::Result;
use crate::transformer::util::template;

pub struct Qwen2Tokenizer {
    pub config: TokenizerConfig,
}

impl Tokenizer for Qwen2Tokenizer {
    fn apply_chat_template(
        &self,
        messages: &[Message],
        chat_template: &str,
        _tokenize: bool,
        _add_generation_prompt: bool,
        _enable_thinking: bool,
    ) -> String {
        template::apply_chat_template(chat_template, messages, &[])
    }

    fn encode(&self, _texts: &[String]) -> Vec<usize> {
        todo!()
    }

    fn decode(&self, _ids: &[usize]) -> String {
        todo!()
    }
}

impl Qwen2Tokenizer {
    pub fn new(_model_id: &str, _cache_dir: &str, config: TokenizerConfig) -> Result<Self> {
        Ok(Self { config })
    }
}
