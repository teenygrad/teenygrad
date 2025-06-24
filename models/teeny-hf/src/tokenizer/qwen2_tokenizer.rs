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

use teeny_nlp::tokenizer::{Message, Tokenizer};

use crate::tokenizer::tokenizer_config::TokenizerConfig;

use crate::error::Result;
use crate::util::template;

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
