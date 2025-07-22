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

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use crate::{error::Error, error::Result};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TokenizerClass {
    Qwen2Tokenizer,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenX {
    pub content: String,
    pub lstrip: bool,
    pub normalized: bool,
    pub rstrip: bool,
    pub single_word: bool,
    pub special: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenizerConfig {
    pub add_bos_token: bool,
    pub add_prefix_space: bool,
    pub added_tokens_decoder: HashMap<String, TokenX>,
    pub additional_special_tokens: Vec<String>,
    pub bos_token: Option<String>,
    pub chat_template: String,
    pub clean_up_tokenization_spaces: bool,
    pub eos_token: String,
    pub errors: String,
    pub model_max_length: usize,
    pub pad_token: String,
    pub split_special_tokens: bool,
    pub tokenizer_class: TokenizerClass,
    pub unk_token: Option<String>,
}

impl TokenizerConfig {
    pub fn from_pretrained(model_id: &str, cache_dir: &str) -> Result<Self> {
        let config_path = format!("{cache_dir}/{model_id}/tokenizer_config.json");
        let config_str = std::fs::read_to_string(config_path).map_err(Error::IoError)?;
        let config: TokenizerConfig =
            serde_json::from_str(&config_str).map_err(Error::SerdeError)?;

        Ok(config)
    }
}
