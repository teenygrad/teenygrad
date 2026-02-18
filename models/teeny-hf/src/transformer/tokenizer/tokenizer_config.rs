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

use std::{collections::HashMap, path::Path};

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
    pub fn from_pretrained(model_id: &str, cache_dir: &Path) -> Result<Self> {
        let config_path = format!(
            "{}/{model_id}/tokenizer_config.json",
            cache_dir.to_string_lossy()
        );
        let config_str = std::fs::read_to_string(config_path).map_err(Error::IoError)?;
        let config: TokenizerConfig =
            serde_json::from_str(&config_str).map_err(Error::SerdeError)?;

        Ok(config)
    }
}
