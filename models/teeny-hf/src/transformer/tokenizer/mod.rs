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

use tokenizers::models::bpe::BPE;
use tokenizers::tokenizer::Tokenizer;

pub mod qwen2_tokenizer;
pub mod tokenizer_config;

use crate::{
    error::Error, error::Result, transformer::tokenizer::tokenizer_config::TokenizerConfig,
};

pub fn from_pretrained(model_id: &str, cache_dir: &str) -> Result<Tokenizer> {
    let _tokenizer_config = TokenizerConfig::from_pretrained(model_id, cache_dir)?;
    let vocab_file = format!("{cache_dir}/{model_id}/vocab.json");
    let merges_file = format!("{cache_dir}/{model_id}/merges.txt");
    let bpe_builder = BPE::from_file(&vocab_file, &merges_file);

    let bpe = bpe_builder.build().map_err(Error::TokenizerError)?;

    Ok(Tokenizer::new(bpe))
}
