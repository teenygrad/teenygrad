/*
 * Copyright (c) 2026 teenygrad (https://teenygrad.org).
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

//! Turning text into token ids and back: the [`Tokenizer`] every model exposes.
//!
//! Frontends and validation use it to count tokens (usage, context limits) without
//! generating, and models use it to encode prompts and stream replies. The trait is
//! object-safe and carries no dependency on a tokenizer library, so it is available in
//! every build. [`ByteTokenizer`] is the reference implementation (UTF-8 bytes as ids);
//! `HfTokenizer`, behind the `hf-tokenizer` feature, wraps a Hugging Face
//! `tokenizer.json` through the `tokenizers` crate.

mod byte;
#[cfg(feature = "hf-tokenizer")]
mod hf;

pub use byte::ByteTokenizer;
#[cfg(feature = "hf-tokenizer")]
pub use hf::HfTokenizer;

use crate::Result;

/// Whether encoding adds the special tokens a model expects around a sequence (a BOS
/// token, say). Special tokens written in the text itself are recognised either way by
/// tokenizers that know them.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AddSpecial {
    /// Add them, as for raw text sent to the model.
    Add,
    /// Leave them out, as for text a chat template has already framed.
    Omit,
}

/// Whether decoding writes special tokens out or drops them.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SpecialTokens {
    /// Write them out, as for model output that carries markers (tool calls).
    Keep,
    /// Drop them, as for text shown to a reader.
    Skip,
}

/// A special (control) token: its id and how it is written.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SpecialToken {
    pub id: u32,
    pub content: String,
}

/// A model's vocabulary: text to token ids and back.
pub trait Tokenizer: Send + Sync {
    /// The token ids of `text`.
    fn encode(&self, text: &str, add_special: AddSpecial) -> Result<Vec<u32>>;

    /// The text of `ids`. Bytes that do not form valid UTF-8 become U+FFFD.
    fn decode(&self, ids: &[u32], special: SpecialTokens) -> Result<String>;

    /// A decoder for ids arriving one at a time, as a model generates them.
    fn stream_decoder(&self, special: SpecialTokens) -> Box<dyn StreamDecoder + Send + '_>;

    /// How many tokens `text` is, without any added special tokens: what it adds to a
    /// prompt.
    fn count(&self, text: &str) -> Result<usize> {
        Ok(self.encode(text, AddSpecial::Omit)?.len())
    }

    /// The number of token ids, special tokens included.
    fn vocab_size(&self) -> usize;

    /// The id of a token, given as the vocabulary writes it.
    fn token_to_id(&self, token: &str) -> Option<u32>;

    /// A token as the vocabulary writes it (not necessarily its decoded text: byte-level
    /// vocabularies spell bytes with stand-in characters).
    fn id_to_token(&self, id: u32) -> Option<String>;

    /// The special tokens, in id order.
    fn special_tokens(&self) -> Vec<SpecialToken>;
}

/// Decodes generated ids incrementally.
///
/// [`StreamDecoder::step`] returns the text that `id` completes, or `None` while it is
/// still part of something unfinished: a token can end in the middle of a UTF-8
/// character, and some vocabularies decide on a leading space only once the next token
/// is known. Concatenating every piece gives the text [`Tokenizer::decode`] gives for the
/// same ids.
pub trait StreamDecoder {
    fn step(&mut self, id: u32) -> Result<Option<String>>;
}
