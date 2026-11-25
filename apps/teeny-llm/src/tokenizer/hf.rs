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

//! [`HfTokenizer`]: a Hugging Face `tokenizer.json`, through the `tokenizers` crate.

use std::path::Path;

use tokenizers::{
    DecodeStream, DecoderWrapper, ModelWrapper, NormalizerWrapper, PostProcessorWrapper,
    PreTokenizerWrapper,
};

use super::{AddSpecial, SpecialToken, SpecialTokens, StreamDecoder, Tokenizer};
use crate::{Error, Result};

/// A Hugging Face tokenizer, loaded from a model folder's `tokenizer.json`.
///
/// Everything is delegated to the `tokenizers` crate (built without its default
/// features: no C regex library or HTTP client, so it cross-compiles for RISC-V), and
/// streaming to its `DecodeStream`, which handles tokens that end mid-character and the
/// leading-space rules of byte-level and SentencePiece vocabularies.
pub struct HfTokenizer {
    inner: tokenizers::Tokenizer,
}

impl std::fmt::Debug for HfTokenizer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HfTokenizer")
            .field("vocab_size", &self.vocab_size())
            .finish_non_exhaustive()
    }
}

fn tokenizer_error(what: &str, e: impl std::fmt::Display) -> Error {
    Error::Tokenizer(format!("{what}: {e}"))
}

impl HfTokenizer {
    /// Loads `tokenizer.json` from `path`.
    pub fn from_file(path: &Path) -> Result<Self> {
        let inner = tokenizers::Tokenizer::from_file(path)
            .map_err(|e| tokenizer_error(&format!("loading {}", path.display()), e))?;
        Ok(Self { inner })
    }

    /// Parses a `tokenizer.json` document.
    pub fn from_json(json: &str) -> Result<Self> {
        let inner = json
            .parse::<tokenizers::Tokenizer>()
            .map_err(|e| tokenizer_error("parsing tokenizer.json", e))?;
        Ok(Self { inner })
    }

    /// The underlying `tokenizers` tokenizer, for what the trait does not cover
    /// (offsets, truncation, padding).
    pub fn inner(&self) -> &tokenizers::Tokenizer {
        &self.inner
    }
}

impl Tokenizer for HfTokenizer {
    fn encode(&self, text: &str, add_special: AddSpecial) -> Result<Vec<u32>> {
        let enc = self
            .inner
            .encode(text, add_special == AddSpecial::Add)
            .map_err(|e| tokenizer_error(&format!("encoding {} bytes of text", text.len()), e))?;
        Ok(enc.get_ids().to_vec())
    }

    fn decode(&self, ids: &[u32], special: SpecialTokens) -> Result<String> {
        self.inner
            .decode(ids, special == SpecialTokens::Skip)
            .map_err(|e| tokenizer_error(&format!("decoding {} token ids", ids.len()), e))
    }

    fn stream_decoder(&self, special: SpecialTokens) -> Box<dyn StreamDecoder + Send + '_> {
        Box::new(HfStream(
            self.inner.decode_stream(special == SpecialTokens::Skip),
        ))
    }

    fn vocab_size(&self) -> usize {
        self.inner.get_vocab_size(true)
    }

    fn token_to_id(&self, token: &str) -> Option<u32> {
        self.inner.token_to_id(token)
    }

    fn id_to_token(&self, id: u32) -> Option<String> {
        self.inner.id_to_token(id)
    }

    fn special_tokens(&self) -> Vec<SpecialToken> {
        let mut tokens: Vec<SpecialToken> = self
            .inner
            .get_added_tokens_decoder()
            .into_iter()
            .filter(|(_, t)| t.special)
            .map(|(id, t)| SpecialToken {
                id,
                content: t.content,
            })
            .collect();
        tokens.sort_by_key(|t| t.id);
        tokens
    }
}

struct HfStream<'t>(
    DecodeStream<
        't,
        ModelWrapper,
        NormalizerWrapper,
        PreTokenizerWrapper,
        PostProcessorWrapper,
        DecoderWrapper,
    >,
);

impl StreamDecoder for HfStream<'_> {
    fn step(&mut self, id: u32) -> Result<Option<String>> {
        self.0
            .step(id)
            .map_err(|e| tokenizer_error(&format!("decoding token id {id}"), e))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use tokenizers::pre_tokenizers::byte_level::ByteLevel;

    /// A byte-level BPE vocabulary (as GPT-2 and Qwen use) with one token per byte and a
    /// few merges, so non-ASCII characters split across tokens, plus a special `<eos>`.
    fn byte_level() -> HfTokenizer {
        let mut alphabet: Vec<char> = ByteLevel::alphabet().into_iter().collect();
        alphabet.sort();
        let mut vocab = serde_json::Map::new();
        for (i, c) in alphabet.iter().enumerate() {
            vocab.insert(c.to_string(), json!(i));
        }
        let n = alphabet.len();
        vocab.insert("he".into(), json!(n));
        vocab.insert("hel".into(), json!(n + 1));
        let tokenizer = json!({
            "version": "1.0",
            "truncation": null,
            "padding": null,
            "added_tokens": [{
                "id": n + 2, "content": "<eos>", "single_word": false, "lstrip": false,
                "rstrip": false, "normalized": false, "special": true
            }],
            "normalizer": null,
            "pre_tokenizer": {
                "type": "ByteLevel", "add_prefix_space": false, "trim_offsets": true,
                "use_regex": true
            },
            "post_processor": null,
            "decoder": {
                "type": "ByteLevel", "add_prefix_space": true, "trim_offsets": true,
                "use_regex": true
            },
            "model": {
                "type": "BPE", "dropout": null, "unk_token": null,
                "continuing_subword_prefix": null, "end_of_word_suffix": null,
                "fuse_unk": false, "byte_fallback": false, "ignore_merges": false,
                "vocab": vocab, "merges": [["h", "e"], ["he", "l"]]
            }
        });
        HfTokenizer::from_json(&tokenizer.to_string()).unwrap()
    }

    #[test]
    fn test_hf_tokenizer_round_trips_and_merges() {
        let t = byte_level();
        let text = "hello wörld";
        let ids = t.encode(text, AddSpecial::Omit).unwrap();
        // "hel" is one token; "ö" is two bytes, so two tokens.
        assert_eq!(ids[0], t.token_to_id("hel").unwrap());
        assert_eq!(ids.len(), text.len() - 2);
        assert_eq!(t.decode(&ids, SpecialTokens::Keep).unwrap(), text);
        assert_eq!(t.count(text).unwrap(), ids.len());
    }

    #[test]
    fn test_hf_stream_decoder_waits_for_a_split_character() {
        let t = byte_level();
        let ids = t.encode("aé", AddSpecial::Omit).unwrap();
        assert_eq!(ids.len(), 3);
        let mut d = t.stream_decoder(SpecialTokens::Keep);
        let pieces: Vec<Option<String>> = ids.iter().map(|&id| d.step(id).unwrap()).collect();
        assert_eq!(pieces, [Some("a".into()), None, Some("é".into())]);
    }

    #[test]
    fn test_hf_stream_pieces_concatenate_to_the_decoded_text() {
        let t = byte_level();
        let eos = t.token_to_id("<eos>").unwrap();
        let mut ids = t.encode("hello, Zürich 🦀", AddSpecial::Omit).unwrap();
        ids.push(eos);
        for special in [SpecialTokens::Keep, SpecialTokens::Skip] {
            let mut d = t.stream_decoder(special);
            let joined: String = ids.iter().filter_map(|&id| d.step(id).unwrap()).collect();
            assert_eq!(joined, t.decode(&ids, special).unwrap(), "{special:?}");
        }
        assert!(
            t.decode(&ids, SpecialTokens::Keep)
                .unwrap()
                .ends_with("<eos>")
        );
        assert!(
            !t.decode(&ids, SpecialTokens::Skip)
                .unwrap()
                .contains("<eos>")
        );
    }

    #[test]
    fn test_hf_tokenizer_lists_its_special_tokens() {
        let t = byte_level();
        let special = t.special_tokens();
        assert_eq!(special.len(), 1);
        assert_eq!(special[0].content, "<eos>");
        assert_eq!(t.vocab_size(), 256 + 3);
    }

    #[test]
    fn test_hf_tokenizer_reports_a_bad_file_by_path() {
        let err = HfTokenizer::from_file(Path::new("/nonexistent/tokenizer.json"))
            .unwrap_err()
            .to_string();
        assert!(err.contains("/nonexistent/tokenizer.json"), "{err}");
    }
}
