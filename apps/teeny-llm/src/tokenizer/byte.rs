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

//! [`ByteTokenizer`], the reference [`Tokenizer`].

use super::{AddSpecial, SpecialToken, SpecialTokens, StreamDecoder, Tokenizer};
use crate::{Error, Result};

/// A tokenizer whose tokens are UTF-8 bytes (ids 0 to 255), plus a BOS (256) and an
/// EOS (257) token.
///
/// It needs no files, round-trips any text, and splits every non-ASCII character across
/// tokens, which makes it the reference implementation of [`Tokenizer`] and a test of
/// anything that streams. Special tokens are not recognised in text: `"<s>"` encodes as
/// its three bytes.
#[derive(Debug, Clone, Copy, Default)]
pub struct ByteTokenizer;

impl ByteTokenizer {
    pub const BOS: u32 = 256;
    pub const EOS: u32 = 257;
    const BOS_TEXT: &'static str = "<s>";
    const EOS_TEXT: &'static str = "</s>";

    fn special_text(id: u32) -> Option<&'static str> {
        match id {
            Self::BOS => Some(Self::BOS_TEXT),
            Self::EOS => Some(Self::EOS_TEXT),
            _ => None,
        }
    }

    fn unknown(id: u32) -> Error {
        Error::Tokenizer(format!(
            "token id {id} is outside the byte tokenizer's vocabulary of 258 (bytes 0-255, \
             BOS 256, EOS 257)"
        ))
    }
}

impl Tokenizer for ByteTokenizer {
    fn encode(&self, text: &str, add_special: AddSpecial) -> Result<Vec<u32>> {
        let mut ids = Vec::with_capacity(text.len() + 1);
        if add_special == AddSpecial::Add {
            ids.push(Self::BOS);
        }
        ids.extend(text.bytes().map(u32::from));
        Ok(ids)
    }

    fn decode(&self, ids: &[u32], special: SpecialTokens) -> Result<String> {
        let mut out = String::new();
        let mut d = ByteStream {
            special,
            pending: Vec::new(),
        };
        for &id in ids {
            if let Some(piece) = d.step(id)? {
                out.push_str(&piece);
            }
        }
        // Bytes still waiting for the rest of a character never get it.
        out.push_str(&d.flush());
        Ok(out)
    }

    fn stream_decoder(&self, special: SpecialTokens) -> Box<dyn StreamDecoder + Send + '_> {
        Box::new(ByteStream {
            special,
            pending: Vec::new(),
        })
    }

    fn vocab_size(&self) -> usize {
        258
    }

    fn token_to_id(&self, token: &str) -> Option<u32> {
        match token {
            Self::BOS_TEXT => Some(Self::BOS),
            Self::EOS_TEXT => Some(Self::EOS),
            t if t.len() == 1 => Some(u32::from(t.as_bytes()[0])),
            _ => None,
        }
    }

    fn id_to_token(&self, id: u32) -> Option<String> {
        match (Self::special_text(id), u8::try_from(id)) {
            (Some(s), _) => Some(s.to_owned()),
            (None, Ok(b)) => Some(String::from_utf8_lossy(&[b]).into_owned()),
            (None, Err(_)) => None,
        }
    }

    fn special_tokens(&self) -> Vec<SpecialToken> {
        vec![
            SpecialToken {
                id: Self::BOS,
                content: Self::BOS_TEXT.to_owned(),
            },
            SpecialToken {
                id: Self::EOS,
                content: Self::EOS_TEXT.to_owned(),
            },
        ]
    }
}

/// Bytes are held until they form whole characters.
struct ByteStream {
    special: SpecialTokens,
    pending: Vec<u8>,
}

impl ByteStream {
    /// The text the pending bytes make so far, keeping back a character still being
    /// written. Bytes that can never start or continue a character become U+FFFD.
    fn take_complete(&mut self) -> String {
        let mut out = String::new();
        loop {
            match std::str::from_utf8(&self.pending) {
                Ok(s) => {
                    out.push_str(s);
                    self.pending.clear();
                    return out;
                }
                Err(e) => {
                    let good = e.valid_up_to();
                    // The first `valid_up_to` bytes are valid UTF-8, so this is lossless.
                    out.push_str(&String::from_utf8_lossy(&self.pending[..good]));
                    match e.error_len() {
                        // The rest is the start of a character: wait for its other bytes.
                        None => {
                            self.pending.drain(..good);
                            return out;
                        }
                        Some(bad) => {
                            out.push(char::REPLACEMENT_CHARACTER);
                            self.pending.drain(..good + bad);
                        }
                    }
                }
            }
        }
    }

    /// Everything pending, an unfinished character included (as U+FFFD).
    fn flush(&mut self) -> String {
        let out = String::from_utf8_lossy(&self.pending).into_owned();
        self.pending.clear();
        out
    }
}

impl StreamDecoder for ByteStream {
    fn step(&mut self, id: u32) -> Result<Option<String>> {
        let piece = if let Ok(b) = u8::try_from(id) {
            self.pending.push(b);
            self.take_complete()
        } else if let Some(text) = ByteTokenizer::special_text(id) {
            // A special token ends whatever character was being written.
            let mut out = self.flush();
            if self.special == SpecialTokens::Keep {
                out.push_str(text);
            }
            out
        } else {
            return Err(ByteTokenizer::unknown(id));
        };
        Ok((!piece.is_empty()).then_some(piece))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn stream(ids: &[u32], special: SpecialTokens) -> Vec<Option<String>> {
        let t = ByteTokenizer;
        let mut d = t.stream_decoder(special);
        ids.iter().map(|&id| d.step(id).unwrap()).collect()
    }

    #[test]
    fn test_byte_tokenizer_round_trips_text() {
        let t = ByteTokenizer;
        for text in ["", "hello", "naïve café 日本語 🦀"] {
            let ids = t.encode(text, AddSpecial::Omit).unwrap();
            assert_eq!(ids.len(), text.len());
            assert_eq!(t.decode(&ids, SpecialTokens::Keep).unwrap(), text);
            assert_eq!(t.count(text).unwrap(), text.len());
        }
        let ids = t.encode("hi", AddSpecial::Add).unwrap();
        assert_eq!(ids, [ByteTokenizer::BOS, 104, 105]);
        assert_eq!(t.decode(&ids, SpecialTokens::Keep).unwrap(), "<s>hi");
        assert_eq!(t.decode(&ids, SpecialTokens::Skip).unwrap(), "hi");
    }

    #[test]
    fn test_stream_decoder_holds_a_split_character_until_it_is_whole() {
        // "é" is 0xC3 0xA9; "🦀" is four bytes.
        let ids: Vec<u32> = "aé🦀".bytes().map(u32::from).collect();
        let pieces = stream(&ids, SpecialTokens::Keep);
        let expected: Vec<Option<String>> = vec![
            Some("a".into()),
            None,
            Some("é".into()),
            None,
            None,
            None,
            Some("🦀".into()),
        ];
        assert_eq!(pieces, expected);
    }

    #[test]
    fn test_stream_decoder_replaces_bytes_that_cannot_be_text() {
        // A lone continuation byte, and a character cut short by EOS.
        let pieces = stream(&[0x80, 0x41, 0xC3, ByteTokenizer::EOS], SpecialTokens::Keep);
        assert_eq!(
            pieces,
            vec![
                Some("\u{FFFD}".into()),
                Some("A".into()),
                None,
                Some("\u{FFFD}</s>".into())
            ]
        );
        let t = ByteTokenizer;
        assert_eq!(
            t.decode(&[0x41, 0xC3], SpecialTokens::Skip).unwrap(),
            "A\u{FFFD}"
        );
    }

    #[test]
    fn test_stream_pieces_concatenate_to_the_decoded_text() {
        let t = ByteTokenizer;
        let text = "tool → call: {\"city\": \"Zürich\"}";
        let ids = t.encode(text, AddSpecial::Add).unwrap();
        for special in [SpecialTokens::Keep, SpecialTokens::Skip] {
            let joined: String = stream(&ids, special).into_iter().flatten().collect();
            assert_eq!(joined, t.decode(&ids, special).unwrap());
        }
    }

    #[test]
    fn test_byte_tokenizer_vocabulary() {
        let t = ByteTokenizer;
        assert_eq!(t.vocab_size(), 258);
        assert_eq!(t.token_to_id("A"), Some(65));
        assert_eq!(t.token_to_id("</s>"), Some(ByteTokenizer::EOS));
        assert_eq!(t.id_to_token(ByteTokenizer::BOS).as_deref(), Some("<s>"));
        assert_eq!(t.id_to_token(300), None);
        assert_eq!(t.special_tokens().len(), 2);
        let mut d = t.stream_decoder(SpecialTokens::Keep);
        assert!(matches!(d.step(300), Err(Error::Tokenizer(_))));
    }
}
