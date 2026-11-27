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

use base64::Engine as _;
use base64::prelude::BASE64_STANDARD;
use serde::{Deserialize, Serialize};
use teeny_llm::embedding::{EmbeddingInput, EmbeddingRequest};

use super::error::ApiError;

/// `input`: a string, strings, token ids, or arrays of token ids.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum EmbeddingInputs {
    One(String),
    Many(Vec<String>),
    Tokens(Vec<u32>),
    TokenBatches(Vec<Vec<u32>>),
}

/// `CreateEmbeddingRequest`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CreateEmbeddingRequest {
    pub model: String,
    pub input: EmbeddingInputs,
    /// `"float"` (default) or `"base64"` (little-endian f32s).
    #[serde(default)]
    pub encoding_format: Option<String>,
    #[serde(default)]
    pub dimensions: Option<u32>,
    #[serde(default)]
    pub user: Option<String>,
}

/// How vectors are written in the response.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EncodingFormat {
    Float,
    Base64,
}

impl CreateEmbeddingRequest {
    /// The core request, and how to write the vectors.
    pub fn into_core(self) -> Result<(EmbeddingRequest, EncodingFormat), ApiError> {
        let format = match self.encoding_format.as_deref() {
            None | Some("float") => EncodingFormat::Float,
            Some("base64") => EncodingFormat::Base64,
            Some(f) => {
                return Err(ApiError::invalid_request(
                    format!("encoding_format `{f}` is not float or base64"),
                    Some("encoding_format"),
                ));
            }
        };
        let inputs = match self.input {
            EmbeddingInputs::One(s) => vec![EmbeddingInput::Text(s)],
            EmbeddingInputs::Many(v) => v.into_iter().map(EmbeddingInput::Text).collect(),
            EmbeddingInputs::Tokens(t) => vec![EmbeddingInput::Tokens(t)],
            EmbeddingInputs::TokenBatches(b) => b.into_iter().map(EmbeddingInput::Tokens).collect(),
        };
        Ok((
            EmbeddingRequest {
                model: self.model,
                inputs,
                dimensions: self.dimensions,
                user: self.user,
            },
            format,
        ))
    }
}

/// A vector, as floats or as base64 of its little-endian f32s.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum EmbeddingVector {
    Float(Vec<f32>),
    Base64(String),
}

impl EmbeddingVector {
    pub fn encode(vector: Vec<f32>, format: EncodingFormat) -> Self {
        match format {
            EncodingFormat::Float => Self::Float(vector),
            EncodingFormat::Base64 => {
                let bytes: Vec<u8> = vector.iter().flat_map(|x| x.to_le_bytes()).collect();
                Self::Base64(BASE64_STANDARD.encode(&bytes))
            }
        }
    }
}

/// `Embedding`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Embedding {
    /// Always `"embedding"`.
    pub object: String,
    pub index: u32,
    pub embedding: EmbeddingVector,
}

/// The `usage` object of `CreateEmbeddingResponse`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct EmbeddingUsage {
    pub prompt_tokens: u32,
    pub total_tokens: u32,
}

/// `CreateEmbeddingResponse`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CreateEmbeddingResponse {
    /// Always `"list"`.
    pub object: String,
    pub data: Vec<Embedding>,
    pub model: String,
    pub usage: EmbeddingUsage,
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_embedding_inputs_take_every_wire_shape() {
        let count = |input| {
            serde_json::from_value::<CreateEmbeddingRequest>(json!({"model": "m", "input": input}))
                .unwrap()
                .into_core()
                .unwrap()
                .0
                .inputs
        };
        assert_eq!(count(json!("a")), vec![EmbeddingInput::Text("a".into())]);
        assert_eq!(count(json!(["a", "b"])).len(), 2);
        assert_eq!(
            count(json!([1, 2])),
            vec![EmbeddingInput::Tokens(vec![1, 2])]
        );
        assert_eq!(count(json!([[1], [2, 3]])).len(), 2);
    }

    #[test]
    fn test_base64_vectors_are_little_endian_f32s() {
        let v = EmbeddingVector::encode(vec![1.0], EncodingFormat::Base64);
        assert_eq!(v, EmbeddingVector::Base64("AACAPw==".into()));
    }
}
