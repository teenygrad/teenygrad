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

//! Embeddings: the interface embedding models implement to be served.
//!
//! Embeddings usually come from different models than chat (BERT-style encoders), so they
//! have their own trait. A model implements [`EmbeddingModel`]: it declares its
//! [`EmbeddingCapabilities`] and answers an [`EmbeddingRequest`] with [`Embeddings`], one
//! vector per input. [`validate`] checks each request against the capabilities first, so
//! what a model cannot do (token-id input, reduced dimensions, a batch too large) is
//! refused with an error naming the parameter. [`EchoEmbeddingModel`] is the reference
//! implementation.

mod echo;

pub use echo::EchoEmbeddingModel;

use futures::future::BoxFuture;

use crate::{Error, ModelInfo, Result};

/// The OpenAI limit on inputs in one request.
pub const MAX_INPUTS: usize = 2048;

/// One thing to embed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EmbeddingInput {
    Text(String),
    /// Already tokenized, as the model's tokenizer would.
    Tokens(Vec<u32>),
}

/// A request for embeddings. `Default` gives an empty one, to fill in with struct update
/// syntax.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct EmbeddingRequest {
    pub model: String,
    /// What to embed, in order.
    pub inputs: Vec<EmbeddingInput>,
    /// Truncate each vector to this many dimensions (and renormalise), for models trained
    /// for it. `None` is the model's full size.
    pub dimensions: Option<u32>,
    /// An opaque id for the end user, for abuse tracking.
    pub user: Option<String>,
}

/// The result of embedding a batch of inputs, one vector per input in order.
#[derive(Debug, Clone, PartialEq)]
pub struct Embeddings {
    pub vectors: Vec<Vec<f32>>,
    pub prompt_tokens: u32,
}

/// How a model turns per-token states into one vector.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum Pooling {
    /// The mean over the tokens.
    #[default]
    Mean,
    /// The first (`[CLS]`) token's state.
    Cls,
    /// The last token's state, as decoder models embed.
    LastToken,
}

/// What an embedding model can do.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EmbeddingCapabilities {
    /// Most tokens in one input; longer inputs are refused, not truncated.
    pub max_input_tokens: u32,
    /// Most inputs in one request.
    pub max_batch: u32,
    /// The vectors' size.
    pub dimensions: u32,
    /// Whether vectors can be truncated to fewer dimensions (Matryoshka training).
    pub reducible_dimensions: bool,
    /// Whether inputs can be given as token ids.
    pub token_input: bool,
    pub pooling: Pooling,
    /// Whether vectors have unit length.
    pub normalized: bool,
}

impl Default for EmbeddingCapabilities {
    fn default() -> Self {
        Self {
            max_input_tokens: 512,
            max_batch: 32,
            dimensions: 384,
            reducible_dimensions: false,
            token_input: false,
            pooling: Pooling::Mean,
            normalized: true,
        }
    }
}

fn invalid(parameter: impl Into<String>, detail: impl Into<String>) -> Error {
    Error::InvalidParameter {
        parameter: parameter.into(),
        detail: detail.into(),
    }
}

fn unsupported(model: &str, parameter: impl Into<String>, detail: impl Into<String>) -> Error {
    Error::UnsupportedParameter {
        model: model.to_owned(),
        parameter: parameter.into(),
        detail: detail.into(),
    }
}

/// Checks `request` for the model `model` with capabilities `caps`. Text inputs are
/// checked against `max_input_tokens` by the model, which has the tokenizer.
pub fn validate(
    model: &str,
    caps: &EmbeddingCapabilities,
    request: &EmbeddingRequest,
) -> Result<()> {
    let n = request.inputs.len();
    if n == 0 {
        return Err(invalid("input", "at least one input is needed"));
    }
    if n > MAX_INPUTS {
        return Err(invalid(
            "input",
            format!("{n} inputs given, at most {MAX_INPUTS} are allowed"),
        ));
    }
    for (i, input) in request.inputs.iter().enumerate() {
        match input {
            EmbeddingInput::Text(t) if t.is_empty() => {
                return Err(invalid(format!("input[{i}]"), "an input cannot be empty"));
            }
            EmbeddingInput::Tokens(t) if t.is_empty() => {
                return Err(invalid(format!("input[{i}]"), "an input cannot be empty"));
            }
            EmbeddingInput::Tokens(_) if !caps.token_input => {
                return Err(unsupported(model, format!("input[{i}]"), "token-id input"));
            }
            EmbeddingInput::Tokens(t) if t.len() > caps.max_input_tokens as usize => {
                return Err(invalid(
                    format!("input[{i}]"),
                    format!(
                        "{} tokens, more than the model's {}",
                        t.len(),
                        caps.max_input_tokens
                    ),
                ));
            }
            _ => {}
        }
    }
    if n > caps.max_batch as usize {
        return Err(unsupported(
            model,
            "input",
            format!(
                "{n} inputs in one request; it takes at most {}",
                caps.max_batch
            ),
        ));
    }
    match request.dimensions {
        None => {}
        Some(0) => return Err(invalid("dimensions", "must be at least 1")),
        Some(d) if d > caps.dimensions => {
            return Err(invalid(
                "dimensions",
                format!("{d} is more than the model's {}", caps.dimensions),
            ));
        }
        Some(d) if d < caps.dimensions && !caps.reducible_dimensions => {
            return Err(unsupported(
                model,
                "dimensions",
                format!(
                    "reducing its {} dimensions (it was not trained for it)",
                    caps.dimensions
                ),
            ));
        }
        Some(_) => {}
    }
    Ok(())
}

/// A model that embeds text.
///
/// Object-safe, so models can be held as `Arc<dyn EmbeddingModel>`; as with
/// [`ChatModel`](crate::chat::ChatModel), a model owns its threading and device.
pub trait EmbeddingModel: Send + Sync + 'static {
    /// The model's id and provenance, as `/v1/models` lists it.
    fn info(&self) -> ModelInfo;

    /// What the model can do. Requests are checked against this before they reach
    /// [`EmbeddingModel::embed`].
    fn capabilities(&self) -> EmbeddingCapabilities;

    /// Embeds `request`, which has passed [`validate`]: one vector per input, in order,
    /// each of `request.dimensions` (or the model's full size).
    fn embed(&self, request: EmbeddingRequest) -> BoxFuture<'_, Result<Embeddings>>;
}

impl dyn EmbeddingModel {
    /// Validates `request` against the model and embeds it.
    pub async fn embed_checked(&self, request: EmbeddingRequest) -> Result<Embeddings> {
        validate(&self.info().id, &self.capabilities(), &request)?;
        self.embed(request).await
    }
}

/// `vector` cut to its first `dimensions` and scaled back to unit length, as
/// Matryoshka-trained models are reduced.
pub fn truncate_normalized(mut vector: Vec<f32>, dimensions: usize) -> Vec<f32> {
    vector.truncate(dimensions);
    let norm = vector.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        vector.iter_mut().for_each(|x| *x /= norm);
    }
    vector
}

#[cfg(test)]
mod tests {
    use super::*;

    fn caps() -> EmbeddingCapabilities {
        EmbeddingCapabilities {
            max_input_tokens: 4,
            max_batch: 2,
            dimensions: 8,
            ..EmbeddingCapabilities::default()
        }
    }

    fn request(inputs: Vec<EmbeddingInput>) -> EmbeddingRequest {
        EmbeddingRequest {
            model: "m".into(),
            inputs,
            ..EmbeddingRequest::default()
        }
    }

    fn param(r: Result<()>) -> String {
        match r {
            Err(Error::InvalidParameter { parameter, .. })
            | Err(Error::UnsupportedParameter { parameter, .. }) => parameter,
            other => panic!("not refused by parameter: {other:?}"),
        }
    }

    #[test]
    fn test_validate_accepts_what_the_model_can_do() {
        let ok = request(vec![EmbeddingInput::Text("hi".into())]);
        assert!(validate("m", &caps(), &ok).is_ok());
        let full = EmbeddingRequest {
            dimensions: Some(8),
            ..ok
        };
        assert!(validate("m", &caps(), &full).is_ok());
    }

    #[test]
    fn test_validate_refuses_by_parameter() {
        let text = || EmbeddingInput::Text("hi".into());
        let c = caps();
        assert_eq!(param(validate("m", &c, &request(vec![]))), "input");
        assert_eq!(
            param(validate(
                "m",
                &c,
                &request(vec![text(), EmbeddingInput::Text(String::new())])
            )),
            "input[1]"
        );
        assert_eq!(
            param(validate("m", &c, &request(vec![text(), text(), text()]))),
            "input"
        );
        assert_eq!(
            param(validate(
                "m",
                &c,
                &request(vec![EmbeddingInput::Tokens(vec![1])])
            )),
            "input[0]"
        );
        let tokens = EmbeddingCapabilities {
            token_input: true,
            ..c
        };
        assert_eq!(
            param(validate(
                "m",
                &tokens,
                &request(vec![EmbeddingInput::Tokens(vec![1; 5])])
            )),
            "input[0]"
        );
        for d in [0, 9, 4] {
            let r = EmbeddingRequest {
                dimensions: Some(d),
                ..request(vec![text()])
            };
            assert_eq!(param(validate("m", &c, &r)), "dimensions", "{d}");
        }
        let reducible = EmbeddingCapabilities {
            reducible_dimensions: true,
            ..c
        };
        let r = EmbeddingRequest {
            dimensions: Some(4),
            ..request(vec![text()])
        };
        assert!(validate("m", &reducible, &r).is_ok());
    }

    #[test]
    fn test_truncate_normalized_keeps_unit_length() {
        let v = truncate_normalized(vec![3.0, 4.0, 12.0], 2);
        assert_eq!(v, vec![0.6, 0.8]);
        assert_eq!(truncate_normalized(vec![0.0, 0.0], 1), vec![0.0]);
    }
}
