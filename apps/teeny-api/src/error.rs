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

//! [`Error`]: what serving a request can fail with, whichever domain crate's model it reached.

/// Errors produced by an [`Engine`](crate::Engine).
///
/// The domain crates (teeny-llm, ...) each have an error of their own; they convert into
/// this one, so routes map one type to HTTP.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("model `{requested}` is not loaded; available models: {available:?}")]
    ModelNotFound {
        requested: String,
        available: Vec<String>,
    },

    #[error("invalid request: {0}")]
    InvalidRequest(String),

    /// A request parameter is malformed or out of range, whatever the model.
    #[error("invalid `{parameter}`: {detail}")]
    InvalidParameter { parameter: String, detail: String },

    /// The model cannot do what a request parameter asks for. Refused before it runs,
    /// rather than ignored.
    #[error("model `{model}` does not support `{parameter}`: {detail}")]
    UnsupportedParameter {
        model: String,
        parameter: String,
        detail: String,
    },

    /// The engine does not serve an API at all (it was built without its feature, say).
    #[error("engine `{engine}` does not support {operation}")]
    Unsupported {
        engine: &'static str,
        operation: &'static str,
    },

    /// A model was asked for an API it does not serve (embeddings from a chat model, say).
    #[error("model `{model}` does not serve {api}; it serves {serves}")]
    ModelLacksApi {
        model: String,
        api: &'static str,
        serves: String,
    },

    /// The model failed while running (its tokenizer, weights or inference): the
    /// server's fault, not the request's.
    #[error("{0}")]
    Model(String),
}

pub type Result<T, E = Error> = std::result::Result<T, E>;

#[cfg(feature = "llm")]
impl From<teeny_llm::Error> for Error {
    fn from(e: teeny_llm::Error) -> Self {
        use teeny_llm::Error as E;
        match e {
            E::ModelNotFound {
                requested,
                available,
            } => Self::ModelNotFound {
                requested,
                available,
            },
            E::InvalidRequest(m) => Self::InvalidRequest(m),
            E::InvalidParameter { parameter, detail } => {
                Self::InvalidParameter { parameter, detail }
            }
            E::UnsupportedParameter {
                model,
                parameter,
                detail,
            } => Self::UnsupportedParameter {
                model,
                parameter,
                detail,
            },
            e @ (E::Tokenizer(_) | E::Weights(_) | E::Inference(_)) => Self::Model(e.to_string()),
        }
    }
}
