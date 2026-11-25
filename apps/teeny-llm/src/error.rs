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

/// Errors produced by a model, its API's `validate`, or the model factory.
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

    /// The model cannot do what a request parameter asks for. Refused before generating,
    /// rather than ignored.
    #[error("model `{model}` does not support `{parameter}`: {detail}")]
    UnsupportedParameter {
        model: String,
        parameter: String,
        detail: String,
    },

    /// Text could not be tokenized, or token ids decoded.
    #[error("tokenizer: {0}")]
    Tokenizer(String),

    /// A model's weights could not be found or are incomplete.
    #[error("model weights: {0}")]
    Weights(String),

    #[error("inference failed: {0}")]
    Inference(String),
}

pub type Result<T, E = Error> = std::result::Result<T, E>;
