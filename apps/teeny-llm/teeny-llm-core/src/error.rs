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

/// Errors produced by an [`Engine`](crate::Engine).
#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("model `{requested}` is not loaded; available models: {available:?}")]
    ModelNotFound {
        requested: String,
        available: Vec<String>,
    },

    #[error("invalid request: {0}")]
    InvalidRequest(String),

    #[error("engine `{engine}` does not support {operation}")]
    Unsupported {
        engine: &'static str,
        operation: &'static str,
    },

    #[error("inference failed: {0}")]
    Inference(String),
}

pub type Result<T, E = Error> = std::result::Result<T, E>;
