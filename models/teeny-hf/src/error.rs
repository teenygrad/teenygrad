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

use std::sync::Arc;

use reqwest::StatusCode;

pub type Result<T> = anyhow::Result<T>;

#[derive(thiserror::Error, Debug)]
pub enum Error {
    #[error("StdError error: {0}")]
    StdError(#[from] std::io::Error),

    #[error("IO error: {0}")]
    IoError(std::io::Error),

    #[error("Serde error: {0}")]
    SerdeError(serde_json::Error),

    #[error("Failed to parse config: {0}")]
    ConfigParseError(#[from] serde_json::Error),

    #[error("Tokenizer error: {0}")]
    TokenizerError(tokenizers::tokenizer::Error),

    #[error("HTTP request failed: {0}")]
    TeenyHttpError(#[from] teeny_http::error::Error),

    #[error("Model not found: {model_id}")]
    ModelNotFound { model_id: String },

    #[error("Model error: {0}")]
    ModelError(String),

    #[error("File not found: {file_path}")]
    FileNotFound { file_path: String },

    #[error("Authentication required for model: {model_id}")]
    AuthenticationRequired { model_id: String },

    #[error("Download failed for file {file_path}: {reason}")]
    DownloadFailed { file_path: String, reason: String },

    #[error("Invalid model ID format: {model_id}")]
    InvalidModelId { model_id: String },

    #[error("Internal server error: {status_code}")]
    InternalServerError { status_code: StatusCode },

    #[error("Parse error: {0}")]
    InvalidHeaderValue(#[from] reqwest::header::InvalidHeaderValue),

    #[error("Acquire error: {0}")]
    AcquireError(#[from] tokio::sync::AcquireError),

    #[error("Invalid response")]
    InvalidResponse,

    #[error("Teeny error: {0}")]
    CoreError(#[from] teeny_core::error::Error),

    #[error("Builder error: {0}")]
    BuilderError(#[from] Arc<dyn std::error::Error + Send + Sync>),
}
