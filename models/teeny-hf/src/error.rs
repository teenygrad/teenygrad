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

use reqwest::StatusCode;

pub type Result<T> = std::result::Result<T, Error>;

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
    BuilderError(String),
}
