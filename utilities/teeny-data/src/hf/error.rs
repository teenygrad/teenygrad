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

use std::io;

use reqwest::StatusCode;

/// Result type for download operations
pub type DownloadResult<T> = Result<T, DownloadError>;

/// Error types for download operations
#[derive(Debug, thiserror::Error)]
pub enum DownloadError {
    #[error("HTTP request failed: {0}")]
    HttpError(#[from] reqwest::Error),

    #[error("IO error: {0}")]
    IoError(#[from] io::Error),

    #[error("JSON parsing error: {0}")]
    JsonError(#[from] serde_json::Error),

    #[error("Model not found: {model_id}")]
    ModelNotFound { model_id: String },

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
}
