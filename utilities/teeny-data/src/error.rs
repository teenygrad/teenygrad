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

pub type Result<T> = anyhow::Result<T>;

#[derive(thiserror::Error, Debug)]
pub enum Error {
    #[error("Failed to download CSV: {0}")]
    DownloadError(#[from] reqwest::Error),

    #[error("Failed to parse CSV: {0}")]
    ParseError(#[from] csv::Error),

    #[error("Failed to convert to ndarray: {0}")]
    ArrayError(#[from] ndarray::ShapeError),

    #[error("Failed to parse value: {0}")]
    ParseValueError(String),

    #[error("SafeTensors error: {0}")]
    SafeTensorsError(teeny_core::safetensors::SafeTensorsError),
}
