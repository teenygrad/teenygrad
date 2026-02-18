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

use thiserror::Error;

pub type Result<T> = anyhow::Result<T>;

#[derive(Error, Debug)]
pub enum Error {
    #[error("Request failed: {0}")]
    RequestFailed(reqwest::Error),

    #[error("File create failed: {0}")]
    FileCreateFailed(std::io::Error),

    #[error("File write failed: {0}")]
    FileWriteFailed(std::io::Error),

    #[error("HTTP error: {0}")]
    HttpError(reqwest::StatusCode),
}
