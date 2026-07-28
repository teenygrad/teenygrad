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

/// `teeny-onnx`'s result alias.
pub type Result<T> = anyhow::Result<T>;

/// Errors produced by [`crate::Onnx`].
#[derive(thiserror::Error, Debug)]
pub enum Error {
    /// The `.onnx` file parsed as valid protobuf but its contents don't form a valid/supported
    /// model (e.g. an unsupported op).
    #[error("Invalid model: {0}")]
    InvalidModel(String),
}
