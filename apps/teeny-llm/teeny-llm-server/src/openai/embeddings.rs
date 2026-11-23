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

use serde::{Deserialize, Serialize};

use super::OneOrMany;

/// `CreateEmbeddingRequest`. Token-array inputs aren't supported yet.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CreateEmbeddingRequest {
    pub model: String,
    pub input: OneOrMany,
    /// `"float"` (default) or `"base64"`; only `"float"` is supported.
    #[serde(default)]
    pub encoding_format: Option<String>,
    #[serde(default)]
    pub dimensions: Option<u32>,
    #[serde(default)]
    pub user: Option<String>,
}

/// `Embedding`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Embedding {
    /// Always `"embedding"`.
    pub object: String,
    pub index: u32,
    pub embedding: Vec<f32>,
}

/// The `usage` object of `CreateEmbeddingResponse`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct EmbeddingUsage {
    pub prompt_tokens: u32,
    pub total_tokens: u32,
}

/// `CreateEmbeddingResponse`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CreateEmbeddingResponse {
    /// Always `"list"`.
    pub object: String,
    pub data: Vec<Embedding>,
    pub model: String,
    pub usage: EmbeddingUsage,
}
