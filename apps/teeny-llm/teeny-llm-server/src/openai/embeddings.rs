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
