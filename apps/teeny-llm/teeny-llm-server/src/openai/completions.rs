//! Legacy `/v1/completions`.

use serde::{Deserialize, Serialize};

use super::{FinishReason, OneOrMany, StreamOptions, Usage};

/// `CreateCompletionRequest`. Token-array prompts aren't supported yet.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CreateCompletionRequest {
    pub model: String,
    pub prompt: OneOrMany,
    /// The spec's default is 16.
    #[serde(default = "default_max_tokens")]
    pub max_tokens: Option<u32>,
    #[serde(default)]
    pub temperature: Option<f32>,
    #[serde(default)]
    pub top_p: Option<f32>,
    #[serde(default)]
    pub n: Option<u32>,
    #[serde(default)]
    pub stop: Option<OneOrMany>,
    #[serde(default)]
    pub stream: Option<bool>,
    #[serde(default)]
    pub stream_options: Option<StreamOptions>,
    #[serde(default)]
    pub seed: Option<i64>,
    #[serde(default)]
    pub presence_penalty: Option<f32>,
    #[serde(default)]
    pub frequency_penalty: Option<f32>,
    #[serde(default)]
    pub echo: Option<bool>,
    #[serde(default)]
    pub suffix: Option<String>,
    #[serde(default)]
    pub user: Option<String>,
}

fn default_max_tokens() -> Option<u32> {
    Some(16)
}

/// One element of `CreateCompletionResponse.choices`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CompletionChoice {
    pub index: u32,
    pub text: String,
    /// Required-but-nullable; always `null` until logprobs are supported.
    pub logprobs: Option<serde_json::Value>,
    /// `null` on intermediate stream chunks.
    pub finish_reason: Option<FinishReason>,
}

/// `CreateCompletionResponse`; also the shape of each streamed chunk.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CreateCompletionResponse {
    pub id: String,
    /// Always `"text_completion"`.
    pub object: String,
    pub created: u64,
    pub model: String,
    pub choices: Vec<CompletionChoice>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub usage: Option<Usage>,
}
