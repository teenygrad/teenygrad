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

//! `/v1/chat/completions`.

use serde::{Deserialize, Serialize};

use super::{FinishReason, OneOrMany, StreamOptions, Usage};

/// Message roles, as in `ChatCompletionRequestMessage` and `ChatCompletionStreamResponseDelta`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    Developer,
    System,
    User,
    Assistant,
    Tool,
}

impl From<Role> for teeny_llm_core::Role {
    fn from(r: Role) -> Self {
        match r {
            Role::Developer => teeny_llm_core::Role::Developer,
            Role::System => teeny_llm_core::Role::System,
            Role::User => teeny_llm_core::Role::User,
            Role::Assistant => teeny_llm_core::Role::Assistant,
            Role::Tool => teeny_llm_core::Role::Tool,
        }
    }
}

/// One element of an array-form message `content`. Only `type: "text"` is supported; other
/// part types (`image_url`, `input_audio`, `file`, ...) are parsed so they can be rejected by name.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ContentPart {
    #[serde(rename = "type")]
    pub kind: String,
    #[serde(default)]
    pub text: Option<String>,
}

/// Message `content`: a string or an array of content parts.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum MessageContent {
    Text(String),
    Parts(Vec<ContentPart>),
}

/// `ChatCompletionRequestMessage` (all role variants, flattened).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ChatCompletionRequestMessage {
    pub role: Role,
    /// Nullable for assistant messages that only carry tool calls.
    #[serde(default)]
    pub content: Option<MessageContent>,
    #[serde(default)]
    pub name: Option<String>,
}

/// `CreateChatCompletionRequest`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CreateChatCompletionRequest {
    pub model: String,
    pub messages: Vec<ChatCompletionRequestMessage>,
    /// Deprecated in favour of `max_completion_tokens`, but still widely sent.
    #[serde(default)]
    pub max_tokens: Option<u32>,
    #[serde(default)]
    pub max_completion_tokens: Option<u32>,
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
    pub user: Option<String>,
}

/// `ChatCompletionResponseMessage`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ChatCompletionResponseMessage {
    pub role: Role,
    /// Required-but-nullable.
    pub content: Option<String>,
    pub refusal: Option<String>,
}

/// One element of `CreateChatCompletionResponse.choices`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ChatCompletionChoice {
    pub index: u32,
    pub message: ChatCompletionResponseMessage,
    /// Required-but-nullable; always `null` until logprobs are supported.
    pub logprobs: Option<serde_json::Value>,
    pub finish_reason: FinishReason,
}

/// `CreateChatCompletionResponse`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CreateChatCompletionResponse {
    pub id: String,
    /// Always `"chat.completion"`.
    pub object: String,
    pub created: u64,
    pub model: String,
    pub choices: Vec<ChatCompletionChoice>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub usage: Option<Usage>,
}

/// `ChatCompletionStreamResponseDelta`.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ChatCompletionStreamResponseDelta {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub role: Option<Role>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
}

/// One element of `CreateChatCompletionStreamResponse.choices`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ChatCompletionStreamChoice {
    pub index: u32,
    pub delta: ChatCompletionStreamResponseDelta,
    pub logprobs: Option<serde_json::Value>,
    pub finish_reason: Option<FinishReason>,
}

/// `CreateChatCompletionStreamResponse`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CreateChatCompletionStreamResponse {
    pub id: String,
    /// Always `"chat.completion.chunk"`.
    pub object: String,
    pub created: u64,
    pub model: String,
    pub choices: Vec<ChatCompletionStreamChoice>,
    /// Only on the final chunk, and only with `stream_options.include_usage`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub usage: Option<Usage>,
}
