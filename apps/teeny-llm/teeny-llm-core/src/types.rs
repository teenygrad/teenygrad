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

/// Who authored a [`ChatMessage`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Role {
    /// Instructions from the application developer (OpenAI's newer name for `System`).
    Developer,
    System,
    User,
    Assistant,
    /// The result of a tool call.
    Tool,
}

impl Role {
    pub fn as_str(self) -> &'static str {
        match self {
            Role::Developer => "developer",
            Role::System => "system",
            Role::User => "user",
            Role::Assistant => "assistant",
            Role::Tool => "tool",
        }
    }
}

/// One turn of a conversation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChatMessage {
    pub role: Role,
    pub content: String,
}

impl ChatMessage {
    pub fn new(role: Role, content: impl Into<String>) -> Self {
        Self {
            role,
            content: content.into(),
        }
    }

    pub fn system(content: impl Into<String>) -> Self {
        Self::new(Role::System, content)
    }

    pub fn user(content: impl Into<String>) -> Self {
        Self::new(Role::User, content)
    }

    pub fn assistant(content: impl Into<String>) -> Self {
        Self::new(Role::Assistant, content)
    }
}

/// The input to generate from.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Prompt {
    /// A conversation, rendered through the model's chat template.
    Chat(Vec<ChatMessage>),
    /// Raw text, continued as-is (legacy completions).
    Text(String),
}

/// Decoding controls. `None` means "use the model's default".
#[derive(Debug, Clone, Default, PartialEq)]
pub struct SamplingParams {
    pub max_tokens: Option<u32>,
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    pub presence_penalty: Option<f32>,
    pub frequency_penalty: Option<f32>,
    pub seed: Option<u64>,
    /// Generation stops before emitting any of these strings.
    pub stop: Vec<String>,
}

/// A single generation job.
#[derive(Debug, Clone, PartialEq)]
pub struct GenerationRequest {
    pub model: String,
    pub prompt: Prompt,
    pub sampling: SamplingParams,
}

/// Why generation ended.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FinishReason {
    /// The model emitted end-of-sequence or hit a stop string.
    Stop,
    /// `max_tokens` (or the context window) was reached.
    Length,
}

/// Token accounting for a request.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Usage {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
}

impl Usage {
    pub fn total_tokens(&self) -> u32 {
        self.prompt_tokens + self.completion_tokens
    }
}

/// One increment of a streamed generation.
///
/// Every chunk but the last has `finish_reason == None`; the last one carries the finish reason
/// and the final usage (its `text` may be empty).
#[derive(Debug, Clone, PartialEq)]
pub struct GenerationChunk {
    pub text: String,
    pub finish_reason: Option<FinishReason>,
    pub usage: Option<Usage>,
}

/// A complete (non-streamed) generation.
#[derive(Debug, Clone, PartialEq)]
pub struct Generation {
    pub text: String,
    pub finish_reason: FinishReason,
    pub usage: Usage,
}

/// The result of embedding a batch of inputs, one vector per input in order.
#[derive(Debug, Clone, PartialEq)]
pub struct Embeddings {
    pub vectors: Vec<Vec<f32>>,
    pub prompt_tokens: u32,
}

/// A model an [`Engine`](crate::Engine) can serve.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelInfo {
    pub id: String,
    /// Unix timestamp (seconds) the model was loaded/created.
    pub created: u64,
    pub owned_by: String,
}
