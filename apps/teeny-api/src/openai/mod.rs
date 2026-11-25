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

//! OpenAI API wire types, following `openai-openapi.yaml` (spec v2.3.0).
//!
//! Only the inference endpoints are modelled. Request structs deliberately don't use
//! `deny_unknown_fields`: clients send many optional fields we don't act on yet, and OpenAI-
//! compatible servers conventionally ignore them.

pub mod audio;
mod base64;
pub mod chat;
pub mod completions;
pub mod embeddings;
pub mod error;
pub mod images;
pub mod models;
pub mod moderations;
pub mod speech;

use serde::{Deserialize, Serialize};

/// A field the spec types as `string | string[]` (e.g. `stop`, `prompt`, `input`).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum OneOrMany {
    One(String),
    Many(Vec<String>),
}

impl OneOrMany {
    pub fn into_vec(self) -> Vec<String> {
        match self {
            OneOrMany::One(s) => vec![s],
            OneOrMany::Many(v) => v,
        }
    }
}

/// `CompletionUsage`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct Usage {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub total_tokens: u32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prompt_tokens_details: Option<PromptTokensDetails>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub completion_tokens_details: Option<CompletionTokensDetails>,
}

/// `CompletionUsage.prompt_tokens_details`.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct PromptTokensDetails {
    pub cached_tokens: u32,
    pub audio_tokens: u32,
}

/// `CompletionUsage.completion_tokens_details`.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct CompletionTokensDetails {
    pub reasoning_tokens: u32,
    pub audio_tokens: u32,
    pub accepted_prediction_tokens: u32,
    pub rejected_prediction_tokens: u32,
}

impl From<teeny_llm::Usage> for Usage {
    fn from(u: teeny_llm::Usage) -> Self {
        Self {
            prompt_tokens: u.prompt_tokens,
            completion_tokens: u.completion_tokens,
            total_tokens: u.total_tokens(),
            prompt_tokens_details: None,
            completion_tokens_details: None,
        }
    }
}

impl From<teeny_llm::chat::ChatUsage> for Usage {
    fn from(u: teeny_llm::chat::ChatUsage) -> Self {
        Self {
            prompt_tokens: u.prompt_tokens,
            completion_tokens: u.completion_tokens,
            total_tokens: u.total_tokens(),
            prompt_tokens_details: Some(PromptTokensDetails {
                cached_tokens: u.cached_prompt_tokens,
                audio_tokens: u.audio_prompt_tokens,
            }),
            completion_tokens_details: Some(CompletionTokensDetails {
                reasoning_tokens: u.reasoning_tokens,
                audio_tokens: u.audio_completion_tokens,
                accepted_prediction_tokens: u.accepted_prediction_tokens,
                rejected_prediction_tokens: u.rejected_prediction_tokens,
            }),
        }
    }
}

/// `ChatCompletionStreamOptions`.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct StreamOptions {
    /// Send a final chunk with empty `choices` carrying the request's `usage`.
    #[serde(default)]
    pub include_usage: Option<bool>,
}

/// The `finish_reason` enum shared by chat and legacy completions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FinishReason {
    Stop,
    Length,
    ToolCalls,
    ContentFilter,
    FunctionCall,
}
