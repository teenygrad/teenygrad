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

//! Legacy `/v1/completions`.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};
use teeny_llm::chat::Sampling;
use teeny_llm::completion::{
    self as core, CompletionFinishReason, CompletionLogprob, CompletionPrompt, CompletionRequest,
};

use super::error::ApiError;
use super::timings::Timings;
use super::{FinishReason, OneOrMany, StreamOptions, Usage};

/// `prompt`: a string, strings, token ids, or arrays of token ids.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum PromptInput {
    One(String),
    Many(Vec<String>),
    Tokens(Vec<u32>),
    TokenBatches(Vec<Vec<u32>>),
}

/// `CreateCompletionRequest`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CreateCompletionRequest {
    pub model: String,
    pub prompt: PromptInput,
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
    pub best_of: Option<u32>,
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
    pub logit_bias: Option<BTreeMap<String, f32>>,
    #[serde(default)]
    pub logprobs: Option<u8>,
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

impl CreateCompletionRequest {
    /// Whether the completion is to be streamed.
    pub fn streamed(&self) -> bool {
        self.stream.unwrap_or(false)
    }

    /// The request as the core's [`CompletionRequest`].
    pub fn into_core(self) -> Result<CompletionRequest, ApiError> {
        if self.streamed() && self.best_of.is_some_and(|b| b > 1) {
            return Err(ApiError::invalid_request(
                "best_of above 1 cannot be streamed: the candidates are ranked once finished",
                Some("best_of"),
            ));
        }
        let prompts = match self.prompt {
            PromptInput::One(s) => vec![CompletionPrompt::Text(s)],
            PromptInput::Many(v) => v.into_iter().map(CompletionPrompt::Text).collect(),
            PromptInput::Tokens(t) => vec![CompletionPrompt::Tokens(t)],
            PromptInput::TokenBatches(b) => b.into_iter().map(CompletionPrompt::Tokens).collect(),
        };
        let logit_bias = self
            .logit_bias
            .unwrap_or_default()
            .into_iter()
            .map(|(token, bias)| {
                token.parse::<u32>().map(|id| (id, bias)).map_err(|_| {
                    ApiError::invalid_request(
                        format!("`{token}` is not a token id"),
                        Some(&format!("logit_bias.{token}")),
                    )
                })
            })
            .collect::<Result<_, _>>()?;
        Ok(CompletionRequest {
            model: self.model,
            prompts,
            suffix: self.suffix,
            echo: self.echo.unwrap_or(false),
            best_of: self.best_of,
            n: self.n.unwrap_or(1),
            max_tokens: self.max_tokens,
            sampling: Sampling {
                temperature: self.temperature,
                top_p: self.top_p,
                presence_penalty: self.presence_penalty,
                frequency_penalty: self.frequency_penalty,
                logit_bias,
                seed: self.seed.map(i64::cast_unsigned),
                stop: self.stop.map(OneOrMany::into_vec).unwrap_or_default(),
            },
            logprobs: self.logprobs,
            include_usage: self
                .stream_options
                .and_then(|o| o.include_usage)
                .unwrap_or(false),
            user: self.user,
        })
    }
}

/// A choice's `logprobs`, the legacy shape: parallel arrays.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct CompletionLogprobs {
    pub tokens: Vec<String>,
    pub token_logprobs: Vec<f32>,
    pub top_logprobs: Vec<BTreeMap<String, f32>>,
    pub text_offset: Vec<u32>,
}

impl CompletionLogprobs {
    /// The wire logprobs for `logprobs`, when they were asked for.
    pub fn from_core(asked: bool, logprobs: Vec<CompletionLogprob>) -> Option<Self> {
        asked.then(|| {
            let mut out = Self::default();
            for l in logprobs {
                out.tokens.push(l.token);
                out.token_logprobs.push(l.logprob);
                out.top_logprobs.push(l.top.into_iter().collect());
                out.text_offset.push(l.text_offset);
            }
            out
        })
    }
}

impl From<CompletionFinishReason> for FinishReason {
    fn from(r: CompletionFinishReason) -> Self {
        match r {
            CompletionFinishReason::Stop => Self::Stop,
            CompletionFinishReason::Length => Self::Length,
            CompletionFinishReason::ContentFilter => Self::ContentFilter,
        }
    }
}

/// One element of `CreateCompletionResponse.choices`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CompletionChoice {
    pub index: u32,
    pub text: String,
    /// Required-but-nullable: `null` unless `logprobs` was asked for.
    pub logprobs: Option<CompletionLogprobs>,
    /// `null` on intermediate stream chunks.
    pub finish_reason: Option<FinishReason>,
}

impl CompletionChoice {
    /// A finished choice as the wire writes it.
    pub fn from_core(c: core::CompletionChoice, logprobs: bool) -> Self {
        Self {
            index: c.index,
            text: c.text,
            logprobs: CompletionLogprobs::from_core(logprobs, c.logprobs),
            finish_reason: Some(c.finish_reason.into()),
        }
    }
}

/// `CreateCompletionResponse`; also the shape of each streamed chunk.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CreateCompletionResponse {
    pub id: String,
    /// Always `"text_completion"`.
    pub object: String,
    pub created: u64,
    pub model: String,
    pub choices: Vec<CompletionChoice>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub usage: Option<Usage>,
    /// How long it took (llama.cpp's `timings`), from a model that measures itself; when
    /// streamed, on the usage chunk.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub timings: Option<Timings>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_completion_requests_map_every_prompt_shape_and_option() {
        let r: CreateCompletionRequest = serde_json::from_value(json!({
            "model": "m", "prompt": [[1, 2], [3]], "n": 2, "echo": true, "suffix": "}",
            "logprobs": 3, "best_of": 2, "logit_bias": {"5": 1.5}
        }))
        .unwrap();
        let c = r.into_core().unwrap();
        assert_eq!(
            c.prompts,
            vec![
                CompletionPrompt::Tokens(vec![1, 2]),
                CompletionPrompt::Tokens(vec![3])
            ]
        );
        assert_eq!(
            (c.n, c.echo, c.best_of, c.logprobs),
            (2, true, Some(2), Some(3))
        );
        assert_eq!(c.suffix.as_deref(), Some("}"));
        assert_eq!(c.sampling.logit_bias, vec![(5, 1.5)]);
        assert_eq!(c.max_tokens, Some(16));
        let streamed: CreateCompletionRequest = serde_json::from_value(
            json!({"model": "m", "prompt": "x", "best_of": 2, "stream": true}),
        )
        .unwrap();
        assert_eq!(
            streamed.into_core().unwrap_err().body.param.as_deref(),
            Some("best_of")
        );
    }

    #[test]
    fn test_completion_logprobs_use_the_legacy_shape() {
        let l = CompletionLogprobs::from_core(
            true,
            vec![CompletionLogprob {
                token: " a".into(),
                logprob: -0.5,
                text_offset: 3,
                top: vec![(" a".into(), -0.5)],
            }],
        )
        .unwrap();
        assert_eq!(
            serde_json::to_value(&l).unwrap(),
            json!({"tokens": [" a"], "token_logprobs": [-0.5], "top_logprobs": [{" a": -0.5}], "text_offset": [3]})
        );
        assert!(CompletionLogprobs::from_core(false, vec![]).is_none());
    }
}
