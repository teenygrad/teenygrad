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

//! Text completions (OpenAI's legacy `/v1/completions`): raw continuation of a prompt,
//! for base and code models.
//!
//! A model implements [`CompletionModel`]: it declares its [`CompletionCapabilities`] and
//! answers a [`CompletionRequest`] with a [`CompletionStream`] of [`CompletionEvent`]s.
//! [`validate`] checks each request first, refusing what the model cannot do (`echo`,
//! `suffix`, `best_of`, log probabilities) by parameter. [`collect`] puts a stream together
//! into a [`Completion`]. [`EchoCompletionModel`] is the reference implementation.
//!
//! Chat models do not serve completions through their chat template: continuing raw text
//! is a different contract, and wrapping a prompt in a template a client did not ask for
//! would answer something else. Completions come only from models that declare this trait.

mod echo;

pub use echo::EchoCompletionModel;

use std::pin::Pin;

use futures::future::BoxFuture;
use futures::{Stream, StreamExt};

use crate::chat::{MAX_STOP_STRINGS, Sampling, SamplingSupport};
use crate::{Error, ModelInfo, Result, Timings, Usage};

/// The OpenAI limit on `logprobs` in a completion request.
pub const MAX_LOGPROBS: u8 = 5;

/// One prompt to continue.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CompletionPrompt {
    Text(String),
    /// Already tokenized, as the model's tokenizer would.
    Tokens(Vec<u32>),
}

/// A completion request. Choices are numbered prompt-major: prompt `p`'s `j`th choice is
/// index `p * n + j`. `Default` gives an empty request with one choice.
#[derive(Debug, Clone, PartialEq)]
pub struct CompletionRequest {
    pub model: String,
    pub prompts: Vec<CompletionPrompt>,
    /// Text that comes after the completion (fill-in-the-middle).
    pub suffix: Option<String>,
    /// Whether each choice starts with its prompt.
    pub echo: bool,
    /// Generate this many candidates per prompt and return the `n` most likely.
    pub best_of: Option<u32>,
    /// Choices per prompt.
    pub n: u32,
    /// Most tokens per choice.
    pub max_tokens: Option<u32>,
    pub sampling: Sampling,
    /// How many alternatives to give per token, with each token's log probability.
    pub logprobs: Option<u8>,
    /// Whether a stream ends with the request's usage.
    pub include_usage: bool,
    pub user: Option<String>,
}

impl Default for CompletionRequest {
    fn default() -> Self {
        Self {
            model: String::new(),
            prompts: Vec::new(),
            suffix: None,
            echo: false,
            best_of: None,
            n: 1,
            max_tokens: None,
            sampling: Sampling::default(),
            logprobs: None,
            include_usage: false,
            user: None,
        }
    }
}

/// A token's log probability, where it starts in the choice's text, and its likeliest
/// alternatives.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct CompletionLogprob {
    pub token: String,
    pub logprob: f32,
    /// Byte offset of the token in the choice's text (echoed prompt included).
    pub text_offset: u32,
    /// The likeliest tokens at this position, most likely first.
    pub top: Vec<(String, f32)>,
}

/// Why a choice ended.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompletionFinishReason {
    Stop,
    Length,
    ContentFilter,
}

/// One increment of a completion: per choice any number of texts then one finish, and a
/// final usage, as for chat.
#[derive(Debug, Clone, PartialEq)]
pub enum CompletionEvent {
    Text {
        index: u32,
        text: String,
        /// For the tokens in `text`, when asked for.
        logprobs: Vec<CompletionLogprob>,
    },
    Finish {
        index: u32,
        reason: CompletionFinishReason,
    },
    /// How long the request took, phase by phase.
    Timings(Timings),
    Usage(Usage),
}

/// A completion model's output, as it is generated.
pub type CompletionStream = Pin<Box<dyn Stream<Item = Result<CompletionEvent>> + Send>>;

/// One finished choice.
#[derive(Debug, Clone, PartialEq)]
pub struct CompletionChoice {
    pub index: u32,
    pub text: String,
    pub logprobs: Vec<CompletionLogprob>,
    pub finish_reason: CompletionFinishReason,
}

/// A whole completion.
#[derive(Debug, Clone, PartialEq)]
pub struct Completion {
    pub choices: Vec<CompletionChoice>,
    pub usage: Usage,
    /// How long it took, from a model that measures itself.
    pub timings: Option<Timings>,
}

/// What a completion model can do.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CompletionCapabilities {
    /// Most tokens of prompt and completion together.
    pub context_length: u32,
    /// Whether `suffix` (fill-in-the-middle) is supported.
    pub fill_in_middle: bool,
    /// Whether choices can start with their prompt (`echo`).
    pub echo: bool,
    /// The largest `best_of`; 1 without candidate ranking.
    pub max_best_of: u32,
    /// The most choices per prompt (`n`).
    pub max_choices: u32,
    /// The most alternatives per token, or `None` without log probabilities.
    pub max_logprobs: Option<u8>,
    pub sampling: SamplingSupport,
    pub stop_strings: bool,
    /// Whether prompts can be given as token ids.
    pub token_prompts: bool,
}

impl Default for CompletionCapabilities {
    fn default() -> Self {
        Self {
            context_length: 2048,
            fill_in_middle: false,
            echo: false,
            max_best_of: 1,
            max_choices: 1,
            max_logprobs: None,
            sampling: SamplingSupport::default(),
            stop_strings: false,
            token_prompts: false,
        }
    }
}

fn invalid(parameter: impl Into<String>, detail: impl Into<String>) -> Error {
    Error::InvalidParameter {
        parameter: parameter.into(),
        detail: detail.into(),
    }
}

fn unsupported(model: &str, parameter: impl Into<String>, detail: impl Into<String>) -> Error {
    Error::UnsupportedParameter {
        model: model.to_owned(),
        parameter: parameter.into(),
        detail: detail.into(),
    }
}

fn check_range(parameter: &str, value: Option<f32>, lo: f32, hi: f32) -> Result<()> {
    match value {
        Some(v) if !(lo..=hi).contains(&v) => {
            Err(invalid(parameter, format!("{v} is outside {lo} to {hi}")))
        }
        _ => Ok(()),
    }
}

/// Checks `request` for the model `model` with capabilities `caps`: well formed, then
/// within what the model can do. The first problem is returned, naming the parameter.
pub fn validate(model: &str, caps: &CompletionCapabilities, r: &CompletionRequest) -> Result<()> {
    if r.prompts.is_empty() {
        return Err(invalid("prompt", "at least one prompt is needed"));
    }
    if r.n == 0 {
        return Err(invalid("n", "at least one choice must be asked for"));
    }
    if r.max_tokens == Some(0) {
        return Err(invalid("max_tokens", "must be at least 1 when given"));
    }
    if let Some(b) = r.best_of
        && b < r.n
    {
        return Err(invalid("best_of", format!("{b} is less than n ({})", r.n)));
    }
    if let Some(l) = r.logprobs
        && l > MAX_LOGPROBS
    {
        return Err(invalid(
            "logprobs",
            format!("{l} is above the limit of {MAX_LOGPROBS}"),
        ));
    }
    let s = &r.sampling;
    check_range("temperature", s.temperature, 0.0, 2.0)?;
    check_range("top_p", s.top_p, 0.0, 1.0)?;
    check_range("presence_penalty", s.presence_penalty, -2.0, 2.0)?;
    check_range("frequency_penalty", s.frequency_penalty, -2.0, 2.0)?;
    if s.stop.len() > MAX_STOP_STRINGS {
        return Err(invalid(
            "stop",
            format!(
                "{} stop strings given, at most {MAX_STOP_STRINGS} are allowed",
                s.stop.len()
            ),
        ));
    }
    if let Some(i) = s.stop.iter().position(String::is_empty) {
        return Err(invalid(
            format!("stop[{i}]"),
            "a stop string cannot be empty",
        ));
    }

    for (i, p) in r.prompts.iter().enumerate() {
        if let CompletionPrompt::Tokens(t) = p {
            if !caps.token_prompts {
                return Err(unsupported(
                    model,
                    format!("prompt[{i}]"),
                    "token-id prompts",
                ));
            }
            if t.len() >= caps.context_length as usize {
                return Err(invalid(
                    format!("prompt[{i}]"),
                    format!(
                        "{} tokens leave no room in the model's {}-token context",
                        t.len(),
                        caps.context_length
                    ),
                ));
            }
        }
    }
    if r.suffix.is_some() && !caps.fill_in_middle {
        return Err(unsupported(model, "suffix", "fill-in-the-middle"));
    }
    if r.echo && !caps.echo {
        return Err(unsupported(model, "echo", "echoing the prompt"));
    }
    if r.n > caps.max_choices {
        return Err(unsupported(
            model,
            "n",
            format!("{} choices; it gives at most {}", r.n, caps.max_choices),
        ));
    }
    if let Some(b) = r.best_of.filter(|&b| b > caps.max_best_of) {
        return Err(unsupported(
            model,
            "best_of",
            format!("{b} candidates; it ranks at most {}", caps.max_best_of),
        ));
    }
    if let Some(l) = r.logprobs {
        match caps.max_logprobs {
            None => return Err(unsupported(model, "logprobs", "log probabilities")),
            Some(m) if l > m => {
                return Err(unsupported(
                    model,
                    "logprobs",
                    format!("{l} alternatives; it gives at most {m}"),
                ));
            }
            Some(_) => {}
        }
    }
    let sc = &caps.sampling;
    let refuse = |p: &str, what: &str| Err(unsupported(model, p, what.to_owned()));
    if !sc.temperature && s.temperature.is_some_and(|t| t != 0.0) {
        return refuse("temperature", "sampling (it decodes greedily)");
    }
    if !sc.top_p && s.top_p.is_some_and(|p| p != 1.0) {
        return refuse("top_p", "nucleus sampling");
    }
    if !sc.penalties
        && (s.presence_penalty.is_some_and(|p| p != 0.0)
            || s.frequency_penalty.is_some_and(|p| p != 0.0))
    {
        return refuse("presence_penalty", "repetition penalties");
    }
    if !sc.logit_bias && !s.logit_bias.is_empty() {
        return refuse("logit_bias", "logit bias");
    }
    if !sc.seed && s.seed.is_some() && sc != &SamplingSupport::default() {
        return refuse("seed", "the model samples but cannot be seeded");
    }
    if !caps.stop_strings && !s.stop.is_empty() {
        return refuse("stop", "stop strings");
    }
    Ok(())
}

/// A model that continues raw text.
///
/// Object-safe, so models can be held as `Arc<dyn CompletionModel>`; as with
/// [`ChatModel`](crate::chat::ChatModel), a model owns its threading and device.
pub trait CompletionModel: Send + Sync + 'static {
    /// The model's id and provenance, as `/v1/models` lists it.
    fn info(&self) -> ModelInfo;

    /// What the model can do. Requests are checked against this first.
    fn capabilities(&self) -> CompletionCapabilities;

    /// Starts completing `request`, which has passed [`validate`], resolving to the stream
    /// of its output once generation has begun. Errors before any output resolve the
    /// future to `Err`; errors part-way end the stream with one.
    fn complete(&self, request: CompletionRequest) -> BoxFuture<'_, Result<CompletionStream>>;
}

/// Puts `stream`, the output of a request for `choices` choices in all (prompts × `n`),
/// together into a [`Completion`].
pub async fn collect(choices: usize, mut stream: CompletionStream) -> Result<Completion> {
    let mut texts: Vec<(
        String,
        Vec<CompletionLogprob>,
        Option<CompletionFinishReason>,
    )> = vec![(String::new(), Vec::new(), None); choices];
    let mut usage = None;
    let mut timings = None;
    while let Some(event) = stream.next().await {
        match event? {
            CompletionEvent::Text {
                index,
                text,
                logprobs,
            } => {
                let c = texts
                    .get_mut(index as usize)
                    .ok_or_else(|| out_of_range(index, choices))?;
                c.0.push_str(&text);
                c.1.extend(logprobs);
            }
            CompletionEvent::Finish { index, reason } => {
                texts
                    .get_mut(index as usize)
                    .ok_or_else(|| out_of_range(index, choices))?
                    .2 = Some(reason);
            }
            CompletionEvent::Timings(t) => timings = Some(t),
            CompletionEvent::Usage(u) => usage = Some(u),
        }
    }
    let choices = texts
        .into_iter()
        .enumerate()
        .map(|(i, (text, logprobs, finish))| {
            Ok(CompletionChoice {
                index: u32::try_from(i).unwrap_or(u32::MAX),
                text,
                logprobs,
                finish_reason: finish.ok_or_else(|| {
                    Error::Inference(format!(
                        "the completion stream ended without finishing choice {i}"
                    ))
                })?,
            })
        })
        .collect::<Result<_>>()?;
    Ok(Completion {
        choices,
        usage: usage.ok_or_else(|| {
            Error::Inference("the completion stream ended without its usage".into())
        })?,
        timings,
    })
}

fn out_of_range(index: u32, choices: usize) -> Error {
    Error::Inference(format!(
        "the completion stream sent choice {index} of a request for {choices}"
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn request() -> CompletionRequest {
        CompletionRequest {
            model: "m".into(),
            prompts: vec![CompletionPrompt::Text("once upon".into())],
            ..CompletionRequest::default()
        }
    }

    fn param(r: Result<()>) -> String {
        match r {
            Err(Error::InvalidParameter { parameter, .. })
            | Err(Error::UnsupportedParameter { parameter, .. }) => parameter,
            other => panic!("not refused by parameter: {other:?}"),
        }
    }

    #[test]
    fn test_validate_refuses_by_parameter() {
        let caps = CompletionCapabilities::default();
        assert!(validate("m", &caps, &request()).is_ok());
        let cases: Vec<(CompletionRequest, &str)> = vec![
            (
                CompletionRequest {
                    prompts: vec![],
                    ..request()
                },
                "prompt",
            ),
            (CompletionRequest { n: 0, ..request() }, "n"),
            (CompletionRequest { n: 2, ..request() }, "n"),
            (
                CompletionRequest {
                    best_of: Some(1),
                    n: 2,
                    ..request()
                },
                "best_of",
            ),
            (
                CompletionRequest {
                    best_of: Some(3),
                    ..request()
                },
                "best_of",
            ),
            (
                CompletionRequest {
                    echo: true,
                    ..request()
                },
                "echo",
            ),
            (
                CompletionRequest {
                    suffix: Some("}".into()),
                    ..request()
                },
                "suffix",
            ),
            (
                CompletionRequest {
                    logprobs: Some(6),
                    ..request()
                },
                "logprobs",
            ),
            (
                CompletionRequest {
                    logprobs: Some(1),
                    ..request()
                },
                "logprobs",
            ),
            (
                CompletionRequest {
                    prompts: vec![CompletionPrompt::Tokens(vec![1])],
                    ..request()
                },
                "prompt[0]",
            ),
            (
                CompletionRequest {
                    sampling: Sampling {
                        temperature: Some(0.5),
                        ..Sampling::default()
                    },
                    ..request()
                },
                "temperature",
            ),
            (
                CompletionRequest {
                    sampling: Sampling {
                        stop: vec!["\n".into()],
                        ..Sampling::default()
                    },
                    ..request()
                },
                "stop",
            ),
        ];
        for (r, want) in cases {
            assert_eq!(param(validate("m", &caps, &r)), want);
        }
    }

    #[tokio::test]
    async fn test_collect_assembles_choices_in_order() {
        let events = vec![
            CompletionEvent::Text {
                index: 1,
                text: "b".into(),
                logprobs: vec![],
            },
            CompletionEvent::Text {
                index: 0,
                text: "a".into(),
                logprobs: vec![],
            },
            CompletionEvent::Finish {
                index: 0,
                reason: CompletionFinishReason::Stop,
            },
            CompletionEvent::Finish {
                index: 1,
                reason: CompletionFinishReason::Length,
            },
            CompletionEvent::Usage(Usage {
                prompt_tokens: 1,
                completion_tokens: 2,
            }),
        ];
        let stream: CompletionStream = Box::pin(futures::stream::iter(events.into_iter().map(Ok)));
        let c = collect(2, stream).await.unwrap();
        assert_eq!(
            (c.choices[0].text.as_str(), c.choices[1].text.as_str()),
            ("a", "b")
        );
        assert_eq!(c.choices[1].finish_reason, CompletionFinishReason::Length);
        let short: CompletionStream = Box::pin(futures::stream::iter(vec![Ok(
            CompletionEvent::Usage(Usage::default()),
        )]));
        assert!(collect(1, short).await.is_err());
    }
}
