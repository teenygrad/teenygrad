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

//! What a chat model produces: a stream of [`ChatEvent`]s, and the [`ChatCompletion`] they
//! add up to.

use std::pin::Pin;

use futures::{Stream, StreamExt};

use super::request::{AssistantMessage, AudioReply, ToolCall, ToolCallKind};
use crate::{Error, Result, Timings};

/// A chat model's output, as it is generated.
pub type ChatStream = Pin<Box<dyn Stream<Item = Result<ChatEvent>> + Send>>;

/// One increment of a chat completion.
///
/// For each choice (`index` 0 to `n - 1`) a stream carries any number of
/// [`ChatEvent::Delta`]s and then exactly one [`ChatEvent::Finish`]. Choices may interleave.
/// The stream ends with one [`ChatEvent::Usage`], whether or not the request asked to see it
/// (frontends drop it when `include_usage` is off), after at most one [`ChatEvent::Timings`]
/// from a model that measures itself.
#[derive(Debug, Clone, PartialEq)]
pub enum ChatEvent {
    Delta {
        index: u32,
        delta: Delta,
    },
    Finish {
        index: u32,
        reason: ChatFinishReason,
    },
    /// How long the request took, phase by phase.
    Timings(Timings),
    Usage(ChatUsage),
}

/// New output for one choice. Text fields are appended to what came before.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Delta {
    pub content: Option<String>,
    pub refusal: Option<String>,
    pub tool_calls: Vec<ToolCallDelta>,
    pub audio: Option<AudioDelta>,
    /// Log probabilities of the tokens in `content`, when they were asked for.
    pub content_logprobs: Vec<TokenLogprob>,
    /// Log probabilities of the tokens in `refusal`, when they were asked for.
    pub refusal_logprobs: Vec<TokenLogprob>,
}

/// Part of a tool call. The first delta for an `index` carries its `id`, `name` and
/// `kind`; later ones only append to `input`.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct ToolCallDelta {
    /// Which of the choice's tool calls this extends, from 0.
    pub index: u32,
    pub id: Option<String>,
    pub name: Option<String>,
    pub kind: Option<ToolCallKind>,
    pub input: String,
}

/// Part of a spoken reply.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct AudioDelta {
    pub id: Option<String>,
    pub data: Vec<u8>,
    pub transcript: String,
    pub expires_at: Option<u64>,
}

/// A generated token's log probability, and the most likely alternatives.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct TokenLogprob {
    pub token: String,
    /// The token's UTF-8 bytes; a token can be part of a character.
    pub bytes: Vec<u8>,
    pub logprob: f32,
    /// The `Logprobs::top` most likely tokens at this position, most likely first.
    pub top: Vec<TopLogprob>,
}

/// An alternative token at a position.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct TopLogprob {
    pub token: String,
    pub bytes: Vec<u8>,
    pub logprob: f32,
}

/// Why a choice ended.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChatFinishReason {
    /// The model finished its turn, or hit a stop string.
    Stop,
    /// The token limit (or the context window) was reached.
    Length,
    /// The model called one or more tools and is waiting for their results.
    ToolCalls,
    /// Output was withheld by a content filter.
    ContentFilter,
}

/// Token accounting for a whole request, every choice included.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ChatUsage {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    /// Prompt tokens served from a cache (a KV-cache prefix shared with an earlier turn).
    pub cached_prompt_tokens: u32,
    /// Completion tokens spent reasoning rather than answering.
    pub reasoning_tokens: u32,
    pub audio_prompt_tokens: u32,
    pub audio_completion_tokens: u32,
    /// Predicted-output tokens the reply used.
    pub accepted_prediction_tokens: u32,
    /// Predicted-output tokens the reply did not use (still billed as completion).
    pub rejected_prediction_tokens: u32,
}

impl ChatUsage {
    pub fn total_tokens(&self) -> u32 {
        self.prompt_tokens.saturating_add(self.completion_tokens)
    }
}

/// A finished chat completion.
#[derive(Debug, Clone, PartialEq)]
pub struct ChatCompletion {
    pub model: String,
    /// One per requested choice, in `index` order.
    pub choices: Vec<Choice>,
    pub usage: ChatUsage,
    /// How long it took, from a model that measures itself.
    pub timings: Option<Timings>,
}

/// One finished choice.
#[derive(Debug, Clone, PartialEq)]
pub struct Choice {
    pub index: u32,
    pub message: AssistantMessage,
    pub finish_reason: ChatFinishReason,
    pub content_logprobs: Vec<TokenLogprob>,
    pub refusal_logprobs: Vec<TokenLogprob>,
}

/// A choice being put together from its deltas.
#[derive(Default)]
struct Partial {
    message: AssistantMessage,
    /// Tool calls by their delta `index`, which need not arrive in order.
    calls: Vec<(u32, ToolCall)>,
    finish: Option<ChatFinishReason>,
    content_logprobs: Vec<TokenLogprob>,
    refusal_logprobs: Vec<TokenLogprob>,
}

/// Puts a [`ChatStream`] together into the [`ChatCompletion`] it describes, checking that
/// the stream keeps its contract: `n` choices, each finished once and not extended after,
/// tool calls introduced (with id and name) before they are extended, and a final usage.
pub async fn collect(model: &str, n: u32, mut stream: ChatStream) -> Result<ChatCompletion> {
    let broken = |what: String| Error::Inference(format!("model `{model}` {what}"));
    let mut partials: Vec<Partial> = (0..n).map(|_| Partial::default()).collect();
    let mut usage = None;
    let mut timings = None;
    while let Some(event) = stream.next().await {
        if usage.is_some() {
            return Err(broken("sent an event after its usage".into()));
        }
        match event? {
            ChatEvent::Delta { index, delta } => {
                let p = partials
                    .get_mut(index as usize)
                    .ok_or_else(|| broken(format!("sent a delta for choice {index} of {n}")))?;
                if p.finish.is_some() {
                    return Err(broken(format!(
                        "extended choice {index} after finishing it"
                    )));
                }
                apply(p, delta).map_err(|e| broken(format!("in choice {index}: {e}")))?;
            }
            ChatEvent::Finish { index, reason } => {
                let p = partials
                    .get_mut(index as usize)
                    .ok_or_else(|| broken(format!("finished choice {index} of {n}")))?;
                if p.finish.replace(reason).is_some() {
                    return Err(broken(format!("finished choice {index} twice")));
                }
            }
            ChatEvent::Timings(t) => {
                if timings.replace(t).is_some() {
                    return Err(broken("sent its timings twice".into()));
                }
            }
            ChatEvent::Usage(u) => usage = Some(u),
        }
    }
    let usage = usage.ok_or_else(|| broken("ended its stream without a usage".into()))?;
    let choices = partials
        .into_iter()
        .zip(0..)
        .map(|(mut p, index)| {
            let finish_reason = p
                .finish
                .ok_or_else(|| broken(format!("never finished choice {index}")))?;
            p.calls.sort_by_key(|(i, _)| *i);
            p.message.tool_calls = p.calls.into_iter().map(|(_, c)| c).collect();
            Ok(Choice {
                index,
                message: p.message,
                finish_reason,
                content_logprobs: p.content_logprobs,
                refusal_logprobs: p.refusal_logprobs,
            })
        })
        .collect::<Result<_>>()?;
    Ok(ChatCompletion {
        model: model.to_owned(),
        choices,
        usage,
        timings,
    })
}

fn append(into: &mut Option<String>, more: Option<String>) {
    if let Some(more) = more {
        into.get_or_insert_with(String::new).push_str(&more);
    }
}

fn apply(p: &mut Partial, delta: Delta) -> std::result::Result<(), String> {
    append(&mut p.message.content, delta.content);
    append(&mut p.message.refusal, delta.refusal);
    p.content_logprobs.extend(delta.content_logprobs);
    p.refusal_logprobs.extend(delta.refusal_logprobs);
    for d in delta.tool_calls {
        match p.calls.iter_mut().find(|(i, _)| *i == d.index) {
            Some((_, call)) => {
                if d.id.is_some() || d.name.is_some() {
                    return Err(format!("re-introduced tool call {}", d.index));
                }
                call.input.push_str(&d.input);
            }
            None => {
                let (Some(id), Some(name)) = (d.id, d.name) else {
                    return Err(format!(
                        "extended tool call {} before giving its id and name",
                        d.index
                    ));
                };
                p.calls.push((
                    d.index,
                    ToolCall {
                        id,
                        name,
                        kind: d.kind.unwrap_or_default(),
                        input: d.input,
                    },
                ));
            }
        }
    }
    if let Some(a) = delta.audio {
        let audio = p.message.audio.get_or_insert_with(AudioReply::default);
        if let Some(id) = a.id {
            audio.id = id;
        }
        audio.data.extend(a.data);
        audio.transcript.push_str(&a.transcript);
        audio.expires_at = a.expires_at.or(audio.expires_at);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn stream(events: Vec<ChatEvent>) -> ChatStream {
        Box::pin(futures::stream::iter(events.into_iter().map(Ok)))
    }

    fn text(index: u32, s: &str) -> ChatEvent {
        ChatEvent::Delta {
            index,
            delta: Delta {
                content: Some(s.into()),
                ..Delta::default()
            },
        }
    }

    fn call(index: u32, call: u32, id: Option<&str>, name: Option<&str>, input: &str) -> ChatEvent {
        ChatEvent::Delta {
            index,
            delta: Delta {
                tool_calls: vec![ToolCallDelta {
                    index: call,
                    id: id.map(Into::into),
                    name: name.map(Into::into),
                    kind: None,
                    input: input.into(),
                }],
                ..Delta::default()
            },
        }
    }

    fn finish(index: u32, reason: ChatFinishReason) -> ChatEvent {
        ChatEvent::Finish { index, reason }
    }

    fn usage() -> ChatEvent {
        ChatEvent::Usage(ChatUsage {
            prompt_tokens: 3,
            completion_tokens: 4,
            ..ChatUsage::default()
        })
    }

    #[tokio::test]
    async fn test_collect_concatenates_interleaved_choices() {
        let events = vec![
            text(0, "Hel"),
            text(1, "Bon"),
            text(0, "lo"),
            finish(0, ChatFinishReason::Stop),
            text(1, "jour"),
            finish(1, ChatFinishReason::Length),
            usage(),
        ];
        let c = collect("m", 2, stream(events)).await.unwrap();
        assert_eq!(c.choices[0].message.content.as_deref(), Some("Hello"));
        assert_eq!(c.choices[1].message.content.as_deref(), Some("Bonjour"));
        assert_eq!(c.choices[1].finish_reason, ChatFinishReason::Length);
        assert_eq!(c.usage.total_tokens(), 7);
    }

    #[tokio::test]
    async fn test_collect_assembles_tool_calls_in_index_order() {
        let events = vec![
            call(0, 1, Some("b"), Some("get_time"), "{\"tz\":"),
            call(0, 0, Some("a"), Some("get_weather"), "{\"city\":"),
            call(0, 0, None, None, "\"Oslo\"}"),
            call(0, 1, None, None, "\"UTC\"}"),
            finish(0, ChatFinishReason::ToolCalls),
            usage(),
        ];
        let c = collect("m", 1, stream(events)).await.unwrap();
        let calls = &c.choices[0].message.tool_calls;
        assert_eq!(calls.len(), 2);
        assert_eq!(
            (
                calls[0].id.as_str(),
                calls[0].name.as_str(),
                calls[0].input.as_str()
            ),
            ("a", "get_weather", "{\"city\":\"Oslo\"}")
        );
        assert_eq!(calls[1].input, "{\"tz\":\"UTC\"}");
        assert_eq!(calls[0].kind, ToolCallKind::Function);
        assert_eq!(c.choices[0].message.content, None);
    }

    #[tokio::test]
    async fn test_collect_rejects_a_broken_stream() {
        let cases: Vec<(Vec<ChatEvent>, &str)> = vec![
            (vec![text(0, "x"), usage()], "never finished choice 0"),
            (
                vec![text(0, "x"), finish(0, ChatFinishReason::Stop)],
                "without a usage",
            ),
            (
                vec![finish(0, ChatFinishReason::Stop), text(0, "x"), usage()],
                "after finishing",
            ),
            (vec![text(1, "x"), usage()], "choice 1 of 1"),
            (
                vec![call(0, 0, None, None, "{}"), usage()],
                "before giving its id and name",
            ),
            (
                vec![
                    finish(0, ChatFinishReason::Stop),
                    ChatEvent::Timings(crate::Timings::default()),
                    ChatEvent::Timings(crate::Timings::default()),
                    usage(),
                ],
                "timings twice",
            ),
            (
                vec![finish(0, ChatFinishReason::Stop), usage(), usage()],
                "after its usage",
            ),
        ];
        for (events, want) in cases {
            let err = collect("m", 1, stream(events))
                .await
                .unwrap_err()
                .to_string();
            assert!(err.contains(want), "{err:?} should mention {want:?}");
        }
    }
}
