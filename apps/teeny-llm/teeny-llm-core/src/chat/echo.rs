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

//! [`EchoChatModel`], the reference [`ChatModel`].

use futures::future::BoxFuture;
use serde_json::json;

use super::capabilities::{Capabilities, ToolSupport};
use super::model::ChatModel;
use super::request::{AllowedMode, ChatRequest, ContentPart, Message, ToolCallKind, ToolChoice};
use super::response::{ChatEvent, ChatFinishReason, ChatStream, ChatUsage, Delta, ToolCallDelta};
use crate::{Error, ModelInfo, Result};

/// A chat model that answers with the conversation's last user message, word by word.
///
/// It runs no network, so it is the reference implementation of [`ChatModel`] and a
/// fixture for frontends: it honours `n`, stop strings and the token limit, and when a
/// request forces a tool call (`tool_choice` `required`, a named tool, or required
/// `allowed_tools`) it calls the tool with `{"text": <the message>}` instead, streamed in
/// two pieces. Tokens are whitespace-separated words.
#[derive(Debug, Clone, Default)]
pub struct EchoChatModel;

impl EchoChatModel {
    pub const MODEL_ID: &'static str = "teeny-echo";
}

fn words(text: &str) -> u32 {
    u32::try_from(text.split_whitespace().count()).unwrap_or(u32::MAX)
}

/// The text of a message, for counting prompt tokens.
fn message_text(m: &Message) -> String {
    match m {
        Message::Developer { content, .. } | Message::System { content, .. } => content.clone(),
        Message::Tool { content, .. } => content.clone(),
        Message::User { content, .. } => content
            .iter()
            .filter_map(|p| match p {
                ContentPart::Text(t) => Some(t.as_str()),
                _ => None,
            })
            .collect::<Vec<_>>()
            .join(" "),
        Message::Assistant(a) => {
            let mut t = a.content.clone().unwrap_or_default();
            for c in &a.tool_calls {
                t.push(' ');
                t.push_str(&c.input);
            }
            t
        }
    }
}

/// The tool a request forces a call to, if it forces one.
fn forced_tool(r: &ChatRequest) -> Option<&str> {
    match &r.tool_choice {
        ToolChoice::Named(name) => Some(name),
        ToolChoice::Required => r.tools.first().map(|t| t.name.as_str()),
        ToolChoice::Allowed {
            mode: AllowedMode::Required,
            tools,
        } => tools.first().map(String::as_str),
        _ => None,
    }
}

impl ChatModel for EchoChatModel {
    fn info(&self) -> ModelInfo {
        ModelInfo {
            id: Self::MODEL_ID.to_owned(),
            created: 0,
            owned_by: "teenygrad".to_owned(),
        }
    }

    fn capabilities(&self) -> Capabilities {
        Capabilities {
            context_length: 4096,
            tools: ToolSupport {
                functions: true,
                custom: true,
                parallel: false,
                forced: true,
                strict: false,
            },
            stop_strings: true,
            max_choices: 4,
            ..Capabilities::default()
        }
    }

    fn chat(&self, request: ChatRequest) -> BoxFuture<'_, Result<ChatStream>> {
        Box::pin(async move {
            let reply = request
                .messages
                .iter()
                .rev()
                .find(|m| matches!(m, Message::User { .. }))
                .map(message_text)
                .ok_or_else(|| {
                    Error::InvalidRequest(format!(
                        "the conversation has {} message(s) but none from the user, so there \
                         is nothing to echo",
                        request.messages.len()
                    ))
                })?;
            let prompt_tokens: u32 = request
                .messages
                .iter()
                .map(|m| words(&message_text(m)))
                .sum();

            let mut events = Vec::new();
            let mut completion_tokens = 0u32;
            for index in 0..request.n {
                if let Some(name) = forced_tool(&request) {
                    let kind = request.tools.iter().find(|t| t.name == name).map_or(
                        ToolCallKind::Function,
                        |t| match t.kind {
                            super::request::ToolKind::Function { .. } => ToolCallKind::Function,
                            super::request::ToolKind::Custom { .. } => ToolCallKind::Custom,
                        },
                    );
                    let input = match kind {
                        ToolCallKind::Function => json!({ "text": reply }).to_string(),
                        ToolCallKind::Custom => reply.clone(),
                    };
                    // Two pieces, split on a character boundary, to exercise reassembly.
                    let mid = (0..=input.len() / 2)
                        .rev()
                        .find(|&i| input.is_char_boundary(i))
                        .unwrap_or(0);
                    let (head, tail) = input.split_at(mid);
                    let head = head.to_owned();
                    let tail = tail.to_owned();
                    for (i, piece) in [head, tail].into_iter().enumerate() {
                        events.push(ChatEvent::Delta {
                            index,
                            delta: Delta {
                                tool_calls: vec![ToolCallDelta {
                                    index: 0,
                                    id: (i == 0).then(|| format!("call_echo_{index}")),
                                    name: (i == 0).then(|| name.to_owned()),
                                    kind: (i == 0).then_some(kind),
                                    input: piece,
                                }],
                                ..Delta::default()
                            },
                        });
                    }
                    completion_tokens = completion_tokens.saturating_add(words(&input));
                    events.push(ChatEvent::Finish {
                        index,
                        reason: ChatFinishReason::ToolCalls,
                    });
                    continue;
                }

                // Stop before the first stop string, as if the model had produced it.
                let cut = request
                    .sampling
                    .stop
                    .iter()
                    .filter_map(|s| reply.find(s.as_str()))
                    .min()
                    .unwrap_or(reply.len());
                let text = &reply[..cut];
                let limit = request
                    .max_completion_tokens
                    .map_or(usize::MAX, |n| n as usize);
                let all: Vec<&str> = text.split_whitespace().collect();
                let (emitted, reason) = if all.len() > limit {
                    (&all[..limit], ChatFinishReason::Length)
                } else {
                    (&all[..], ChatFinishReason::Stop)
                };
                for (i, w) in emitted.iter().enumerate() {
                    let content = if i == 0 {
                        (*w).to_owned()
                    } else {
                        format!(" {w}")
                    };
                    events.push(ChatEvent::Delta {
                        index,
                        delta: Delta {
                            content: Some(content),
                            ..Delta::default()
                        },
                    });
                }
                completion_tokens = completion_tokens
                    .saturating_add(u32::try_from(emitted.len()).unwrap_or(u32::MAX));
                events.push(ChatEvent::Finish { index, reason });
            }
            events.push(ChatEvent::Usage(ChatUsage {
                prompt_tokens,
                completion_tokens,
                ..ChatUsage::default()
            }));
            let stream: ChatStream = Box::pin(futures::stream::iter(events.into_iter().map(Ok)));
            Ok(stream)
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chat::request::{Sampling, Tool, ToolKind};

    fn model() -> Box<dyn ChatModel> {
        Box::new(EchoChatModel)
    }

    fn request(text: &str) -> ChatRequest {
        ChatRequest {
            model: EchoChatModel::MODEL_ID.into(),
            messages: vec![Message::system("be brief"), Message::user(text)],
            ..ChatRequest::default()
        }
    }

    #[tokio::test]
    async fn test_echo_answers_with_the_last_user_message() {
        let c = model().complete(request("hello there")).await.unwrap();
        assert_eq!(c.choices.len(), 1);
        assert_eq!(c.choices[0].message.content.as_deref(), Some("hello there"));
        assert_eq!(c.choices[0].finish_reason, ChatFinishReason::Stop);
        assert_eq!(c.usage.prompt_tokens, 4);
        assert_eq!(c.usage.completion_tokens, 2);
    }

    #[tokio::test]
    async fn test_echo_honours_n_stop_and_the_token_limit() {
        let c = model()
            .complete(ChatRequest {
                n: 3,
                sampling: Sampling {
                    stop: vec![" STOP".into()],
                    ..Sampling::default()
                },
                ..request("one two STOP three")
            })
            .await
            .unwrap();
        assert_eq!(c.choices.len(), 3);
        for ch in &c.choices {
            assert_eq!(ch.message.content.as_deref(), Some("one two"));
        }
        let c = model()
            .complete(ChatRequest {
                max_completion_tokens: Some(2),
                ..request("a b c d")
            })
            .await
            .unwrap();
        assert_eq!(c.choices[0].message.content.as_deref(), Some("a b"));
        assert_eq!(c.choices[0].finish_reason, ChatFinishReason::Length);
    }

    #[tokio::test]
    async fn test_echo_makes_a_forced_tool_call() {
        let c = model()
            .complete(ChatRequest {
                tools: vec![Tool {
                    name: "shout".into(),
                    description: None,
                    kind: ToolKind::Function {
                        parameters: None,
                        strict: false,
                    },
                }],
                tool_choice: ToolChoice::Required,
                ..request("hi")
            })
            .await
            .unwrap();
        let ch = &c.choices[0];
        assert_eq!(ch.finish_reason, ChatFinishReason::ToolCalls);
        assert_eq!(ch.message.content, None);
        assert_eq!(ch.message.tool_calls.len(), 1);
        let call = &ch.message.tool_calls[0];
        assert_eq!(
            (call.id.as_str(), call.name.as_str()),
            ("call_echo_0", "shout")
        );
        assert_eq!(call.input, r#"{"text":"hi"}"#);
    }

    #[tokio::test]
    async fn test_echo_refuses_what_it_cannot_do_before_answering() {
        let err = model()
            .complete(ChatRequest {
                sampling: Sampling {
                    temperature: Some(0.8),
                    ..Sampling::default()
                },
                ..request("hi")
            })
            .await
            .unwrap_err();
        assert!(
            matches!(&err, Error::UnsupportedParameter { parameter, .. } if parameter == "temperature"),
            "{err}"
        );
    }
}
