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

//! [`ChatEngine`]: chat models served as an [`Engine`].

use std::sync::Arc;

use futures::{StreamExt, TryStreamExt};

use super::capabilities::validate;
use super::model::ChatModel;
use super::request::{ChatRequest, Message, Sampling};
use super::response::{ChatEvent, ChatFinishReason, ChatStream};
use crate::{
    ChatMessage, Engine, Error, FinishReason, GenerationChunk, GenerationRequest, ModelInfo,
    Prompt, Result, Role, TokenStream, Usage,
};

/// One or more [`ChatModel`]s, served as an [`Engine`].
///
/// [`Engine::chat_stream`] passes the full chat API through, validated against the chosen
/// model's capabilities. [`Engine::generate_stream`] maps the older text-only
/// [`GenerationRequest`] onto it, for frontends not yet moved to chat requests; raw-text
/// prompts (legacy completions) are not something a chat model takes, and are refused.
#[derive(Clone)]
pub struct ChatEngine {
    models: Vec<Arc<dyn ChatModel>>,
}

impl ChatEngine {
    /// Serves `models`. Their ids must be distinct.
    pub fn new(models: Vec<Arc<dyn ChatModel>>) -> Result<Self> {
        let mut ids: Vec<String> = models.iter().map(|m| m.info().id).collect();
        ids.sort();
        if let Some(w) = ids.windows(2).find(|w| w[0] == w[1]) {
            return Err(Error::InvalidRequest(format!(
                "two chat models share the id `{}`; each served model needs its own",
                w[0]
            )));
        }
        Ok(Self { models })
    }

    /// The model `id`, or the error listing those served.
    fn find(&self, id: &str) -> Result<&Arc<dyn ChatModel>> {
        self.models
            .iter()
            .find(|m| m.info().id == id)
            .ok_or_else(|| Error::ModelNotFound {
                requested: id.to_owned(),
                available: self.models.iter().map(|m| m.info().id).collect(),
            })
    }
}

fn to_message(m: ChatMessage) -> Result<Message> {
    Ok(match m.role {
        Role::Developer => Message::Developer {
            content: m.content,
            name: None,
        },
        Role::System => Message::system(m.content),
        Role::User => Message::user(m.content),
        Role::Assistant => Message::assistant(m.content),
        Role::Tool => {
            return Err(Error::InvalidRequest(
                "a tool message needs the id of the call it answers, which a text-only \
                 GenerationRequest cannot carry; send it through Engine::chat_stream"
                    .to_owned(),
            ));
        }
    })
}

fn to_finish(reason: ChatFinishReason) -> FinishReason {
    match reason {
        ChatFinishReason::Length => FinishReason::Length,
        ChatFinishReason::Stop | ChatFinishReason::ToolCalls | ChatFinishReason::ContentFilter => {
            FinishReason::Stop
        }
    }
}

impl Engine for ChatEngine {
    fn name(&self) -> &'static str {
        "chat"
    }

    fn models(&self) -> Vec<ModelInfo> {
        self.models.iter().map(|m| m.info()).collect()
    }

    fn chat_stream(&self, request: ChatRequest) -> Result<ChatStream> {
        let model = Arc::clone(self.find(&request.model)?);
        validate(&model.info().id, &model.capabilities(), &request)?;
        let started = async move { model.chat(request).await };
        Ok(Box::pin(futures::stream::once(started).try_flatten()))
    }

    fn generate_stream(&self, request: GenerationRequest) -> Result<TokenStream> {
        let messages = match request.prompt {
            Prompt::Chat(messages) => messages
                .into_iter()
                .map(to_message)
                .collect::<Result<Vec<_>>>()?,
            Prompt::Text(_) => {
                return Err(Error::Unsupported {
                    engine: self.name(),
                    operation: "text completions (chat models take a conversation)",
                });
            }
        };
        let s = request.sampling;
        let chat = ChatRequest {
            model: request.model,
            messages,
            max_completion_tokens: s.max_tokens,
            sampling: Sampling {
                temperature: s.temperature,
                top_p: s.top_p,
                presence_penalty: s.presence_penalty,
                frequency_penalty: s.frequency_penalty,
                seed: s.seed,
                stop: s.stop,
                ..Sampling::default()
            },
            include_usage: true,
            ..ChatRequest::default()
        };
        let events = self.chat_stream(chat)?;
        // Choice 0's text, then one closing chunk with the finish reason and usage.
        let chunks = events
            .scan(None, |finish: &mut Option<FinishReason>, event| {
                let out = match event {
                    Err(e) => Some(Err(e)),
                    Ok(ChatEvent::Delta { index: 0, delta }) => {
                        delta.content.filter(|t| !t.is_empty()).map(|text| {
                            Ok(GenerationChunk {
                                text,
                                finish_reason: None,
                                usage: None,
                            })
                        })
                    }
                    Ok(ChatEvent::Delta { .. }) => None,
                    Ok(ChatEvent::Finish { index, reason }) => {
                        if index == 0 {
                            *finish = Some(to_finish(reason));
                        }
                        None
                    }
                    Ok(ChatEvent::Usage(u)) => Some(Ok(GenerationChunk {
                        text: String::new(),
                        finish_reason: Some(finish.unwrap_or(FinishReason::Stop)),
                        usage: Some(Usage {
                            prompt_tokens: u.prompt_tokens,
                            completion_tokens: u.completion_tokens,
                        }),
                    })),
                };
                futures::future::ready(Some(out))
            })
            .filter_map(futures::future::ready);
        Ok(Box::pin(chunks))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::SamplingParams;
    use crate::chat::EchoChatModel;
    use crate::chat::response::collect;

    fn engine() -> ChatEngine {
        ChatEngine::new(vec![Arc::new(EchoChatModel)]).unwrap()
    }

    #[tokio::test]
    async fn test_chat_engine_serves_text_generation_from_a_chat_model() {
        let e: &dyn Engine = &engine();
        let out = e
            .generate(GenerationRequest {
                model: EchoChatModel::MODEL_ID.into(),
                prompt: Prompt::Chat(vec![ChatMessage::user("hello there")]),
                sampling: SamplingParams {
                    max_tokens: Some(1),
                    ..SamplingParams::default()
                },
            })
            .await
            .unwrap();
        assert_eq!(out.text, "hello");
        assert_eq!(out.finish_reason, FinishReason::Length);
        assert_eq!(out.usage.completion_tokens, 1);
    }

    #[tokio::test]
    async fn test_chat_engine_passes_chat_requests_through() {
        let e = engine();
        let stream = e
            .chat_stream(ChatRequest {
                model: EchoChatModel::MODEL_ID.into(),
                messages: vec![Message::user("hi")],
                n: 2,
                ..ChatRequest::default()
            })
            .unwrap();
        let c = collect(EchoChatModel::MODEL_ID, 2, stream).await.unwrap();
        assert_eq!(c.choices.len(), 2);
        assert_eq!(c.choices[1].message.content.as_deref(), Some("hi"));
    }

    #[test]
    fn test_chat_engine_refuses_before_streaming() {
        let e = engine();
        let unknown = e.chat_stream(ChatRequest {
            model: "gpt-4o".into(),
            messages: vec![Message::user("hi")],
            ..ChatRequest::default()
        });
        assert!(matches!(unknown, Err(Error::ModelNotFound { .. })));
        let too_many = e.chat_stream(ChatRequest {
            model: EchoChatModel::MODEL_ID.into(),
            messages: vec![Message::user("hi")],
            n: 9,
            ..ChatRequest::default()
        });
        assert!(matches!(too_many, Err(Error::UnsupportedParameter { .. })));
        let text = e.generate_stream(GenerationRequest {
            model: EchoChatModel::MODEL_ID.into(),
            prompt: Prompt::Text("x".into()),
            sampling: SamplingParams::default(),
        });
        assert!(matches!(text, Err(Error::Unsupported { .. })));
    }

    #[test]
    fn test_chat_engine_refuses_duplicate_ids() {
        let twice = ChatEngine::new(vec![Arc::new(EchoChatModel), Arc::new(EchoChatModel)]);
        assert!(matches!(twice, Err(Error::InvalidRequest(_))));
    }
}
