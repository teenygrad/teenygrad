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

//! [`ModelEngine`]: loaded models served as an [`Engine`].

use std::sync::Arc;

use futures::future::BoxFuture;
use futures::{StreamExt, TryStreamExt};

use crate::chat::{
    ChatEvent, ChatFinishReason, ChatModel, ChatRequest, ChatStream, Message, Sampling, validate,
};
use crate::completion::{self, CompletionModel, CompletionRequest, CompletionStream};
use crate::embedding::{self, EmbeddingModel, EmbeddingRequest, Embeddings};
use crate::models::{LoadedModel, Model};
use crate::{
    ChatMessage, Engine, Error, FinishReason, GenerationChunk, GenerationRequest, ModelInfo,
    Prompt, Result, Role, TokenStream, Usage,
};

/// Loaded models, served as one [`Engine`].
///
/// Each request is routed by its `model` to the [`LoadedModel`] with that id, or, for a
/// factory model, to the one a short tag names (`functiongemma3` for
/// `functiongemma3:270m`, case-insensitively), then to the API handle the request needs;
/// a model without that API is refused with an error naming what it does serve.
///
/// [`Engine::chat_stream`] passes the full chat API through, validated against the
/// model's capabilities. [`Engine::generate_stream`] maps the older text-only
/// [`GenerationRequest`] onto chat, for frontends not yet moved to chat requests; raw-text
/// prompts (legacy completions) are not something a chat model takes, and are refused.
#[derive(Clone, Debug)]
pub struct ModelEngine {
    models: Vec<LoadedModel>,
}

impl ModelEngine {
    /// Serves `models`. Their ids must be distinct, and each API handle must report its
    /// model's id.
    pub fn new(models: Vec<LoadedModel>) -> Result<Self> {
        let mut ids: Vec<&str> = models.iter().map(|m| m.info.id.as_str()).collect();
        ids.sort_unstable();
        if let Some(w) = ids.windows(2).find(|w| w[0] == w[1]) {
            return Err(Error::InvalidRequest(format!(
                "two loaded models share the id `{}`; each served model needs its own",
                w[0]
            )));
        }
        for m in &models {
            let handles = [
                ("chat", m.chat.as_ref().map(|c| c.info().id)),
                ("completion", m.completion.as_ref().map(|c| c.info().id)),
                ("embedding", m.embedding.as_ref().map(|e| e.info().id)),
            ];
            for (api, id) in handles {
                if let Some(id) = id.filter(|id| *id != m.info.id) {
                    return Err(Error::InvalidRequest(format!(
                        "loaded model `{}` has a {api} handle reporting the id `{id}`",
                        m.info.id
                    )));
                }
            }
        }
        Ok(Self { models })
    }

    /// The models served.
    pub fn loaded(&self) -> &[LoadedModel] {
        &self.models
    }

    /// The model `requested` names: by id, else by a tag of a factory model. Otherwise
    /// the error lists those served.
    pub fn find(&self, requested: &str) -> Result<&LoadedModel> {
        if let Some(m) = self.models.iter().find(|m| m.info.id == requested) {
            return Ok(m);
        }
        requested
            .parse::<Model>()
            .ok()
            .and_then(|tag| self.models.iter().find(|m| m.model == Some(tag)))
            .ok_or_else(|| Error::ModelNotFound {
                requested: requested.to_owned(),
                available: self.models.iter().map(|m| m.info.id.clone()).collect(),
            })
    }

    /// The chat handle of the model `requested` names.
    fn chat_model(&self, requested: &str) -> Result<(&LoadedModel, &Arc<dyn ChatModel>)> {
        let m = self.find(requested)?;
        match &m.chat {
            Some(chat) => Ok((m, chat)),
            None => Err(lacks(m, "chat completions")),
        }
    }
}

impl ModelEngine {
    /// The completion handle of the model `requested` names.
    fn completion_model(
        &self,
        requested: &str,
    ) -> Result<(&LoadedModel, &Arc<dyn CompletionModel>)> {
        let m = self.find(requested)?;
        match &m.completion {
            Some(c) => Ok((m, c)),
            None => Err(lacks(m, "text completions")),
        }
    }

    /// The embedding handle of the model `requested` names.
    fn embedding_model(&self, requested: &str) -> Result<(&LoadedModel, &Arc<dyn EmbeddingModel>)> {
        let m = self.find(requested)?;
        match &m.embedding {
            Some(e) => Ok((m, e)),
            None => Err(lacks(m, "embeddings")),
        }
    }
}

/// The error for a request to `model` for an API it does not serve.
fn lacks(model: &LoadedModel, api: &'static str) -> Error {
    let serves = model.apis();
    Error::ModelLacksApi {
        model: model.info.id.clone(),
        api,
        serves: if serves.is_empty() {
            "nothing".into()
        } else {
            serves.join(", ")
        },
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

impl Engine for ModelEngine {
    fn name(&self) -> &'static str {
        "models"
    }

    fn models(&self) -> Vec<ModelInfo> {
        self.models.iter().map(|m| m.info.clone()).collect()
    }

    fn model(&self, id: &str) -> Result<ModelInfo> {
        Ok(self.find(id)?.info.clone())
    }

    fn chat_stream(&self, mut request: ChatRequest) -> Result<ChatStream> {
        let (loaded, chat) = self.chat_model(&request.model)?;
        // The model sees its canonical id, whatever tag the request used.
        request.model.clone_from(&loaded.info.id);
        let model = Arc::clone(chat);
        validate(&loaded.info.id, &model.capabilities(), &request)?;
        let started = async move { model.chat(request).await };
        Ok(Box::pin(futures::stream::once(started).try_flatten()))
    }

    fn complete_stream(
        &self,
        mut request: CompletionRequest,
    ) -> BoxFuture<'_, Result<CompletionStream>> {
        Box::pin(async move {
            let (loaded, model) = self.completion_model(&request.model)?;
            request.model.clone_from(&loaded.info.id);
            completion::validate(&loaded.info.id, &model.capabilities(), &request)?;
            model.complete(request).await
        })
    }

    fn embed(&self, mut request: EmbeddingRequest) -> BoxFuture<'_, Result<Embeddings>> {
        Box::pin(async move {
            let (loaded, model) = self.embedding_model(&request.model)?;
            request.model.clone_from(&loaded.info.id);
            embedding::validate(&loaded.info.id, &model.capabilities(), &request)?;
            model.embed(request).await
        })
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
    #[cfg(feature = "functiongemma3")]
    use crate::chat::Capabilities;
    use crate::chat::{EchoChatModel, collect};

    fn engine() -> ModelEngine {
        ModelEngine::new(vec![LoadedModel::from_chat(Arc::new(EchoChatModel))]).unwrap()
    }

    fn hi(model: &str) -> ChatRequest {
        ChatRequest {
            model: model.into(),
            messages: vec![Message::user("hi")],
            ..ChatRequest::default()
        }
    }

    #[tokio::test]
    async fn test_model_engine_serves_text_generation_from_a_chat_model() {
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
    async fn test_model_engine_passes_chat_requests_through() {
        let e = engine();
        let stream = e
            .chat_stream(ChatRequest {
                n: 2,
                ..hi(EchoChatModel::MODEL_ID)
            })
            .unwrap();
        let c = collect(EchoChatModel::MODEL_ID, 2, stream).await.unwrap();
        assert_eq!(c.choices.len(), 2);
        assert_eq!(c.choices[1].message.content.as_deref(), Some("hi"));
    }

    #[test]
    fn test_model_engine_refuses_before_streaming() {
        let e = engine();
        assert!(matches!(
            e.chat_stream(hi("gpt-4o")),
            Err(Error::ModelNotFound { .. })
        ));
        let too_many = e.chat_stream(ChatRequest {
            n: 9,
            ..hi(EchoChatModel::MODEL_ID)
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
    fn test_model_engine_refuses_duplicate_ids() {
        let echo = || LoadedModel::from_chat(Arc::new(EchoChatModel));
        let twice = ModelEngine::new(vec![echo(), echo()]);
        assert!(matches!(twice, Err(Error::InvalidRequest(_))));
    }

    #[tokio::test]
    async fn test_model_engine_serves_embeddings_from_an_embedding_model() {
        use crate::embedding::{EchoEmbeddingModel, EmbeddingInput};
        let e = ModelEngine::new(vec![
            LoadedModel::from_chat(Arc::new(EchoChatModel)),
            LoadedModel::from_embedding(Arc::new(EchoEmbeddingModel)),
        ])
        .unwrap();
        let request = |model: &str, dimensions| EmbeddingRequest {
            model: model.into(),
            inputs: vec![EmbeddingInput::Text("a b".into())],
            dimensions,
            user: None,
        };
        let out = e
            .embed(request(EchoEmbeddingModel::MODEL_ID, Some(4)))
            .await
            .unwrap();
        assert_eq!(out.vectors[0].len(), 4);
        let refused = e
            .embed(request(EchoEmbeddingModel::MODEL_ID, Some(99)))
            .await;
        assert!(matches!(refused, Err(Error::InvalidParameter { .. })));
        let chat_only = e.embed(request(EchoChatModel::MODEL_ID, None)).await;
        assert!(matches!(chat_only, Err(Error::ModelLacksApi { .. })));
    }

    #[test]
    fn test_model_engine_refuses_a_handle_with_another_id() {
        let mut info = EchoChatModel.info();
        info.id = "not-echo".into();
        let mismatched = LoadedModel::new(info).with_chat(Arc::new(EchoChatModel));
        let err = ModelEngine::new(vec![mismatched]).unwrap_err().to_string();
        assert!(
            err.contains("not-echo") && err.contains(EchoChatModel::MODEL_ID),
            "{err}"
        );
    }

    #[test]
    fn test_model_engine_refuses_an_api_the_model_lacks() {
        let mut info = EchoChatModel.info();
        info.id = "weights-only".into();
        let e = ModelEngine::new(vec![LoadedModel::new(info)]).unwrap();
        let Err(err) = e.chat_stream(hi("weights-only")) else {
            panic!("a model without a chat handle answered a chat request");
        };
        assert!(matches!(err, Error::ModelLacksApi { .. }), "{err}");
        let msg = err.to_string();
        assert!(
            msg.contains("chat completions") && msg.contains("nothing"),
            "{msg}"
        );
        // The model is still listed, and looked up.
        assert_eq!(e.models().len(), 1);
        assert!(e.model("weights-only").is_ok());
    }

    /// A chat model posing as a factory model, to test tag resolution without weights.
    #[cfg(feature = "functiongemma3")]
    struct Posing(Model);

    #[cfg(feature = "functiongemma3")]
    impl ChatModel for Posing {
        fn info(&self) -> ModelInfo {
            ModelInfo {
                id: self.0.tag().into(),
                ..EchoChatModel.info()
            }
        }
        fn capabilities(&self) -> Capabilities {
            EchoChatModel.capabilities()
        }
        fn chat(&self, request: ChatRequest) -> futures::future::BoxFuture<'_, Result<ChatStream>> {
            assert_eq!(
                request.model,
                self.0.tag(),
                "the model sees its canonical id"
            );
            EchoChatModel.chat(request)
        }
    }

    #[cfg(feature = "functiongemma3")]
    #[tokio::test]
    async fn test_model_engine_resolves_short_tags_to_the_canonical_id() {
        let m = Model::FunctionGemma3;
        let loaded = LoadedModel::from_chat(Arc::new(Posing(m))).with_model(m);
        let e = ModelEngine::new(vec![loaded]).unwrap();
        for tag in [
            m.tag(),
            m.spec().family,
            "FunctionGemma3",
            "functiongemma3:270M",
        ] {
            assert_eq!(e.model(tag).unwrap().id, m.tag(), "{tag}");
            let stream = e.chat_stream(hi(tag)).unwrap();
            let c = collect(m.tag(), 1, stream).await.unwrap();
            assert_eq!(c.choices[0].message.content.as_deref(), Some("hi"));
        }
        assert!(matches!(
            e.model("functiongemma3:2b"),
            Err(Error::ModelNotFound { .. })
        ));
    }
}
