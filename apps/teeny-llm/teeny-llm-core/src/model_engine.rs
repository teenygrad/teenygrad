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

use futures::TryStreamExt;
use futures::future::BoxFuture;

use crate::chat::{ChatModel, ChatRequest, ChatStream, validate};
use crate::completion::{self, CompletionModel, CompletionRequest, CompletionStream};
use crate::embedding::{self, EmbeddingModel, EmbeddingRequest, Embeddings};
use crate::models::{LoadedModel, Model};
use crate::transcription::{self, TranscriptionModel, TranscriptionRequest, TranscriptionStream};
use crate::{Engine, Error, ModelInfo, Result};

/// Loaded models, served as one [`Engine`].
///
/// Each request is routed by its `model` to the [`LoadedModel`] with that id, or, for a
/// factory model, to the one a short tag names (`functiongemma3` for
/// `functiongemma3:270m`, case-insensitively), then to the API handle the request needs;
/// a model without that API is refused with an error naming what it does serve.
///
/// Each API is validated against the model's capabilities for it before the model sees
/// the request, and the model is handed its canonical id.
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
                (
                    "transcription",
                    m.transcription.as_ref().map(|t| t.info().id),
                ),
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

    /// The transcription handle of the model `requested` names.
    fn transcription_model(
        &self,
        requested: &str,
    ) -> Result<(&LoadedModel, &Arc<dyn TranscriptionModel>)> {
        let m = self.find(requested)?;
        match &m.transcription {
            Some(t) => Ok((m, t)),
            None => Err(lacks(m, "transcriptions")),
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

    fn transcribe(
        &self,
        mut request: TranscriptionRequest,
    ) -> BoxFuture<'_, Result<TranscriptionStream>> {
        Box::pin(async move {
            let (loaded, model) = self.transcription_model(&request.model)?;
            request.model.clone_from(&loaded.info.id);
            let caps = model.capabilities();
            transcription::validate(&loaded.info.id, &caps, &request)?;
            let container = request.container.ok_or_else(|| Error::InvalidParameter {
                parameter: "file".into(),
                detail: "no audio format".into(),
            })?;
            let audio =
                crate::audio::decode(container, &request.audio)?.resampled_mono(caps.sample_rate);
            transcription::check_duration(&caps, &audio)?;
            // The model has the samples; the upload is not needed twice.
            request.audio = Vec::new();
            model.transcribe(request, audio).await
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[cfg(feature = "functiongemma3")]
    use crate::chat::Capabilities;
    use crate::chat::{EchoChatModel, Message, collect};

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
