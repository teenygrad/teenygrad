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

//! [`ModelEngine`]: [`ServedModel`]s, from any domain crate, served as one [`Engine`].

use std::fmt;
#[cfg(any(feature = "llm", feature = "audio", feature = "image"))]
use std::sync::Arc;
use std::sync::{PoisonError, RwLock, RwLockReadGuard, RwLockWriteGuard};

#[cfg(feature = "llm")]
use futures::TryStreamExt;
#[cfg(any(feature = "llm", feature = "audio", feature = "image"))]
use futures::future::BoxFuture;

use crate::{Engine, Error, ModelInfo, Result};

#[cfg(feature = "audio")]
use teeny_audio::{
    speech::{self, AudioStream, SpeechModel, SpeechRequest},
    transcription::{self, TranscriptionModel, TranscriptionRequest, TranscriptionStream},
};
#[cfg(feature = "image")]
use teeny_image::{self as images, EncodedImageStream, ImageModel, ImageRequest};
#[cfg(feature = "llm")]
use teeny_llm::{
    chat::{self, ChatModel, ChatRequest, ChatStream},
    completion::{self, CompletionModel, CompletionRequest, CompletionStream},
    embedding::{self, EmbeddingModel, EmbeddingRequest, Embeddings},
    models::{LoadedModel, Model},
    moderation::{self, Moderation, ModerationModel, ModerationRequest},
};

/// A model being served: what describes it, and the APIs it implements, each a handle
/// into the domain crate it comes from.
///
/// One model can serve several APIs (chat and completions, say) from one set of weights,
/// so each API is an optional handle rather than the model being one trait. Built from a
/// teeny-llm `LoadedModel` with `From`, or by hand with [`ServedModel::new`] and the
/// `with_` methods.
#[derive(Clone)]
#[non_exhaustive]
pub struct ServedModel {
    /// Its id and provenance, as `/v1/models` lists it.
    pub info: ModelInfo,
    /// The teeny-llm factory model it is, so short tags resolve to it.
    #[cfg(feature = "llm")]
    pub model: Option<Model>,
    /// Chat completions.
    #[cfg(feature = "llm")]
    pub chat: Option<Arc<dyn ChatModel>>,
    /// Text completions.
    #[cfg(feature = "llm")]
    pub completion: Option<Arc<dyn CompletionModel>>,
    /// Embeddings.
    #[cfg(feature = "llm")]
    pub embedding: Option<Arc<dyn EmbeddingModel>>,
    /// Content classification.
    #[cfg(feature = "llm")]
    pub moderation: Option<Arc<dyn ModerationModel>>,
    /// Speech to text.
    #[cfg(feature = "audio")]
    pub transcription: Option<Arc<dyn TranscriptionModel>>,
    /// Text to speech.
    #[cfg(feature = "audio")]
    pub speech: Option<Arc<dyn SpeechModel>>,
    /// Image generation.
    #[cfg(feature = "image")]
    pub images: Option<Arc<dyn ImageModel>>,
}

impl fmt::Debug for ServedModel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ServedModel")
            .field("info", &self.info)
            .field("apis", &self.apis())
            .finish()
    }
}

impl ServedModel {
    /// A model described by `info` that serves no API yet.
    pub fn new(info: ModelInfo) -> Self {
        Self {
            info,
            #[cfg(feature = "llm")]
            model: None,
            #[cfg(feature = "llm")]
            chat: None,
            #[cfg(feature = "llm")]
            completion: None,
            #[cfg(feature = "llm")]
            embedding: None,
            #[cfg(feature = "llm")]
            moderation: None,
            #[cfg(feature = "audio")]
            transcription: None,
            #[cfg(feature = "audio")]
            speech: None,
            #[cfg(feature = "image")]
            images: None,
        }
    }

    /// Serves chat completions with `chat`.
    #[cfg(feature = "llm")]
    pub fn with_chat(mut self, chat: Arc<dyn ChatModel>) -> Self {
        self.chat = Some(chat);
        self
    }

    /// Serves text completions with `completion`.
    #[cfg(feature = "llm")]
    pub fn with_completion(mut self, completion: Arc<dyn CompletionModel>) -> Self {
        self.completion = Some(completion);
        self
    }

    /// Serves embeddings with `embedding`.
    #[cfg(feature = "llm")]
    pub fn with_embedding(mut self, embedding: Arc<dyn EmbeddingModel>) -> Self {
        self.embedding = Some(embedding);
        self
    }

    /// Serves moderations with `moderation`.
    #[cfg(feature = "llm")]
    pub fn with_moderation(mut self, moderation: Arc<dyn ModerationModel>) -> Self {
        self.moderation = Some(moderation);
        self
    }

    /// A transcription model, described by its own [`TranscriptionModel::info`].
    #[cfg(feature = "audio")]
    pub fn from_transcription(transcription: Arc<dyn TranscriptionModel>) -> Self {
        Self::new(transcription.info().into()).with_transcription(transcription)
    }

    /// Serves transcriptions with `transcription`.
    #[cfg(feature = "audio")]
    pub fn with_transcription(mut self, transcription: Arc<dyn TranscriptionModel>) -> Self {
        self.transcription = Some(transcription);
        self
    }

    /// A speech model, described by its own [`SpeechModel::info`].
    #[cfg(feature = "audio")]
    pub fn from_speech(speech: Arc<dyn SpeechModel>) -> Self {
        Self::new(speech.info().into()).with_speech(speech)
    }

    /// Serves speech with `speech`.
    #[cfg(feature = "audio")]
    pub fn with_speech(mut self, speech: Arc<dyn SpeechModel>) -> Self {
        self.speech = Some(speech);
        self
    }

    /// An image model, described by its own [`ImageModel::info`].
    #[cfg(feature = "image")]
    pub fn from_images(images: Arc<dyn ImageModel>) -> Self {
        Self::new(images.info().into()).with_images(images)
    }

    /// Serves images with `images`.
    #[cfg(feature = "image")]
    pub fn with_images(mut self, images: Arc<dyn ImageModel>) -> Self {
        self.images = Some(images);
        self
    }

    /// The ids its API handles report, by API, for checking they agree with
    /// [`ServedModel::info`].
    fn handle_ids(&self) -> Vec<(&'static str, String)> {
        #[allow(unused_mut)] // which domains extend it depends on the features
        let mut ids: Vec<(&'static str, Option<String>)> = Vec::new();
        #[cfg(feature = "llm")]
        ids.extend([
            ("chat", self.chat.as_ref().map(|h| h.info().id)),
            ("completion", self.completion.as_ref().map(|h| h.info().id)),
            ("embedding", self.embedding.as_ref().map(|h| h.info().id)),
            ("moderation", self.moderation.as_ref().map(|h| h.info().id)),
        ]);
        #[cfg(feature = "audio")]
        ids.extend([
            (
                "transcription",
                self.transcription.as_ref().map(|h| h.info().id),
            ),
            ("speech", self.speech.as_ref().map(|h| h.info().id)),
        ]);
        #[cfg(feature = "image")]
        ids.extend([("images", self.images.as_ref().map(|h| h.info().id))]);
        ids.into_iter()
            .filter_map(|(api, id)| id.map(|id| (api, id)))
            .collect()
    }

    /// The APIs it serves, by name, for messages.
    pub fn apis(&self) -> Vec<&'static str> {
        #[allow(unused_mut)] // which domains extend it depends on the features
        let mut apis: Vec<(&'static str, bool)> = Vec::new();
        #[cfg(feature = "llm")]
        apis.extend([
            ("chat completions", self.chat.is_some()),
            ("text completions", self.completion.is_some()),
            ("embeddings", self.embedding.is_some()),
        ]);
        #[cfg(feature = "audio")]
        apis.extend([
            ("transcriptions", self.transcription.is_some()),
            ("speech", self.speech.is_some()),
        ]);
        #[cfg(feature = "llm")]
        apis.extend([("moderations", self.moderation.is_some())]);
        #[cfg(feature = "image")]
        apis.extend([("images", self.images.is_some())]);
        apis.into_iter()
            .filter_map(|(api, served)| served.then_some(api))
            .collect()
    }

    /// The error for a request to it for an API it does not serve.
    #[cfg(any(feature = "llm", feature = "audio", feature = "image"))]
    fn lacks(&self, api: &'static str) -> Error {
        let serves = self.apis();
        Error::ModelLacksApi {
            model: self.info.id.clone(),
            api,
            serves: if serves.is_empty() {
                "nothing".into()
            } else {
                serves.join(", ")
            },
        }
    }
}

/// A teeny-llm model, serving the LLM APIs it implements.
#[cfg(feature = "llm")]
impl From<LoadedModel> for ServedModel {
    fn from(m: LoadedModel) -> Self {
        Self {
            model: m.model,
            chat: m.chat,
            completion: m.completion,
            embedding: m.embedding,
            moderation: m.moderation,
            ..Self::new(m.info.into())
        }
    }
}

/// Served models, as one [`Engine`].
///
/// Each request is routed by its `model` to the [`ServedModel`] with that id, or, for a
/// teeny-llm factory model, to the one a short tag names (`functiongemma3` for
/// `functiongemma3:270m`, case-insensitively), then to the API handle the request needs;
/// a model without that API is refused with an error naming what it does serve.
///
/// Each API is validated against the model's capabilities for it before the model sees
/// the request, and the model is handed its canonical id.
///
/// Models can be added while it serves ([`ModelEngine::add`]), so a server can listen
/// before its models finish loading; requests in flight keep the handles they took.
#[derive(Debug, Default)]
pub struct ModelEngine {
    models: RwLock<Vec<ServedModel>>,
}

/// Checks that each of `model`'s API handles reports its id.
fn check_handles(model: &ServedModel) -> Result<()> {
    for (api, id) in model.handle_ids() {
        if id != model.info.id {
            return Err(Error::InvalidRequest(format!(
                "served model `{}` has a {api} handle reporting the id `{id}`",
                model.info.id
            )));
        }
    }
    Ok(())
}

impl ModelEngine {
    /// Serves `models`. Their ids must be distinct, and each API handle must report its
    /// model's id.
    pub fn new(models: impl IntoIterator<Item = impl Into<ServedModel>>) -> Result<Self> {
        let engine = Self::default();
        for m in models {
            engine.add(m)?;
        }
        Ok(engine)
    }

    /// Starts serving `model`, as [`ModelEngine::new`] would: its id must be new, and each
    /// API handle must report it.
    pub fn add(&self, model: impl Into<ServedModel>) -> Result<()> {
        let model = model.into();
        check_handles(&model)?;
        let mut models = self.write();
        if models.iter().any(|m| m.info.id == model.info.id) {
            return Err(Error::InvalidRequest(format!(
                "two served models share the id `{}`; each needs its own",
                model.info.id
            )));
        }
        models.push(model);
        Ok(())
    }

    /// The models served, as they are now.
    pub fn served(&self) -> Vec<ServedModel> {
        self.read().clone()
    }

    fn read(&self) -> RwLockReadGuard<'_, Vec<ServedModel>> {
        // A panic while holding the lock cannot leave the list half-changed (pushes are
        // atomic), so a poisoned lock is still sound to use.
        self.models.read().unwrap_or_else(PoisonError::into_inner)
    }

    fn write(&self) -> RwLockWriteGuard<'_, Vec<ServedModel>> {
        self.models.write().unwrap_or_else(PoisonError::into_inner)
    }

    /// The model `requested` names: by id, else by a tag of a factory model. Otherwise
    /// the error lists those served.
    pub fn find(&self, requested: &str) -> Result<ServedModel> {
        let models = self.read();
        if let Some(m) = models.iter().find(|m| m.info.id == requested) {
            return Ok(m.clone());
        }
        #[cfg(feature = "llm")]
        if let Some(m) = requested
            .parse::<Model>()
            .ok()
            .and_then(|tag| models.iter().find(|m| m.model == Some(tag)))
        {
            return Ok(m.clone());
        }
        Err(Error::ModelNotFound {
            requested: requested.to_owned(),
            available: models.iter().map(|m| m.info.id.clone()).collect(),
        })
    }

    /// The model `requested` names and its handle for an API, picked by `handle`.
    #[cfg(any(feature = "llm", feature = "audio", feature = "image"))]
    fn handle<H: ?Sized>(
        &self,
        requested: &str,
        api: &'static str,
        handle: impl Fn(&ServedModel) -> Option<&Arc<H>>,
    ) -> Result<(ServedModel, Arc<H>)> {
        let m = self.find(requested)?;
        let h = handle(&m).cloned().ok_or_else(|| m.lacks(api))?;
        Ok((m, h))
    }

    /// The moderation handle of the model `requested` names, or, with none named, of the
    /// first model that classifies.
    #[cfg(feature = "llm")]
    fn moderation_model(
        &self,
        requested: Option<&str>,
    ) -> Result<(ServedModel, Arc<dyn ModerationModel>)> {
        let Some(requested) = requested else {
            return self
                .read()
                .iter()
                .find_map(|m| m.moderation.clone().map(|h| (m.clone(), h)))
                .ok_or_else(|| Error::InvalidParameter {
                    parameter: "model".into(),
                    detail: "no served model classifies content, so one cannot be chosen".into(),
                });
        };
        self.handle(requested, "moderations", |m| m.moderation.as_ref())
    }
}

impl Engine for ModelEngine {
    fn name(&self) -> &'static str {
        "models"
    }

    fn models(&self) -> Vec<ModelInfo> {
        self.read().iter().map(|m| m.info.clone()).collect()
    }

    fn model(&self, id: &str) -> Result<ModelInfo> {
        Ok(self.find(id)?.info.clone())
    }

    #[cfg(feature = "llm")]
    fn chat_stream(&self, mut request: ChatRequest) -> Result<ChatStream> {
        let (served, model) =
            self.handle(&request.model, "chat completions", |m| m.chat.as_ref())?;
        // The model sees its canonical id, whatever tag the request used.
        request.model.clone_from(&served.info.id);
        chat::validate(&served.info.id, &model.capabilities(), &request)?;
        let started = async move { model.chat(request).await };
        Ok(Box::pin(futures::stream::once(started).try_flatten()))
    }

    #[cfg(feature = "llm")]
    fn complete_stream(
        &self,
        mut request: CompletionRequest,
    ) -> BoxFuture<'_, Result<CompletionStream>> {
        Box::pin(async move {
            let (served, model) = self.handle(&request.model, "text completions", |m| {
                m.completion.as_ref()
            })?;
            request.model.clone_from(&served.info.id);
            completion::validate(&served.info.id, &model.capabilities(), &request)?;
            Ok(model.complete(request).await?)
        })
    }

    #[cfg(feature = "llm")]
    fn embed(&self, mut request: EmbeddingRequest) -> BoxFuture<'_, Result<Embeddings>> {
        Box::pin(async move {
            let (served, model) =
                self.handle(&request.model, "embeddings", |m| m.embedding.as_ref())?;
            request.model.clone_from(&served.info.id);
            embedding::validate(&served.info.id, &model.capabilities(), &request)?;
            Ok(model.embed(request).await?)
        })
    }

    #[cfg(feature = "llm")]
    fn moderate(&self, mut request: ModerationRequest) -> BoxFuture<'_, Result<Moderation>> {
        Box::pin(async move {
            let (served, model) = self.moderation_model(request.model.as_deref())?;
            request.model = Some(served.info.id.clone());
            moderation::validate(&served.info.id, &model.capabilities(), &request)?;
            let mut out = model.moderate(request).await?;
            out.model.clone_from(&served.info.id);
            Ok(out)
        })
    }

    #[cfg(feature = "audio")]
    fn transcribe(
        &self,
        mut request: TranscriptionRequest,
    ) -> BoxFuture<'_, Result<TranscriptionStream>> {
        Box::pin(async move {
            let (served, model) = self.handle(&request.model, "transcriptions", |m| {
                m.transcription.as_ref()
            })?;
            request.model.clone_from(&served.info.id);
            let caps = model.capabilities();
            transcription::validate(&served.info.id, &caps, &request)?;
            let container = request.container.ok_or_else(|| Error::InvalidParameter {
                parameter: "file".into(),
                detail: "no audio format".into(),
            })?;
            let audio = teeny_audio::codec::decode(container, &request.audio)?
                .resampled_mono(caps.sample_rate);
            transcription::check_duration(&caps, &audio)?;
            // The model has the samples; the upload is not needed twice.
            request.audio = Vec::new();
            Ok(model.transcribe(request, audio).await?)
        })
    }

    #[cfg(feature = "audio")]
    fn speak(&self, mut request: SpeechRequest) -> BoxFuture<'_, Result<AudioStream>> {
        Box::pin(async move {
            let (served, model) = self.handle(&request.model, "speech", |m| m.speech.as_ref())?;
            request.model.clone_from(&served.info.id);
            let caps = model.capabilities();
            speech::validate(&served.info.id, &caps, &request)?;
            let (format, streamed) = (request.format, request.stream);
            let samples = model.speak(request).await?;
            Ok(speech::encode_stream(
                samples,
                format,
                caps.sample_rate,
                streamed,
            ))
        })
    }

    #[cfg(feature = "image")]
    fn paint(&self, mut request: ImageRequest) -> BoxFuture<'_, Result<EncodedImageStream>> {
        Box::pin(async move {
            let (served, model) = self.handle(&request.model, "images", |m| m.images.as_ref())?;
            request.model.clone_from(&served.info.id);
            images::validate(&served.info.id, &model.capabilities(), &request)?;
            let inputs = images::decode_inputs(&request)?;
            // The model has the decoded images; the uploads are not needed twice.
            request.images = Vec::new();
            request.mask = None;
            let format = request.format;
            let painted = model.paint(request, inputs).await?;
            Ok(images::encode_stream(painted, format))
        })
    }
}

#[cfg(all(test, feature = "llm"))]
mod tests {
    use super::*;
    use teeny_llm::chat::{EchoChatModel, Message, collect};

    fn echo() -> LoadedModel {
        LoadedModel::from_chat(Arc::new(EchoChatModel))
    }

    fn engine() -> ModelEngine {
        ModelEngine::new([echo()]).unwrap()
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
        let twice = ModelEngine::new([echo(), echo()]);
        assert!(matches!(twice, Err(Error::InvalidRequest(_))));
    }

    #[tokio::test]
    async fn test_model_engine_serves_embeddings_from_an_embedding_model() {
        use teeny_llm::embedding::{EchoEmbeddingModel, EmbeddingInput};
        let e = ModelEngine::new([
            echo(),
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
        let mut info = ModelInfo::from(EchoChatModel.info());
        info.id = "not-echo".into();
        let mismatched = ServedModel::new(info).with_chat(Arc::new(EchoChatModel));
        let err = ModelEngine::new([mismatched]).unwrap_err().to_string();
        assert!(
            err.contains("not-echo") && err.contains(EchoChatModel::MODEL_ID),
            "{err}"
        );
    }

    #[test]
    fn test_model_engine_refuses_an_api_the_model_lacks() {
        let mut info = ModelInfo::from(EchoChatModel.info());
        info.id = "weights-only".into();
        let e = ModelEngine::new([ServedModel::new(info)]).unwrap();
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

    #[test]
    fn test_served_model_takes_a_loaded_models_handles() {
        let m = ServedModel::from(echo());
        assert_eq!(m.info.id, EchoChatModel::MODEL_ID);
        assert_eq!(m.apis(), ["chat completions"]);
        assert!(m.model.is_none());
    }

    /// A chat model posing as a factory model, to test tag resolution without weights.
    #[cfg(feature = "functiongemma3")]
    struct Posing(Model);

    #[cfg(feature = "functiongemma3")]
    impl ChatModel for Posing {
        fn info(&self) -> teeny_llm::ModelInfo {
            teeny_llm::ModelInfo {
                id: self.0.tag().into(),
                ..EchoChatModel.info()
            }
        }
        fn capabilities(&self) -> teeny_llm::chat::Capabilities {
            EchoChatModel.capabilities()
        }
        fn chat(&self, request: ChatRequest) -> BoxFuture<'_, teeny_llm::Result<ChatStream>> {
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
        let e = ModelEngine::new([loaded]).unwrap();
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
