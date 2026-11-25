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

//! The model factory: [`Model::load`] turns a [`Model`] into a [`LoadedModel`], whatever
//! its weights, device and kernels, so frontends never handle them.

use std::fmt;
use std::path::PathBuf;
use std::sync::Arc;

use super::Model;
use crate::chat::ChatModel;
use crate::completion::CompletionModel;
use crate::embedding::EmbeddingModel;
use crate::speech::SpeechModel;
use crate::tokenizer::Tokenizer;
use crate::transcription::TranscriptionModel;
use crate::{Error, ModelInfo, Result};

/// Called with a line of loading progress (`"compiling the 1-token step"`), for frontends
/// to show while a model loads: it can take minutes on a board.
pub type Progress = Arc<dyn Fn(&str) + Send + Sync>;

/// How to load a [`Model`]. The defaults cover the common case; set what differs with
/// the `with_` methods.
#[derive(Clone, Default)]
#[non_exhaustive]
pub struct LoadOptions {
    /// The weights folder; `None` looks in the Hugging Face cache (see
    /// [`Model::locate`]).
    pub model_dir: Option<PathBuf>,
    /// The longest conversation, in tokens, the model keeps (its KV cache). `None`
    /// takes the model's own default; it may not exceed the context the model was
    /// trained for ([`ModelSpec::context_length`](super::ModelSpec::context_length)).
    pub context_length: Option<u32>,
    /// Told what loading is doing.
    pub progress: Option<Progress>,
}

impl fmt::Debug for LoadOptions {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("LoadOptions")
            .field("model_dir", &self.model_dir)
            .field("context_length", &self.context_length)
            .field("progress", &self.progress.as_ref().map(|_| "..."))
            .finish()
    }
}

impl LoadOptions {
    /// Loads the weights from `dir`.
    pub fn with_model_dir(mut self, dir: impl Into<PathBuf>) -> Self {
        self.model_dir = Some(dir.into());
        self
    }

    /// Keeps conversations of up to `tokens` tokens.
    pub fn with_context_length(mut self, tokens: u32) -> Self {
        self.context_length = Some(tokens);
        self
    }

    /// Reports loading progress to `progress`.
    pub fn with_progress(mut self, progress: impl Fn(&str) + Send + Sync + 'static) -> Self {
        self.progress = Some(Arc::new(progress));
        self
    }

    /// Reports `what` to the progress callback, if there is one.
    pub fn report(&self, what: &str) {
        if let Some(p) = &self.progress {
            p(what);
        }
    }

    /// The context length to load `model` with: the one asked for, checked against what
    /// the model was trained for, else `default`.
    pub fn context_length_for(&self, model: Model, default: u32) -> Result<u32> {
        let trained = model.spec().context_length;
        match self.context_length {
            None => Ok(default.min(trained)),
            Some(0) => Err(Error::InvalidParameter {
                parameter: "context_length".into(),
                detail: "must be at least 1 token".into(),
            }),
            Some(n) if n > trained => Err(Error::InvalidParameter {
                parameter: "context_length".into(),
                detail: format!("{model} was trained for at most {trained} tokens, not {n}"),
            }),
            Some(n) => Ok(n),
        }
    }
}

/// A loaded model, ready to serve: the APIs it implements, each as a trait object, and
/// what describes it.
///
/// One model can serve several APIs (chat and completions, say) from one set of weights,
/// so each API is an optional handle rather than the model being one trait. Built by
/// [`Model::load`], or by hand for models outside the factory (tests' echo models) with
/// [`LoadedModel::from_chat`] and the `with_` methods; [`ModelEngine`](crate::ModelEngine)
/// serves them.
#[derive(Clone)]
#[non_exhaustive]
pub struct LoadedModel {
    /// The model it is, when it came from the factory; `None` for models built by hand.
    pub model: Option<Model>,
    /// Its id (a [`Model`]'s canonical tag) and provenance, as `/v1/models` lists it.
    pub info: ModelInfo,
    /// The exact number of parameters in its weights, when known.
    pub parameters: Option<u64>,
    /// Its tokenizer, for counting tokens without generating.
    pub tokenizer: Option<Arc<dyn Tokenizer>>,
    /// Chat completions.
    pub chat: Option<Arc<dyn ChatModel>>,
    /// Text completions.
    pub completion: Option<Arc<dyn CompletionModel>>,
    /// Embeddings.
    pub embedding: Option<Arc<dyn EmbeddingModel>>,
    /// Speech to text.
    pub transcription: Option<Arc<dyn TranscriptionModel>>,
    /// Text to speech.
    pub speech: Option<Arc<dyn SpeechModel>>,
}

impl fmt::Debug for LoadedModel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("LoadedModel")
            .field("model", &self.model)
            .field("info", &self.info)
            .field("parameters", &self.parameters)
            .field("tokenizer", &self.tokenizer.is_some())
            .field("chat", &self.chat.is_some())
            .field("completion", &self.completion.is_some())
            .field("embedding", &self.embedding.is_some())
            .field("transcription", &self.transcription.is_some())
            .field("speech", &self.speech.is_some())
            .finish()
    }
}

impl LoadedModel {
    /// A model described by `info` that serves no API yet.
    pub fn new(info: ModelInfo) -> Self {
        Self {
            model: None,
            info,
            parameters: None,
            tokenizer: None,
            chat: None,
            completion: None,
            embedding: None,
            transcription: None,
            speech: None,
        }
    }

    /// A chat model, described by its own [`ChatModel::info`].
    pub fn from_chat(chat: Arc<dyn ChatModel>) -> Self {
        Self::new(chat.info()).with_chat(chat)
    }

    /// A completion model, described by its own [`CompletionModel::info`].
    pub fn from_completion(completion: Arc<dyn CompletionModel>) -> Self {
        Self::new(completion.info()).with_completion(completion)
    }

    /// An embedding model, described by its own [`EmbeddingModel::info`].
    pub fn from_embedding(embedding: Arc<dyn EmbeddingModel>) -> Self {
        Self::new(embedding.info()).with_embedding(embedding)
    }

    /// A transcription model, described by its own [`TranscriptionModel::info`].
    pub fn from_transcription(transcription: Arc<dyn TranscriptionModel>) -> Self {
        Self::new(transcription.info()).with_transcription(transcription)
    }

    /// A speech model, described by its own [`SpeechModel::info`].
    pub fn from_speech(speech: Arc<dyn SpeechModel>) -> Self {
        Self::new(speech.info()).with_speech(speech)
    }

    /// Records which [`Model`] it is, so short tags resolve to it.
    pub fn with_model(mut self, model: Model) -> Self {
        self.model = Some(model);
        self
    }

    /// Records its exact parameter count.
    pub fn with_parameters(mut self, parameters: u64) -> Self {
        self.parameters = Some(parameters);
        self
    }

    /// Gives it a tokenizer.
    pub fn with_tokenizer(mut self, tokenizer: Arc<dyn Tokenizer>) -> Self {
        self.tokenizer = Some(tokenizer);
        self
    }

    /// Serves chat completions with `chat`.
    pub fn with_chat(mut self, chat: Arc<dyn ChatModel>) -> Self {
        self.chat = Some(chat);
        self
    }

    /// Serves text completions with `completion`.
    pub fn with_completion(mut self, completion: Arc<dyn CompletionModel>) -> Self {
        self.completion = Some(completion);
        self
    }

    /// Serves embeddings with `embedding`.
    pub fn with_embedding(mut self, embedding: Arc<dyn EmbeddingModel>) -> Self {
        self.embedding = Some(embedding);
        self
    }

    /// Serves transcriptions with `transcription`.
    pub fn with_transcription(mut self, transcription: Arc<dyn TranscriptionModel>) -> Self {
        self.transcription = Some(transcription);
        self
    }

    /// Serves speech with `speech`.
    pub fn with_speech(mut self, speech: Arc<dyn SpeechModel>) -> Self {
        self.speech = Some(speech);
        self
    }

    /// The APIs it serves, by name, for messages.
    pub fn apis(&self) -> Vec<&'static str> {
        let mut apis = Vec::new();
        if self.chat.is_some() {
            apis.push("chat completions");
        }
        if self.completion.is_some() {
            apis.push("text completions");
        }
        if self.embedding.is_some() {
            apis.push("embeddings");
        }
        if self.transcription.is_some() {
            apis.push("transcriptions");
        }
        if self.speech.is_some() {
            apis.push("speech");
        }
        apis
    }
}

impl Model {
    /// Loads the model: finds its weights, opens its device, compiles its kernels, and
    /// returns it ready to serve.
    ///
    /// This blocks for as long as loading takes (seconds on a GPU, minutes compiling for
    /// a board), so async code calls it from a blocking task (`spawn_blocking`). A model
    /// whose runtime is tied to a thread runs it on a worker of its own; the handles in
    /// the [`LoadedModel`] send it work, and dropping the last of them stops the worker
    /// and frees the device.
    pub fn load(self, options: &LoadOptions) -> Result<LoadedModel> {
        // With no model feature enabled there is nothing to load, and `options` is unused.
        let _ = options;
        match self {
            #[cfg(feature = "functiongemma3")]
            Model::FunctionGemma3 => super::functiongemma::load(self, options),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chat::EchoChatModel;

    #[test]
    fn test_loaded_model_from_chat_takes_its_info() {
        let m = LoadedModel::from_chat(Arc::new(EchoChatModel));
        assert_eq!(m.info.id, EchoChatModel::MODEL_ID);
        assert_eq!(m.apis(), ["chat completions"]);
        assert!(m.model.is_none());
        assert!(LoadedModel::new(m.info.clone()).apis().is_empty());
    }

    #[test]
    fn test_load_options_report_progress() {
        let seen = Arc::new(std::sync::Mutex::new(Vec::new()));
        let sink = Arc::clone(&seen);
        let opts = LoadOptions::default()
            .with_model_dir("/weights")
            .with_progress(move |s| sink.lock().unwrap().push(s.to_owned()));
        opts.report("loading weights");
        LoadOptions::default().report("nobody listens");
        assert_eq!(*seen.lock().unwrap(), ["loading weights"]);
        assert!(format!("{opts:?}").contains("/weights"));
    }

    #[cfg(feature = "functiongemma3")]
    #[test]
    fn test_context_length_is_checked_against_the_trained_context() {
        let m = Model::FunctionGemma3;
        let trained = m.spec().context_length;
        let opts = LoadOptions::default();
        assert_eq!(opts.context_length_for(m, 2048).unwrap(), 2048);
        assert_eq!(opts.context_length_for(m, u32::MAX).unwrap(), trained);
        let asked = LoadOptions::default().with_context_length(512);
        assert_eq!(asked.context_length_for(m, 2048).unwrap(), 512);
        for bad in [0, trained + 1] {
            let opts = LoadOptions::default().with_context_length(bad);
            assert!(matches!(
                opts.context_length_for(m, 2048),
                Err(Error::InvalidParameter { .. })
            ));
        }
    }
}
