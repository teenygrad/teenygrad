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

//! [`Engine`], the trait routes drive, and [`ModelInfo`], what it serves.

#[cfg(any(feature = "llm", feature = "audio", feature = "image"))]
use futures::future::BoxFuture;

use crate::{Error, Result};

#[cfg(feature = "image")]
use teeny_llm::images::{EncodedImageStream, ImageRequest};
#[cfg(feature = "llm")]
use teeny_llm::{
    chat::{ChatRequest, ChatStream},
    completion::{CompletionRequest, CompletionStream},
    embedding::{EmbeddingRequest, Embeddings},
    moderation::{Moderation, ModerationRequest},
};
#[cfg(feature = "audio")]
use teeny_llm::{
    speech::{AudioStream, SpeechRequest},
    transcription::{TranscriptionRequest, TranscriptionStream},
};

/// A model an [`Engine`] serves, as `/v1/models` lists it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelInfo {
    pub id: String,
    /// Unix timestamp (seconds) the model was loaded/created.
    pub created: u64,
    pub owned_by: String,
}

#[cfg(feature = "llm")]
impl From<teeny_llm::ModelInfo> for ModelInfo {
    fn from(m: teeny_llm::ModelInfo) -> Self {
        Self {
            id: m.id,
            created: m.created,
            owned_by: m.owned_by,
        }
    }
}

/// The future an engine method returns when it does not serve that API.
#[cfg(any(feature = "llm", feature = "audio", feature = "image"))]
fn unsupported<T: Send + 'static>(
    engine: &'static str,
    operation: &'static str,
) -> BoxFuture<'static, Result<T>> {
    Box::pin(async move { Err(Error::Unsupported { engine, operation }) })
}

/// An inference backend, over the APIs of every domain this build includes (each a
/// feature: `llm`, `audio`, `image`).
///
/// Object-safe so routes can hold an `Arc<dyn Engine>` without being generic over the
/// backend. Each API method defaults to [`Error::Unsupported`]; [`ModelEngine`] implements
/// them all.
///
/// Validation errors (unknown model, bad parameters) are returned before any work starts,
/// so routes can map them to a proper error response.
///
/// [`ModelEngine`]: crate::ModelEngine
pub trait Engine: Send + Sync + 'static {
    /// A short, stable name for diagnostics (e.g. `"models"`).
    fn name(&self) -> &'static str;

    /// The models this engine can serve.
    fn models(&self) -> Vec<ModelInfo>;

    /// Looks up a served model by id.
    fn model(&self, id: &str) -> Result<ModelInfo> {
        let models = self.models();
        match models.iter().find(|m| m.id == id) {
            Some(m) => Ok(m.clone()),
            None => Err(Error::ModelNotFound {
                requested: id.to_owned(),
                available: models.into_iter().map(|m| m.id).collect(),
            }),
        }
    }

    /// Starts a chat completion (tools, structured output, content parts, several
    /// choices).
    #[cfg(feature = "llm")]
    fn chat_stream(&self, request: ChatRequest) -> Result<ChatStream> {
        let _ = request;
        Err(Error::Unsupported {
            engine: self.name(),
            operation: "chat requests",
        })
    }

    /// Starts a text completion (raw continuation of prompts).
    #[cfg(feature = "llm")]
    fn complete_stream(
        &self,
        request: CompletionRequest,
    ) -> BoxFuture<'_, Result<CompletionStream>> {
        let _ = request;
        unsupported(self.name(), "text completions")
    }

    /// Embeds `request`.
    #[cfg(feature = "llm")]
    fn embed(&self, request: EmbeddingRequest) -> BoxFuture<'_, Result<Embeddings>> {
        let _ = request;
        unsupported(self.name(), "embeddings")
    }

    /// Classifies `request.inputs`.
    #[cfg(feature = "llm")]
    fn moderate(&self, request: ModerationRequest) -> BoxFuture<'_, Result<Moderation>> {
        let _ = request;
        unsupported(self.name(), "moderations")
    }

    /// Transcribes (or translates) a recording. Audio decoding errors also resolve the
    /// future to `Err` before the model runs.
    #[cfg(feature = "audio")]
    fn transcribe(
        &self,
        request: TranscriptionRequest,
    ) -> BoxFuture<'_, Result<TranscriptionStream>> {
        let _ = request;
        unsupported(self.name(), "transcriptions")
    }

    /// Speaks `request.input`, in `request.format`.
    #[cfg(feature = "audio")]
    fn speak(&self, request: SpeechRequest) -> BoxFuture<'_, Result<AudioStream>> {
        let _ = request;
        unsupported(self.name(), "speech")
    }

    /// Paints images for `request` (a generation, edit or variation), in
    /// `request.format`. Upload decoding errors also resolve the future to `Err` before
    /// the model runs.
    #[cfg(feature = "image")]
    fn paint(&self, request: ImageRequest) -> BoxFuture<'_, Result<EncodedImageStream>> {
        let _ = request;
        unsupported(self.name(), "images")
    }
}
