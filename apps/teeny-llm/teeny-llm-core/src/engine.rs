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

use futures::future::BoxFuture;

use crate::chat::{ChatRequest, ChatStream};
use crate::completion::{CompletionRequest, CompletionStream};
use crate::embedding::{EmbeddingRequest, Embeddings};
use crate::moderation::{Moderation, ModerationRequest};
use crate::speech::{AudioStream, SpeechRequest};
use crate::transcription::{TranscriptionRequest, TranscriptionStream};
use crate::{Error, ModelInfo, Result};

/// An inference backend.
///
/// Object-safe so frontends can hold an `Arc<dyn Engine>` without being generic over the backend.
pub trait Engine: Send + Sync + 'static {
    /// A short, stable name for diagnostics (e.g. `"echo"`).
    fn name(&self) -> &'static str;

    /// The models this engine can serve.
    fn models(&self) -> Vec<ModelInfo>;

    /// Starts a chat completion with the full chat API (tools, structured output, content
    /// parts, several choices). Engines that serve [`ChatModel`](crate::chat::ChatModel)s
    /// implement it; the rest keep the default.
    ///
    /// Validation errors (unknown model, bad parameters) are returned here, before
    /// streaming starts, so frontends can map them to a proper error response.
    fn chat_stream(&self, request: ChatRequest) -> Result<ChatStream> {
        let _ = request;
        Err(Error::Unsupported {
            engine: self.name(),
            operation: "chat requests",
        })
    }

    /// Starts a text completion (raw continuation of prompts). Engines that serve
    /// [`CompletionModel`](crate::completion::CompletionModel)s implement it; the rest keep
    /// the default.
    ///
    /// Validation errors resolve the future to `Err` before streaming starts.
    fn complete_stream(
        &self,
        request: CompletionRequest,
    ) -> BoxFuture<'_, Result<CompletionStream>> {
        let _ = request;
        let name = self.name();
        Box::pin(async move {
            Err(Error::Unsupported {
                engine: name,
                operation: "text completions",
            })
        })
    }

    /// Embeds `request`. Engines that serve
    /// [`EmbeddingModel`](crate::embedding::EmbeddingModel)s implement it; the rest keep the
    /// default.
    ///
    /// Validation errors resolve the future to `Err` before any work is done.
    fn embed(&self, request: EmbeddingRequest) -> BoxFuture<'_, Result<Embeddings>> {
        let _ = request;
        let name = self.name();
        Box::pin(async move {
            Err(Error::Unsupported {
                engine: name,
                operation: "embeddings",
            })
        })
    }

    /// Transcribes (or translates) a recording. Engines that serve
    /// [`TranscriptionModel`](crate::transcription::TranscriptionModel)s implement it; the
    /// rest keep the default.
    ///
    /// Validation and audio decoding errors resolve the future to `Err` before the model
    /// runs.
    fn transcribe(
        &self,
        request: TranscriptionRequest,
    ) -> BoxFuture<'_, Result<TranscriptionStream>> {
        let _ = request;
        let name = self.name();
        Box::pin(async move {
            Err(Error::Unsupported {
                engine: name,
                operation: "transcriptions",
            })
        })
    }

    /// Speaks `request.input`, in `request.format`. Engines that serve
    /// [`SpeechModel`](crate::speech::SpeechModel)s implement it; the rest keep the
    /// default.
    ///
    /// Validation errors resolve the future to `Err` before the model runs.
    fn speak(&self, request: SpeechRequest) -> BoxFuture<'_, Result<AudioStream>> {
        let _ = request;
        let name = self.name();
        Box::pin(async move {
            Err(Error::Unsupported {
                engine: name,
                operation: "speech",
            })
        })
    }

    /// Classifies `request.inputs`. Engines that serve
    /// [`ModerationModel`](crate::moderation::ModerationModel)s implement it; the rest keep
    /// the default.
    fn moderate(&self, request: ModerationRequest) -> BoxFuture<'_, Result<Moderation>> {
        let _ = request;
        let name = self.name();
        Box::pin(async move {
            Err(Error::Unsupported {
                engine: name,
                operation: "moderations",
            })
        })
    }

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
}
