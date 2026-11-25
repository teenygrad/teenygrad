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

//! The [`ChatModel`] trait.

use futures::future::BoxFuture;

use super::capabilities::{Capabilities, validate};
use super::request::ChatRequest;
use super::response::{ChatCompletion, ChatStream, collect};
use crate::{ModelInfo, Result};

/// A model that answers chat completion requests.
///
/// Each model in `teeny_llm_core::models` (behind its feature) implements this; frontends reach
/// models through an [`Engine`](crate::Engine), and [`ModelEngine`](crate::ModelEngine)
/// serves them as one, through the [`LoadedModel`](crate::models::LoadedModel)s that
/// [`Model::load`](crate::models::Model::load) returns.
///
/// Object-safe, so models can be held as `Arc<dyn ChatModel>`; [`ChatModel::chat`] returns
/// a boxed future rather than being an `async fn` for that reason. A model owns its
/// threading and its device: one whose runtime is synchronous, or tied to a thread, runs it
/// on a worker of its own and streams results back.
pub trait ChatModel: Send + Sync + 'static {
    /// The model's id and provenance, as `/v1/models` lists it.
    fn info(&self) -> ModelInfo;

    /// What the model can do. Requests are checked against this before they reach
    /// [`ChatModel::chat`].
    fn capabilities(&self) -> Capabilities;

    /// Starts answering `request`, resolving to the stream of its output once generation
    /// has begun (after any queueing behind other requests).
    ///
    /// `request` has passed [`validate`] against [`ChatModel::capabilities`], so the model
    /// may rely on it asking for nothing it does not support. Errors that arise before any
    /// output (the prompt does not fit the context, say) resolve the future to `Err`;
    /// errors part-way through end the stream with one.
    ///
    /// The stream keeps the contract described on [`ChatEvent`](super::ChatEvent): per
    /// choice any number of deltas then one finish, and a final usage.
    fn chat(&self, request: ChatRequest) -> BoxFuture<'_, Result<ChatStream>>;
}

impl dyn ChatModel {
    /// Validates `request` against the model and starts answering it.
    pub async fn chat_checked(&self, request: ChatRequest) -> Result<ChatStream> {
        validate(&self.info().id, &self.capabilities(), &request)?;
        self.chat(request).await
    }

    /// Validates `request`, answers it, and puts the streamed output together.
    pub async fn complete(&self, request: ChatRequest) -> Result<ChatCompletion> {
        let n = request.n;
        let stream = self.chat_checked(request).await?;
        collect(&self.info().id, n, stream).await
    }
}
