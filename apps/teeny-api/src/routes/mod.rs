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

//! The axum [`Router`] over an [`Engine`]: `/v1/models`, `/v1/events` and `/health`
//! always, and each domain's endpoints with its feature (`llm`, `audio`, `image`).

#[cfg(feature = "audio")]
mod audio;
mod events;
#[cfg(feature = "image")]
mod images;
#[cfg(feature = "llm")]
mod llm;

#[cfg(any(feature = "llm", feature = "audio", feature = "image"))]
use std::convert::Infallible;
use std::sync::Arc;
#[cfg(any(feature = "llm", feature = "image"))]
use std::time::{SystemTime, UNIX_EPOCH};

#[cfg(any(feature = "audio", feature = "image"))]
use axum::extract::{
    Multipart,
    multipart::{MultipartError, MultipartRejection},
};
#[cfg(any(feature = "llm", feature = "audio", feature = "image"))]
use axum::response::{
    IntoResponse, Response,
    sse::{Event, KeepAlive, Sse},
};
use axum::{
    Json, Router,
    extract::{FromRef, Path, State},
    middleware,
    routing::get,
};
#[cfg(feature = "llm")]
use futures::stream;
#[cfg(any(feature = "llm", feature = "audio", feature = "image"))]
use futures::{Stream, StreamExt};

use crate::Engine;
use crate::events::Events;
#[cfg(any(feature = "audio", feature = "image"))]
use crate::openai::form::FormPart;
use crate::openai::{
    error::ApiError,
    models::{ListModelsResponse, Model},
};

/// What most handlers take: the engine.
type AppState = Arc<dyn Engine>;

/// What the router holds; handlers extract the part they need.
#[derive(Clone)]
struct ServerState {
    engine: AppState,
    events: Events,
}

impl FromRef<ServerState> for AppState {
    fn from_ref(state: &ServerState) -> Self {
        Arc::clone(&state.engine)
    }
}

impl FromRef<ServerState> for Events {
    fn from_ref(state: &ServerState) -> Self {
        state.events.clone()
    }
}

/// Builds the OpenAI-compatible router (`/v1/...` plus `/health`) over `engine`.
/// `/v1/events` streams what is published to `events`, along with each API request the
/// router answers.
pub fn router(engine: Arc<dyn Engine>, events: Events) -> Router {
    let router = Router::new()
        .route("/health", get(|| async { "ok" }))
        .route("/v1/models", get(list_models))
        .route("/v1/models/{model}", get(retrieve_model))
        .route("/v1/events", get(events::events_socket));
    #[cfg(feature = "llm")]
    let router = llm::routes(router);
    #[cfg(feature = "audio")]
    let router = audio::routes(router);
    #[cfg(feature = "image")]
    let router = images::routes(router);
    router
        .layer(middleware::from_fn_with_state(
            events.clone(),
            events::track,
        ))
        .with_state(ServerState { engine, events })
}

#[cfg(any(feature = "llm", feature = "image"))]
fn unix_now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_secs())
}

fn new_id(prefix: &str) -> String {
    format!("{prefix}-{}", uuid::Uuid::new_v4().simple())
}

#[cfg(any(feature = "llm", feature = "audio", feature = "image"))]
fn sse_event<T: serde::Serialize>(value: &T) -> Event {
    // Serializing these plain structs can't fail; fall back to an empty object rather than panic.
    Event::default().data(serde_json::to_string(value).unwrap_or_else(|_| "{}".to_owned()))
}

/// An SSE response of `events`, ended with chat's `[DONE]`.
#[cfg(feature = "llm")]
fn sse_response(events: impl Stream<Item = Event> + Send + 'static) -> Response {
    let done = stream::once(async { Event::default().data("[DONE]") });
    sse_only(events.chain(done))
}

/// An SSE response of `events`, as they are (the audio and image streams end with their
/// own done events, without `[DONE]`).
#[cfg(any(feature = "llm", feature = "audio", feature = "image"))]
fn sse_only(events: impl Stream<Item = Event> + Send + 'static) -> Response {
    Sse::new(events.map(Ok::<_, Infallible>))
        .keep_alive(KeepAlive::default())
        .into_response()
}

/// Turns an error that happens mid-stream into an OpenAI-style `data: {"error": ...}` event.
#[cfg(any(feature = "llm", feature = "audio", feature = "image"))]
fn sse_error(e: impl Into<ApiError> + std::fmt::Display) -> Event {
    tracing::warn!(error = %e, "generation failed mid-stream");
    sse_event(&e.into().into_response_body())
}

// ---- models ----------------------------------------------------------------------------------

async fn list_models(State(engine): State<AppState>) -> Json<ListModelsResponse> {
    Json(ListModelsResponse {
        object: "list".to_owned(),
        data: engine.models().into_iter().map(Model::from).collect(),
    })
}

async fn retrieve_model(
    State(engine): State<AppState>,
    Path(model): Path<String>,
) -> Result<Json<Model>, ApiError> {
    Ok(Json(engine.model(&model)?.into()))
}

// ---- multipart uploads -----------------------------------------------------------------------

#[cfg(any(feature = "audio", feature = "image"))]
fn multipart_error(status: axum::http::StatusCode, text: String) -> ApiError {
    let mut e = ApiError::invalid_request(text, None);
    e.status = status;
    e
}

#[cfg(any(feature = "audio", feature = "image"))]
impl From<MultipartRejection> for ApiError {
    fn from(r: MultipartRejection) -> Self {
        multipart_error(r.status(), r.body_text())
    }
}

#[cfg(any(feature = "audio", feature = "image"))]
impl From<MultipartError> for ApiError {
    fn from(e: MultipartError) -> Self {
        multipart_error(e.status(), e.body_text())
    }
}

#[cfg(any(feature = "audio", feature = "image"))]
async fn read_form(mut form: Multipart) -> Result<Vec<FormPart>, ApiError> {
    let mut parts = Vec::new();
    while let Some(field) = form.next_field().await? {
        let name = field.name().unwrap_or_default().to_owned();
        let file_name = field.file_name().map(str::to_owned);
        let data = field.bytes().await?.to_vec();
        parts.push(FormPart {
            name,
            file_name,
            data,
        });
    }
    Ok(parts)
}
