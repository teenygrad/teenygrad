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

use std::{
    convert::Infallible,
    sync::Arc,
    time::{SystemTime, UNIX_EPOCH},
};

use axum::{
    Json, Router,
    extract::{Path, State, rejection::JsonRejection},
    response::{
        IntoResponse, Response,
        sse::{Event, KeepAlive, Sse},
    },
    routing::{get, post},
};
use futures::{Stream, StreamExt, stream};
use teeny_llm_core::Engine;
use teeny_llm_core::chat::{self, ChatEvent};
use teeny_llm_core::completion::{self, CompletionEvent};

use crate::openai::{
    FinishReason, Usage,
    chat::{
        self as chat_wire, ChatCompletionStreamChoice, ChatCompletionStreamResponseDelta,
        CreateChatCompletionRequest, CreateChatCompletionResponse,
        CreateChatCompletionStreamResponse, Role, choice_from_core, delta_from_core,
    },
    completions::{
        CompletionChoice, CompletionLogprobs, CreateCompletionRequest, CreateCompletionResponse,
    },
    embeddings::{
        CreateEmbeddingRequest, CreateEmbeddingResponse, Embedding, EmbeddingUsage, EmbeddingVector,
    },
    error::ApiError,
    models::{ListModelsResponse, Model},
};

type AppState = Arc<dyn Engine>;

/// Builds the OpenAI-compatible router (`/v1/...` plus `/health`) over `engine`.
pub fn router(engine: Arc<dyn Engine>) -> Router {
    Router::new()
        .route("/health", get(|| async { "ok" }))
        .route("/v1/models", get(list_models))
        .route("/v1/models/{model}", get(retrieve_model))
        .route("/v1/chat/completions", post(chat_completions))
        .route("/v1/completions", post(completions))
        .route("/v1/embeddings", post(embeddings))
        .with_state(engine)
}

fn unix_now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_secs())
}

fn new_id(prefix: &str) -> String {
    format!("{prefix}-{}", uuid::Uuid::new_v4().simple())
}

fn sse_event<T: serde::Serialize>(value: &T) -> Event {
    // Serializing these plain structs can't fail; fall back to an empty object rather than panic.
    Event::default().data(serde_json::to_string(value).unwrap_or_else(|_| "{}".to_owned()))
}

fn sse_response(events: impl Stream<Item = Event> + Send + 'static) -> Response {
    let done = stream::once(async { Event::default().data("[DONE]") });
    Sse::new(events.chain(done).map(Ok::<_, Infallible>))
        .keep_alive(KeepAlive::default())
        .into_response()
}

/// Turns an engine error that happens mid-stream into an OpenAI-style `data: {"error": ...}` event.
fn sse_error(e: teeny_llm_core::Error) -> Event {
    tracing::warn!(error = %e, "generation failed mid-stream");
    sse_event(&ApiError::from(e).into_response_body())
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

// ---- chat completions ------------------------------------------------------------------------

async fn chat_completions(
    State(engine): State<AppState>,
    body: Result<Json<CreateChatCompletionRequest>, JsonRejection>,
) -> Result<Response, ApiError> {
    let Json(req) = body?;
    let stream = req.streamed();
    let request = req.into_core()?;
    // The canonical id, whatever tag the request named the model by.
    let model = engine.model(&request.model)?.id;
    let (n, include_usage) = (request.n, request.include_usage);
    let logprobs = request.logprobs.is_some();
    let events = engine.chat_stream(request)?;
    let id = new_id("chatcmpl");
    let created = unix_now();

    if !stream {
        let done = chat::collect(&model, n, events).await?;
        return Ok(Json(CreateChatCompletionResponse {
            id,
            object: "chat.completion".to_owned(),
            created,
            model,
            choices: done
                .choices
                .into_iter()
                .map(|c| choice_from_core(c, logprobs))
                .collect(),
            usage: Some(done.usage.into()),
        })
        .into_response());
    }

    let chunk = move |choices: Vec<ChatCompletionStreamChoice>, usage: Option<Usage>| {
        sse_event(&CreateChatCompletionStreamResponse {
            id: id.clone(),
            object: "chat.completion.chunk".to_owned(),
            created,
            model: model.clone(),
            choices,
            usage,
        })
    };
    let choice = |index, delta, logprobs, finish_reason| ChatCompletionStreamChoice {
        index,
        delta,
        logprobs,
        finish_reason,
    };
    // Each choice opens with its role, as OpenAI's streams do.
    let opening: Vec<Event> = (0..n)
        .map(|index| {
            let delta = ChatCompletionStreamResponseDelta {
                role: Some(Role::Assistant),
                content: Some(String::new()),
                ..ChatCompletionStreamResponseDelta::default()
            };
            chunk(vec![choice(index, delta, None, None)], None)
        })
        .collect();
    let rest = events.filter_map(move |item| {
        let event = match item {
            Err(e) => Some(sse_error(e)),
            Ok(ChatEvent::Delta { index, delta }) => {
                let lp = chat_wire::choice_logprobs(
                    logprobs,
                    delta.content_logprobs.clone(),
                    delta.refusal_logprobs.clone(),
                );
                Some(chunk(
                    vec![choice(index, delta_from_core(delta), lp, None)],
                    None,
                ))
            }
            Ok(ChatEvent::Finish { index, reason }) => Some(chunk(
                vec![choice(
                    index,
                    ChatCompletionStreamResponseDelta::default(),
                    None,
                    Some(FinishReason::from(reason)),
                )],
                None,
            )),
            Ok(ChatEvent::Usage(u)) => include_usage.then(|| chunk(Vec::new(), Some(u.into()))),
        };
        std::future::ready(event)
    });
    Ok(sse_response(stream::iter(opening).chain(rest)))
}

// ---- legacy completions ----------------------------------------------------------------------

async fn completions(
    State(engine): State<AppState>,
    body: Result<Json<CreateCompletionRequest>, JsonRejection>,
) -> Result<Response, ApiError> {
    let Json(req) = body?;
    let stream = req.streamed();
    let request = req.into_core()?;
    let model = engine.model(&request.model)?.id;
    let choices = request.prompts.len() * request.n as usize;
    let (include_usage, logprobs) = (request.include_usage, request.logprobs.is_some());
    let events = engine.complete_stream(request).await?;
    let id = new_id("cmpl");
    let created = unix_now();
    let response = move |choices, usage| CreateCompletionResponse {
        id: id.clone(),
        object: "text_completion".to_owned(),
        created,
        model: model.clone(),
        choices,
        usage,
    };

    if !stream {
        let done = completion::collect(choices, events).await?;
        let choices = done
            .choices
            .into_iter()
            .map(|c| CompletionChoice::from_core(c, logprobs))
            .collect();
        return Ok(Json(response(choices, Some(done.usage.into()))).into_response());
    }

    let events = events.filter_map(move |item| {
        let event = match item {
            Err(e) => Some(sse_error(e)),
            Ok(CompletionEvent::Text {
                index,
                text,
                logprobs: lp,
            }) => {
                let choice = CompletionChoice {
                    index,
                    text,
                    logprobs: CompletionLogprobs::from_core(logprobs, lp),
                    finish_reason: None,
                };
                Some(sse_event(&response(vec![choice], None)))
            }
            Ok(CompletionEvent::Finish { index, reason }) => {
                let choice = CompletionChoice {
                    index,
                    text: String::new(),
                    logprobs: None,
                    finish_reason: Some(reason.into()),
                };
                Some(sse_event(&response(vec![choice], None)))
            }
            Ok(CompletionEvent::Usage(u)) => {
                include_usage.then(|| sse_event(&response(Vec::new(), Some(u.into()))))
            }
        };
        std::future::ready(event)
    });
    Ok(sse_response(events))
}

// ---- embeddings ------------------------------------------------------------------------------

async fn embeddings(
    State(engine): State<AppState>,
    body: Result<Json<CreateEmbeddingRequest>, JsonRejection>,
) -> Result<Json<CreateEmbeddingResponse>, ApiError> {
    let Json(req) = body?;
    let (request, format) = req.into_core()?;
    let model = engine.model(&request.model)?.id;
    let out = engine.embed(request).await?;
    Ok(Json(CreateEmbeddingResponse {
        object: "list".to_owned(),
        data: out
            .vectors
            .into_iter()
            .enumerate()
            .map(|(i, v)| Embedding {
                object: "embedding".to_owned(),
                index: u32::try_from(i).unwrap_or(u32::MAX),
                embedding: EmbeddingVector::encode(v, format),
            })
            .collect(),
        model,
        usage: EmbeddingUsage {
            prompt_tokens: out.prompt_tokens,
            total_tokens: out.prompt_tokens,
        },
    }))
}
