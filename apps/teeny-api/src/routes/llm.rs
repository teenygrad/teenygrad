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

//! The LLM endpoints: chat completions, legacy completions, embeddings and moderations.

use axum::{
    Extension, Json, Router,
    extract::{State, rejection::JsonRejection},
    response::{IntoResponse, Response, sse::Event},
    routing::post,
};
use futures::{StreamExt, stream};
use teeny_llm::chat::{self, ChatEvent};
use teeny_llm::completion::{self, CompletionEvent};

use crate::events::{Events, RequestId, ServerEvent};
use crate::openai::timings::Timings;

use super::{AppState, ServerState, new_id, sse_error, sse_event, sse_response, unix_now};
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
    moderations::{CreateModerationRequest, CreateModerationResponse},
};

pub(super) fn routes(router: Router<ServerState>) -> Router<ServerState> {
    router
        .route("/v1/chat/completions", post(chat_completions))
        .route("/v1/completions", post(completions))
        .route("/v1/embeddings", post(embeddings))
        .route("/v1/moderations", post(moderations))
}

// ---- chat completions ------------------------------------------------------------------------

/// Publishes `timings` for request `id` (when the event middleware named it) on `events`.
fn publish_stats(
    events: &Events,
    id: Option<&RequestId>,
    model: &str,
    timings: &teeny_llm::Timings,
) -> Timings {
    let wire = Timings::from(timings);
    if let Some(RequestId(id)) = id {
        events.publish(ServerEvent::RequestStats {
            id: id.clone(),
            model: model.to_owned(),
            timings: wire.clone(),
        });
    }
    wire
}

async fn chat_completions(
    State(engine): State<AppState>,
    State(events): State<Events>,
    request_id: Option<Extension<RequestId>>,
    body: Result<Json<CreateChatCompletionRequest>, JsonRejection>,
) -> Result<Response, ApiError> {
    let Json(req) = body?;
    let stream = req.streamed();
    let request = req.into_core()?;
    // The canonical id, whatever tag the request named the model by.
    let model = engine.model(&request.model)?.id;
    let (n, include_usage) = (request.n, request.include_usage);
    let logprobs = request.logprobs.is_some();
    let stream_events = engine.chat_stream(request)?;
    let id = new_id("chatcmpl");
    let created = unix_now();

    let request_id = request_id.map(|Extension(id)| id);
    if !stream {
        let done = chat::collect(&model, n, stream_events).await?;
        let timings = done
            .timings
            .as_ref()
            .map(|t| publish_stats(&events, request_id.as_ref(), &model, t));
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
            timings,
        })
        .into_response());
    }

    let stats_model = model.clone();
    let chunk = move |choices: Vec<ChatCompletionStreamChoice>,
                      usage: Option<Usage>,
                      timings: Option<Timings>| {
        sse_event(&CreateChatCompletionStreamResponse {
            id: id.clone(),
            object: "chat.completion.chunk".to_owned(),
            created,
            model: model.clone(),
            choices,
            usage,
            timings,
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
            chunk(vec![choice(index, delta, None, None)], None, None)
        })
        .collect();
    // The timings wait for the usage chunk, so a stream never sends a chunk without
    // choices that its client didn't ask for.
    let mut timings: Option<Timings> = None;
    let rest = stream_events.filter_map(move |item| {
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
                None,
            )),
            Ok(ChatEvent::Timings(t)) => {
                timings = Some(publish_stats(
                    &events,
                    request_id.as_ref(),
                    &stats_model,
                    &t,
                ));
                None
            }
            Ok(ChatEvent::Usage(u)) => {
                include_usage.then(|| chunk(Vec::new(), Some(u.into()), timings.take()))
            }
        };
        std::future::ready(event)
    });
    Ok(sse_response(stream::iter(opening).chain(rest)))
}

// ---- legacy completions ----------------------------------------------------------------------

async fn completions(
    State(engine): State<AppState>,
    State(events): State<Events>,
    request_id: Option<Extension<RequestId>>,
    body: Result<Json<CreateCompletionRequest>, JsonRejection>,
) -> Result<Response, ApiError> {
    let Json(req) = body?;
    let stream = req.streamed();
    let request = req.into_core()?;
    let model = engine.model(&request.model)?.id;
    let choices = request.prompts.len() * request.n as usize;
    let (include_usage, logprobs) = (request.include_usage, request.logprobs.is_some());
    let stream_events = engine.complete_stream(request).await?;
    let request_id = request_id.map(|Extension(id)| id);
    let id = new_id("cmpl");
    let created = unix_now();
    let stats_model = model.clone();
    let response = move |choices, usage, timings| CreateCompletionResponse {
        id: id.clone(),
        object: "text_completion".to_owned(),
        created,
        model: model.clone(),
        choices,
        usage,
        timings,
    };

    if !stream {
        let done = completion::collect(choices, stream_events).await?;
        let timings = done
            .timings
            .as_ref()
            .map(|t| publish_stats(&events, request_id.as_ref(), &stats_model, t));
        let choices = done
            .choices
            .into_iter()
            .map(|c| CompletionChoice::from_core(c, logprobs))
            .collect();
        return Ok(Json(response(choices, Some(done.usage.into()), timings)).into_response());
    }

    let mut timings: Option<Timings> = None;
    let out = stream_events.filter_map(move |item| {
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
                Some(sse_event(&response(vec![choice], None, None)))
            }
            Ok(CompletionEvent::Finish { index, reason }) => {
                let choice = CompletionChoice {
                    index,
                    text: String::new(),
                    logprobs: None,
                    finish_reason: Some(reason.into()),
                };
                Some(sse_event(&response(vec![choice], None, None)))
            }
            Ok(CompletionEvent::Timings(t)) => {
                timings = Some(publish_stats(
                    &events,
                    request_id.as_ref(),
                    &stats_model,
                    &t,
                ));
                None
            }
            Ok(CompletionEvent::Usage(u)) => include_usage
                .then(|| sse_event(&response(Vec::new(), Some(u.into()), timings.take()))),
        };
        std::future::ready(event)
    });
    Ok(sse_response(out))
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

// ---- moderations -----------------------------------------------------------------------------

async fn moderations(
    State(engine): State<AppState>,
    body: Result<Json<CreateModerationRequest>, JsonRejection>,
) -> Result<Json<CreateModerationResponse>, ApiError> {
    let Json(req) = body?;
    let out = engine.moderate(req.into_core()).await?;
    Ok(Json(CreateModerationResponse::from_core(
        new_id("modr"),
        &out,
    )))
}
