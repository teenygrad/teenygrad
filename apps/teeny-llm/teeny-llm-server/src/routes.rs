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
use teeny_llm_core::chat::{self, ChatEvent};
use teeny_llm_core::{Engine, GenerationRequest, Prompt, SamplingParams};

use crate::openai::{
    FinishReason, OneOrMany, StreamOptions, Usage,
    chat::{
        self as chat_wire, ChatCompletionStreamChoice, ChatCompletionStreamResponseDelta,
        CreateChatCompletionRequest, CreateChatCompletionResponse,
        CreateChatCompletionStreamResponse, Role, choice_from_core, delta_from_core,
    },
    completions::{CompletionChoice, CreateCompletionRequest, CreateCompletionResponse},
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

fn include_usage(opts: Option<&StreamOptions>) -> bool {
    opts.and_then(|o| o.include_usage).unwrap_or(false)
}

fn require_single_choice(n: Option<u32>) -> Result<(), ApiError> {
    match n {
        None | Some(1) => Ok(()),
        Some(n) => Err(ApiError::invalid_request(
            format!("n={n} is not supported yet; only n=1 is"),
            Some("n"),
        )),
    }
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
    require_single_choice(req.n)?;
    if req.echo == Some(true) {
        return Err(ApiError::invalid_request(
            "`echo: true` is not supported yet",
            Some("echo"),
        ));
    }
    if req.suffix.is_some() {
        return Err(ApiError::invalid_request(
            "`suffix` is not supported yet",
            Some("suffix"),
        ));
    }

    let stream = req.stream.unwrap_or(false);
    let include_usage = include_usage(req.stream_options.as_ref());
    let sampling = SamplingParams {
        max_tokens: req.max_tokens,
        temperature: req.temperature,
        top_p: req.top_p,
        presence_penalty: req.presence_penalty,
        frequency_penalty: req.frequency_penalty,
        seed: req.seed.map(i64::cast_unsigned),
        stop: req.stop.map(OneOrMany::into_vec).unwrap_or_default(),
    };
    let prompts = req.prompt.into_vec();
    if prompts.is_empty() {
        return Err(ApiError::invalid_request(
            "`prompt` must not be an empty array",
            Some("prompt"),
        ));
    }
    let model = req.model;
    let request = |prompt: String| GenerationRequest {
        model: model.clone(),
        prompt: Prompt::Text(prompt),
        sampling: sampling.clone(),
    };
    let id = new_id("cmpl");
    let created = unix_now();
    let response_model = model.clone();
    let response = move |choices, usage| CreateCompletionResponse {
        id: id.clone(),
        object: "text_completion".to_owned(),
        created,
        model: response_model.clone(),
        choices,
        usage,
    };

    if !stream {
        let mut choices = Vec::with_capacity(prompts.len());
        let mut usage = teeny_llm_core::Usage::default();
        for (index, prompt) in prompts.into_iter().enumerate() {
            let out = engine.generate(request(prompt)).await?;
            usage.prompt_tokens += out.usage.prompt_tokens;
            usage.completion_tokens += out.usage.completion_tokens;
            choices.push(CompletionChoice {
                index: u32::try_from(index).unwrap_or(u32::MAX),
                text: out.text,
                logprobs: None,
                finish_reason: Some(out.finish_reason.into()),
            });
        }
        return Ok(Json(response(choices, Some(usage.into()))).into_response());
    }

    let [prompt] = <[String; 1]>::try_from(prompts).map_err(|p| {
        ApiError::invalid_request(
            format!(
                "streaming supports a single prompt; got {} prompts",
                p.len()
            ),
            Some("prompt"),
        )
    })?;
    let tokens = engine.generate_stream(request(prompt))?;
    let events = tokens.flat_map(move |item| {
        let events = match item {
            Err(e) => vec![sse_error(e)],
            Ok(chunk) => {
                let mut events = Vec::with_capacity(2);
                if !chunk.text.is_empty() || chunk.finish_reason.is_some() {
                    let choice = CompletionChoice {
                        index: 0,
                        text: chunk.text,
                        logprobs: None,
                        finish_reason: chunk.finish_reason.map(Into::into),
                    };
                    events.push(sse_event(&response(vec![choice], None)));
                }
                if chunk.finish_reason.is_some() && include_usage {
                    let usage = chunk.usage.unwrap_or_default().into();
                    events.push(sse_event(&response(Vec::new(), Some(usage))));
                }
                events
            }
        };
        stream::iter(events)
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
