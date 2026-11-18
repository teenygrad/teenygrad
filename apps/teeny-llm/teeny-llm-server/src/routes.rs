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
use teeny_llm_core::{
    ChatMessage, Engine, GenerationChunk, GenerationRequest, Prompt, SamplingParams,
};

use crate::openai::{
    FinishReason, OneOrMany, StreamOptions, Usage,
    chat::{
        ChatCompletionChoice, ChatCompletionRequestMessage, ChatCompletionResponseMessage,
        ChatCompletionStreamChoice, ChatCompletionStreamResponseDelta, CreateChatCompletionRequest,
        CreateChatCompletionResponse, CreateChatCompletionStreamResponse, MessageContent, Role,
    },
    completions::{CompletionChoice, CreateCompletionRequest, CreateCompletionResponse},
    embeddings::{CreateEmbeddingRequest, CreateEmbeddingResponse, Embedding, EmbeddingUsage},
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

fn message_text(
    index: usize,
    message: ChatCompletionRequestMessage,
) -> Result<ChatMessage, ApiError> {
    let content = match message.content {
        None => String::new(),
        Some(MessageContent::Text(text)) => text,
        Some(MessageContent::Parts(parts)) => {
            let mut text = String::new();
            for (p, part) in parts.into_iter().enumerate() {
                match (part.kind.as_str(), part.text) {
                    ("text", Some(t)) => text.push_str(&t),
                    ("text", None) => {
                        return Err(ApiError::invalid_request(
                            "text content part is missing `text`",
                            Some(&format!("messages[{index}].content[{p}].text")),
                        ));
                    }
                    (kind, _) => {
                        return Err(ApiError::invalid_request(
                            format!(
                                "content part type `{kind}` is not supported yet; only `text` is"
                            ),
                            Some(&format!("messages[{index}].content[{p}].type")),
                        ));
                    }
                }
            }
            text
        }
    };
    Ok(ChatMessage::new(message.role.into(), content))
}

fn chat_generation_request(
    req: CreateChatCompletionRequest,
) -> Result<GenerationRequest, ApiError> {
    require_single_choice(req.n)?;
    if req.messages.is_empty() {
        return Err(ApiError::invalid_request(
            "`messages` must contain at least one message",
            Some("messages"),
        ));
    }
    let messages = req
        .messages
        .into_iter()
        .enumerate()
        .map(|(i, m)| message_text(i, m))
        .collect::<Result<Vec<_>, _>>()?;
    Ok(GenerationRequest {
        model: req.model,
        prompt: Prompt::Chat(messages),
        sampling: SamplingParams {
            max_tokens: req.max_completion_tokens.or(req.max_tokens),
            temperature: req.temperature,
            top_p: req.top_p,
            presence_penalty: req.presence_penalty,
            frequency_penalty: req.frequency_penalty,
            seed: req.seed.map(i64::cast_unsigned),
            stop: req.stop.map(OneOrMany::into_vec).unwrap_or_default(),
        },
    })
}

async fn chat_completions(
    State(engine): State<AppState>,
    body: Result<Json<CreateChatCompletionRequest>, JsonRejection>,
) -> Result<Response, ApiError> {
    let Json(req) = body?;
    let stream = req.stream.unwrap_or(false);
    let include_usage = include_usage(req.stream_options.as_ref());
    let generation = chat_generation_request(req)?;
    let model = generation.model.clone();
    let id = new_id("chatcmpl");
    let created = unix_now();

    if !stream {
        let out = engine.generate(generation).await?;
        return Ok(Json(CreateChatCompletionResponse {
            id,
            object: "chat.completion".to_owned(),
            created,
            model,
            choices: vec![ChatCompletionChoice {
                index: 0,
                message: ChatCompletionResponseMessage {
                    role: Role::Assistant,
                    content: Some(out.text),
                    refusal: None,
                },
                logprobs: None,
                finish_reason: out.finish_reason.into(),
            }],
            usage: Some(out.usage.into()),
        })
        .into_response());
    }

    let tokens = engine.generate_stream(generation)?;
    let chunk = move |choices: Vec<ChatCompletionStreamChoice>, usage: Option<Usage>| {
        CreateChatCompletionStreamResponse {
            id: id.clone(),
            object: "chat.completion.chunk".to_owned(),
            created,
            model: model.clone(),
            choices,
            usage,
        }
    };
    let choice = |delta, finish_reason| ChatCompletionStreamChoice {
        index: 0,
        delta,
        logprobs: None,
        finish_reason,
    };

    let first = sse_event(&chunk(
        vec![choice(
            ChatCompletionStreamResponseDelta {
                role: Some(Role::Assistant),
                content: Some(String::new()),
            },
            None,
        )],
        None,
    ));
    let rest = tokens.flat_map(move |item| {
        let events = match item {
            Err(e) => vec![sse_error(e)],
            Ok(GenerationChunk {
                text,
                finish_reason,
                usage,
            }) => {
                let mut events = Vec::with_capacity(3);
                if !text.is_empty() {
                    let delta = ChatCompletionStreamResponseDelta {
                        role: None,
                        content: Some(text),
                    };
                    events.push(sse_event(&chunk(vec![choice(delta, None)], None)));
                }
                if let Some(reason) = finish_reason {
                    let delta = ChatCompletionStreamResponseDelta::default();
                    events.push(sse_event(&chunk(
                        vec![choice(delta, Some(FinishReason::from(reason)))],
                        None,
                    )));
                    if include_usage {
                        let usage = usage.unwrap_or_default().into();
                        events.push(sse_event(&chunk(Vec::new(), Some(usage))));
                    }
                }
                events
            }
        };
        stream::iter(events)
    });
    Ok(sse_response(stream::once(async { first }).chain(rest)))
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
    if let Some(format) = req.encoding_format.as_deref().filter(|f| *f != "float") {
        return Err(ApiError::invalid_request(
            format!("encoding_format `{format}` is not supported yet; only `float` is"),
            Some("encoding_format"),
        ));
    }
    if req.dimensions.is_some() {
        return Err(ApiError::invalid_request(
            "`dimensions` is not supported yet",
            Some("dimensions"),
        ));
    }
    let inputs = req.input.into_vec();
    let out = engine.embed(&req.model, &inputs)?;
    Ok(Json(CreateEmbeddingResponse {
        object: "list".to_owned(),
        data: out
            .vectors
            .into_iter()
            .enumerate()
            .map(|(i, embedding)| Embedding {
                object: "embedding".to_owned(),
                index: u32::try_from(i).unwrap_or(u32::MAX),
                embedding,
            })
            .collect(),
        model: req.model,
        usage: EmbeddingUsage {
            prompt_tokens: out.prompt_tokens,
            total_tokens: out.prompt_tokens,
        },
    }))
}
