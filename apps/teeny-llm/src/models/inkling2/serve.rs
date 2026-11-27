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

//! Inkling-2 as a [`ChatModel`]: [`Model::load`](crate::models::Model::load) starts a
//! worker thread that owns the device and the [`Engine`], and [`Inkling2Chat`] sends it
//! requests and streams the replies back.
//!
//! The engine borrows its device, so it never leaves the worker. Requests run one at a
//! time, in the order they arrive. A request that extends the cached conversation only
//! processes its new tokens; any other starts the cache again, since the engine can only
//! take back its latest step.

use std::sync::Arc;
use std::sync::mpsc;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use futures::channel::{mpsc as async_mpsc, oneshot};
use futures::future::BoxFuture;
use serde_json::{Value, json};
use teeny_test::model::runtime;

use super::Inkling2Config;
use super::chat::Chat;
use super::engine::{Engine, MAX_TOKENS};
use super::weights::Checkpoint;
use crate::chat::{
    Capabilities, ChatEvent, ChatFinishReason, ChatModel, ChatRequest, ChatStream, ChatUsage,
    ContentPart, Delta, Message, ReasoningEffort,
};
use crate::models::{LoadOptions, LoadedModel, Model};
use crate::timings::Recorder;
use crate::tokenizer::{HfTokenizer, Tokenizer};
use crate::{Error, ModelInfo, Result};

/// The KV cache size when [`LoadOptions::context_length`] is not given: the model's
/// sliding window, within which every layer sees the whole conversation.
pub const DEFAULT_CONTEXT: u32 = 1024;

/// The worker thread's stack: compiling the layer graphs recurses more deeply than a
/// spawned thread's default 2 MiB allows for.
const WORKER_STACK: usize = 32 * 1024 * 1024;

/// Loads Inkling-2 for [`Model::load`]: the device, every layer graph and the weights, on
/// a worker thread that then serves chat requests.
///
/// `model_dir` must be a checkpoint converted by `inkling2_convert`; the original
/// checkpoint loads too, but converts on every start.
pub(crate) fn load(model: Model, options: &LoadOptions) -> Result<LoadedModel> {
    let dir = model.locate(options.model_dir.as_deref())?;
    let parameters = model.parameter_count(&dir)?;
    let context = options.context_length_for(model, DEFAULT_CONTEXT)?;
    let tokenizer: Arc<dyn Tokenizer> =
        Arc::new(HfTokenizer::from_file(&dir.join("tokenizer.json"))?);

    let (jobs, inbox) = mpsc::channel::<Job>();
    let (ready_tx, ready) = mpsc::sync_channel::<Result<()>>(1);
    let progress = options.clone();
    std::thread::Builder::new()
        .name(format!("{model}"))
        .stack_size(WORKER_STACK)
        .spawn(move || {
            let device = match runtime::open() {
                Ok(d) => d,
                Err(e) => {
                    let _ = ready_tx.send(Err(Error::Inference(format!(
                        "opening the {} device: {e:#}",
                        runtime::BACKEND_NAME
                    ))));
                    return;
                }
            };
            let mut worker = match Worker::start(&device, &dir, context as usize, &progress) {
                Ok(w) => w,
                Err(e) => {
                    let _ = ready_tx.send(Err(e));
                    return;
                }
            };
            let _ = ready_tx.send(Ok(()));
            // Runs until every handle is dropped, which closes the channel.
            for job in inbox {
                worker.run(&device, job);
            }
        })
        .map_err(|e| Error::Inference(format!("starting the {model} worker thread: {e}")))?;
    ready.recv().map_err(|_| {
        Error::Inference(format!(
            "the {model} worker stopped while loading (it panicked)"
        ))
    })??;
    options.report("ready");

    let chat = Arc::new(Inkling2Chat {
        info: ModelInfo {
            id: model.tag().to_owned(),
            created: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map_or(0, |d| d.as_secs()),
            owned_by: "thinkingmachines".to_owned(),
        },
        capabilities: capabilities(context),
        jobs,
    });
    Ok(LoadedModel::from_chat(chat)
        .with_model(model)
        .with_parameters(parameters)
        .with_tokenizer(tokenizer))
}

/// What this Inkling-2 build does: text in and out, greedy decoding, one choice, and
/// the chat template's reasoning effort.
pub fn capabilities(context_length: u32) -> Capabilities {
    Capabilities {
        context_length,
        max_choices: 1,
        reasoning: true,
        ..Capabilities::default()
    }
}

/// Inkling-2, loaded and served by its worker thread.
pub struct Inkling2Chat {
    info: ModelInfo,
    capabilities: Capabilities,
    jobs: mpsc::Sender<Job>,
}

/// A request for the worker.
struct Job {
    /// When the request was accepted, so its timings include waiting for the worker.
    accepted: Instant,
    messages: Value,
    effort: Option<&'static str>,
    max_tokens: Option<u32>,
    started: oneshot::Sender<Result<()>>,
    events: async_mpsc::UnboundedSender<Result<ChatEvent>>,
}

impl ChatModel for Inkling2Chat {
    fn info(&self) -> ModelInfo {
        self.info.clone()
    }

    fn capabilities(&self) -> Capabilities {
        self.capabilities.clone()
    }

    fn chat(&self, request: ChatRequest) -> BoxFuture<'_, Result<ChatStream>> {
        Box::pin(async move {
            let messages = template_messages(&request)?;
            let (started, on_start) = oneshot::channel();
            let (events, stream) = async_mpsc::unbounded();
            self.jobs
                .send(Job {
                    accepted: Instant::now(),
                    messages,
                    effort: request.reasoning_effort.map(effort_name),
                    max_tokens: request.max_completion_tokens,
                    started,
                    events,
                })
                .map_err(|_| self.stopped())?;
            on_start.await.map_err(|_| self.stopped())??;
            let stream: ChatStream = Box::pin(stream);
            Ok(stream)
        })
    }
}

impl Inkling2Chat {
    fn stopped(&self) -> Error {
        Error::Inference(format!(
            "the {} worker thread has stopped (it panicked); reload the model",
            self.info.id
        ))
    }
}

/// The name the chat template knows a reasoning effort by.
fn effort_name(effort: ReasoningEffort) -> &'static str {
    match effort {
        ReasoningEffort::None => "none",
        ReasoningEffort::Minimal => "minimal",
        ReasoningEffort::Low => "low",
        ReasoningEffort::Medium => "medium",
        ReasoningEffort::High => "high",
    }
}

/// The chat template's `messages` for `request`, as OpenAI-style JSON.
fn template_messages(request: &ChatRequest) -> Result<Value> {
    let mut messages = Vec::with_capacity(request.messages.len());
    for (i, m) in request.messages.iter().enumerate() {
        let text = |parts: &[ContentPart]| -> Result<String> {
            parts
                .iter()
                .map(|p| match p {
                    ContentPart::Text(t) => Ok(t.as_str()),
                    _ => Err(Error::UnsupportedParameter {
                        model: request.model.clone(),
                        parameter: format!("messages[{i}].content"),
                        detail: "only text content is supported".into(),
                    }),
                })
                .collect()
        };
        messages.push(match m {
            Message::Developer { content, .. } | Message::System { content, .. } => {
                json!({"role": "system", "content": content})
            }
            Message::User { content, .. } => json!({"role": "user", "content": text(content)?}),
            Message::Assistant(a) => {
                json!({"role": "assistant", "content": a.content.clone().unwrap_or_default()})
            }
            _ => {
                return Err(Error::UnsupportedParameter {
                    model: request.model.clone(),
                    parameter: format!("messages[{i}].role"),
                    detail: format!("{} messages are not supported", m.role()),
                });
            }
        });
    }
    Ok(Value::Array(messages))
}

/// The parts of a loaded model that live on its worker thread.
struct Worker<'d> {
    chat: Chat,
    engine: Engine<'d>,
    capacity: usize,
}

impl<'d> Worker<'d> {
    /// Reads the config, template and weights and compiles the engine.
    fn start(
        device: &runtime::Device<'d>,
        dir: &std::path::Path,
        capacity: usize,
        options: &LoadOptions,
    ) -> Result<Self> {
        let weights = |what: &str, e: anyhow::Error| Error::Weights(format!("{what}: {e:#}"));
        let chat = Chat::load(dir).map_err(|e| weights("reading the chat template", e))?;
        let cfg = Inkling2Config::load(&dir.join("config.json"))
            .map_err(|e| weights("reading config.json", e))?
            .text_config;
        let mut ck = Checkpoint::open(dir).map_err(|e| weights("opening the checkpoint", e))?;
        options.report(&format!(
            "compiling and loading {} layers",
            cfg.num_hidden_layers
        ));
        // The reference applies no muP scaling to the logits, so the multiplier is 1.
        let engine = Engine::load(device, cfg, 1.0, &mut ck, capacity)
            .map_err(|e| weights("loading the engine", e))?;
        Ok(Self {
            chat,
            engine,
            capacity,
        })
    }

    /// Answers `job`, streaming its events. A client that goes away stops the generation.
    fn run(&mut self, device: &runtime::Device<'d>, job: Job) {
        let Job {
            accepted,
            messages,
            effort,
            max_tokens,
            started,
            events,
        } = job;
        let prompt = match self.prompt(&messages, effort) {
            Ok(p) => p,
            Err(e) => {
                let _ = started.send(Err(e));
                return;
            }
        };
        // Keep the cache only when the prompt extends it, with at least one new token to
        // run for the next logits.
        let shared = self
            .engine
            .tokens
            .iter()
            .zip(&prompt)
            .take_while(|(a, b)| a == b)
            .count();
        if shared < self.engine.tokens.len() || shared == prompt.len() {
            self.engine.reset();
        }
        let cached = self.engine.len();

        let mut timings = Recorder::accepted_at(accepted);
        timings.begin_prompt();
        let mut logits = Vec::new();
        for tokens in prompt[cached..].chunks(MAX_TOKENS) {
            logits = match self.engine.step_many(device, tokens) {
                Ok(l) => l,
                Err(e) => {
                    self.engine.reset();
                    let _ = started.send(Err(Error::Inference(format!("{e:#}"))));
                    return;
                }
            };
        }
        timings.end_prompt(prompt.len() - cached, cached);
        if started.send(Ok(())).is_err() {
            return;
        }

        let room = self.capacity - self.engine.len();
        let limit = max_tokens.map_or(room, |n| (n as usize).min(room));
        let send = |e: Result<ChatEvent>| events.unbounded_send(e).is_ok();
        let mut out: Vec<u32> = Vec::new();
        let mut shown = 0usize;
        let outcome = loop {
            let tok = self.engine.pick(&logits);
            timings.token();
            out.push(tok);
            // Decoded whole each time, as a token can end partway through a character.
            let text = match self.chat.decode_visible(&out) {
                Ok(t) => t,
                Err(e) => break Err(Error::Tokenizer(format!("{e:#}"))),
            };
            if text.len() > shown && text.is_char_boundary(shown) {
                let delta = ChatEvent::Delta {
                    index: 0,
                    delta: Delta {
                        content: Some(text[shown..].to_owned()),
                        ..Delta::default()
                    },
                };
                if !send(Ok(delta)) {
                    // The client went away: stop generating.
                    return;
                }
                shown = text.len();
            }
            if tok == self.chat.eos {
                break Ok(ChatFinishReason::Stop);
            }
            if out.len() >= limit {
                break Ok(ChatFinishReason::Length);
            }
            logits = match self.engine.step(device, tok) {
                Ok(l) => l,
                Err(e) => break Err(Error::Inference(format!("{e:#}"))),
            };
        };
        match outcome {
            Ok(reason) => {
                send(Ok(ChatEvent::Finish { index: 0, reason }));
                send(Ok(ChatEvent::Timings(timings.finish())));
                send(Ok(ChatEvent::Usage(ChatUsage {
                    prompt_tokens: count(prompt.len()),
                    completion_tokens: count(out.len()),
                    cached_prompt_tokens: count(cached),
                    ..ChatUsage::default()
                })));
            }
            Err(e) => {
                // The cache holds a partial reply nobody will extend.
                self.engine.reset();
                send(Err(e));
            }
        }
    }

    /// The prompt's token ids, refused if it leaves no room in the context for a reply.
    fn prompt(&self, messages: &Value, effort: Option<&str>) -> Result<Vec<u32>> {
        let prompt = self
            .chat
            .prompt(messages, effort)
            .map_err(|e| Error::InvalidRequest(format!("rendering the chat template: {e:#}")))?;
        if prompt.len() >= self.capacity {
            return Err(Error::InvalidRequest(format!(
                "the conversation is {} tokens, which leaves no room for a reply in the \
                 model's {}-token context; shorten it or load the model with a longer context",
                prompt.len(),
                self.capacity
            )));
        }
        Ok(prompt)
    }
}

fn count(n: usize) -> u32 {
    u32::try_from(n).unwrap_or(u32::MAX)
}
