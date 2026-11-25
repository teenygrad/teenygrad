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

//! FunctionGemma as a [`ChatModel`]: [`Model::load`](crate::models::Model::load) starts a worker thread that owns the
//! device, the compiled step graphs and the KV cache, and [`FunctionGemmaChat`] sends it
//! requests and streams the replies back.
//!
//! The runner is synchronous and borrows its device (`CompiledStep<'d>`), so it never
//! leaves the worker. Requests run one at a time, in the order they arrive, and the KV
//! cache is kept between them: a conversation re-sent with one more turn only processes
//! the new turn.

use std::collections::HashMap;
use std::sync::Arc;
use std::sync::mpsc;
use std::time::{SystemTime, UNIX_EPOCH};

use futures::channel::{mpsc as async_mpsc, oneshot};
use futures::future::BoxFuture;
use serde_json::{Value, json};
use teeny_test::model::runtime;

use super::GemmaConfig;
use super::chat::{Chat, parse_calls};
use super::runner::{CompiledStep, Session, Weights, argmax};
use crate::chat::{
    Capabilities, ChatEvent, ChatFinishReason, ChatModel, ChatRequest, ChatStream, ChatUsage,
    ContentPart, Delta, Message, ToolCallDelta, ToolCallKind, ToolChoice, ToolKind, ToolSupport,
};
use crate::models::{LoadOptions, LoadedModel, Model};
use crate::timings::Recorder;
use crate::tokenizer::{HfTokenizer, Tokenizer};
use crate::{Error, ModelInfo, Result};

/// The KV cache size when [`LoadOptions::context_length`] is not given.
pub const DEFAULT_CONTEXT: u32 = 1024;

/// Tokens per prompt-processing step: prompts run through a step compiled for this many
/// tokens while a whole chunk remains, then a token at a time.
const CHUNK: usize = 16;

/// Token ids that end a reply besides the config's EOS: `<eos>`,
/// `<start_function_response>` (the model waits for a tool's result) and `<end_of_turn>`.
const STOP_IDS: [u32; 3] = [1, 50, 106];

/// The worker thread's stack: compiling step graphs recurses more deeply than a spawned
/// thread's default 2 MiB allows for.
const WORKER_STACK: usize = 32 * 1024 * 1024;

/// Loads FunctionGemma for [`Model::load`]: the weights, the device and both step graphs,
/// on a worker thread that then serves chat requests.
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
            let worker = Worker::start(&dir, context as usize, &progress);
            let (device, mut parts) = match worker {
                Ok(w) => w,
                Err(e) => {
                    let _ = ready_tx.send(Err(e));
                    return;
                }
            };
            let compiled = parts.compile(&device, &progress);
            let (mut chunk, mut one) = match compiled {
                Ok(steps) => steps,
                Err(e) => {
                    let _ = ready_tx.send(Err(e));
                    return;
                }
            };
            drop(parts.weights.take());
            let _ = ready_tx.send(Ok(()));
            // Runs until every handle is dropped, which closes the channel.
            for job in inbox {
                parts.run(&device, &mut chunk, &mut one, job);
            }
        })
        .map_err(|e| Error::Inference(format!("starting the {model} worker thread: {e}")))?;
    ready.recv().map_err(|_| {
        Error::Inference(format!(
            "the {model} worker stopped while loading (it panicked)"
        ))
    })??;
    options.report("ready");

    let chat = Arc::new(FunctionGemmaChat {
        info: ModelInfo {
            id: model.tag().to_owned(),
            created: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map_or(0, |d| d.as_secs()),
            owned_by: "google".to_owned(),
        },
        capabilities: capabilities(context),
        jobs,
    });
    Ok(LoadedModel::from_chat(chat)
        .with_model(model)
        .with_parameters(parameters)
        .with_tokenizer(tokenizer))
}

/// What FunctionGemma does: text in and out with function calling (several calls a
/// turn, never forced: it decides for itself), stop strings, greedy decoding, one choice.
pub fn capabilities(context_length: u32) -> Capabilities {
    Capabilities {
        context_length,
        tools: ToolSupport {
            functions: true,
            custom: false,
            parallel: true,
            forced: false,
            strict: false,
        },
        stop_strings: true,
        max_choices: 1,
        ..Capabilities::default()
    }
}

/// FunctionGemma, loaded and served by its worker thread.
pub struct FunctionGemmaChat {
    info: ModelInfo,
    capabilities: Capabilities,
    jobs: mpsc::Sender<Job>,
}

/// A request for the worker: the chat template's inputs, how to shape the reply, and
/// where to send it.
struct Job {
    /// When the request was accepted, so its timings include waiting for the worker.
    accepted: std::time::Instant,
    messages: Value,
    tools: Value,
    plan: ReplyPlan,
    started: oneshot::Sender<Result<()>>,
    events: async_mpsc::UnboundedSender<Result<ChatEvent>>,
}

impl ChatModel for FunctionGemmaChat {
    fn info(&self) -> ModelInfo {
        self.info.clone()
    }

    fn capabilities(&self) -> Capabilities {
        self.capabilities.clone()
    }

    fn chat(&self, request: ChatRequest) -> BoxFuture<'_, Result<ChatStream>> {
        Box::pin(async move {
            let (messages, tools) = template_inputs(&request)?;
            let plan = ReplyPlan::for_request(&request);
            let (started, on_start) = oneshot::channel();
            let (events, stream) = async_mpsc::unbounded();
            self.jobs
                .send(Job {
                    accepted: std::time::Instant::now(),
                    messages,
                    tools,
                    plan,
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

impl FunctionGemmaChat {
    fn stopped(&self) -> Error {
        Error::Inference(format!(
            "the {} worker thread has stopped (it panicked); reload the model",
            self.info.id
        ))
    }
}

/// The parts of a loaded model that live on its worker thread.
struct Worker {
    cfg: GemmaConfig,
    chat: Chat,
    session: Session,
    weights: Option<Weights>,
    stop: Vec<u32>,
}

fn weights_error(what: &str, e: anyhow::Error) -> Error {
    Error::Weights(format!("{what}: {e:#}"))
}

fn inference_error(e: anyhow::Error) -> Error {
    Error::Inference(format!("{e:#}"))
}

impl Worker {
    /// Reads the config, template and weights and opens the device.
    fn start<'d>(
        dir: &std::path::Path,
        context: usize,
        options: &LoadOptions,
    ) -> Result<(runtime::Device<'d>, Self)> {
        let cfg = GemmaConfig::load(&dir.join("config.json"))
            .map_err(|e| weights_error("reading config.json", e))?;
        let chat = Chat::load(dir).map_err(|e| weights_error("reading the chat template", e))?;
        options.report("loading weights");
        let weights = Weights::load(&dir.join("model.safetensors"))
            .map_err(|e| weights_error("reading model.safetensors", e))?;
        options.report(&format!("opening the {} device", runtime::BACKEND_NAME));
        let device = runtime::open().map_err(|e| {
            Error::Inference(format!(
                "opening the {} device: {e:#}",
                runtime::BACKEND_NAME
            ))
        })?;
        let mut stop = cfg.eos_token_id.ids();
        stop.extend(STOP_IDS);
        let session = Session::new(cfg.clone(), context);
        Ok((
            device,
            Self {
                cfg,
                chat,
                session,
                weights: Some(weights),
                stop,
            },
        ))
    }

    /// Compiles the chunked and one-token step graphs and loads the weights into them.
    fn compile<'d>(
        &self,
        device: &runtime::Device<'d>,
        options: &LoadOptions,
    ) -> Result<(CompiledStep<'d>, CompiledStep<'d>)> {
        let weights = self
            .weights
            .as_ref()
            .ok_or_else(|| Error::Weights("the weights were already released".into()))?;
        let capacity = self.session.cache.capacity;
        let compile = |tokens: usize| {
            options.report(&format!("compiling the {tokens}-token step"));
            CompiledStep::new(
                device,
                &self.cfg,
                weights,
                tokens,
                capacity,
                &self.session.past,
            )
            .map_err(|e| Error::Inference(format!("compiling the {tokens}-token step: {e:#}")))
        };
        let chunk = compile(CHUNK)?;
        let one = compile(1)?;
        Ok((chunk, one))
    }

    /// Answers `job`, streaming its events. A client that goes away stops the generation.
    fn run<'d>(
        &mut self,
        device: &runtime::Device<'d>,
        chunk: &mut CompiledStep<'d>,
        one: &mut CompiledStep<'d>,
        job: Job,
    ) {
        let Job {
            accepted,
            messages,
            tools,
            plan,
            started,
            events,
        } = job;
        let prompt = match self.prompt(&messages, &tools) {
            Ok(p) => p,
            Err(e) => {
                let _ = started.send(Err(e));
                return;
            }
        };
        let cached = self
            .session
            .tokens
            .iter()
            .zip(&prompt)
            .take_while(|(a, b)| a == b)
            .count()
            .min(prompt.len() - 1);
        let mut timings = Recorder::accepted_at(accepted);
        timings.begin_prompt();
        let mut logits = match self.session.feed(device, chunk, one, &prompt) {
            Ok(l) => l,
            Err(e) => {
                let _ = started.send(Err(inference_error(e)));
                return;
            }
        };
        timings.end_prompt(prompt.len() - cached, cached);
        if started.send(Ok(())).is_err() {
            return;
        }

        let room = self.session.cache.capacity - self.session.cache.len;
        let limit = plan.max_tokens.map_or(room, |n| n.min(room));
        let mut reply = ReplyAssembler::new(plan);
        let mut decoder = self.chat.stream_decoder();
        let mut generated = 0usize;
        let send = |e: Result<ChatEvent>| events.unbounded_send(e).is_ok();
        let outcome = loop {
            let tok = argmax(&logits);
            timings.token();
            generated += 1;
            let piece = match decoder.step(tok) {
                Ok(p) => p.unwrap_or_default(),
                Err(e) => break Err(e),
            };
            let (out, done) = match reply.push(&piece) {
                Ok(r) => r,
                Err(e) => break Err(e),
            };
            if !out.into_iter().all(|e| send(Ok(e))) {
                // The client went away: stop generating.
                return;
            }
            if done || self.stop.contains(&tok) {
                break Ok(false);
            }
            if generated >= limit {
                break Ok(true);
            }
            logits = match self.session.step(device, one, &[tok]) {
                Ok(l) => l,
                Err(e) => break Err(inference_error(e)),
            };
        };
        match outcome {
            Ok(hit_limit) => {
                for e in reply.finish(hit_limit) {
                    send(Ok(e));
                }
                send(Ok(ChatEvent::Timings(timings.finish())));
                send(Ok(ChatEvent::Usage(ChatUsage {
                    prompt_tokens: count(prompt.len()),
                    completion_tokens: count(generated),
                    cached_prompt_tokens: count(cached),
                    ..ChatUsage::default()
                })));
            }
            Err(e) => {
                send(Err(e));
            }
        }
    }

    /// The prompt's token ids, refused if it leaves no room in the context for a reply.
    fn prompt(&self, messages: &Value, tools: &Value) -> Result<Vec<u32>> {
        let prompt = self
            .chat
            .prompt(messages, tools)
            .map_err(|e| Error::InvalidRequest(format!("rendering the chat template: {e:#}")))?;
        let capacity = self.session.cache.capacity;
        if prompt.len() >= capacity {
            return Err(Error::InvalidRequest(format!(
                "the conversation is {} tokens, which leaves no room for a reply in the \
                 model's {capacity}-token context; shorten it or load the model with a \
                 longer context",
                prompt.len()
            )));
        }
        Ok(prompt)
    }
}

fn count(n: usize) -> u32 {
    u32::try_from(n).unwrap_or(u32::MAX)
}

/// The chat template's `messages` and `tools` for `request`, as Hugging Face's
/// `apply_chat_template` takes them (OpenAI-style JSON). Tool results name the function
/// they answer, found by their call id among the earlier assistant turns.
pub fn template_inputs(request: &ChatRequest) -> Result<(Value, Value)> {
    let mut call_names: HashMap<&str, &str> = HashMap::new();
    let mut messages = Vec::with_capacity(request.messages.len());
    for (i, m) in request.messages.iter().enumerate() {
        messages.push(match m {
            Message::Developer { content, .. } => json!({"role": "developer", "content": content}),
            Message::System { content, .. } => json!({"role": "system", "content": content}),
            Message::User { content, .. } => {
                let text: String = content
                    .iter()
                    .map(|p| match p {
                        ContentPart::Text(t) => Ok(t.as_str()),
                        other => Err(Error::UnsupportedParameter {
                            model: request.model.clone(),
                            parameter: format!("messages[{i}].content"),
                            detail: format!(
                                "a `{}` part (the model reads text only)",
                                other.kind()
                            ),
                        }),
                    })
                    .collect::<Result<_>>()?;
                json!({"role": "user", "content": text})
            }
            Message::Assistant(a) => {
                for c in &a.tool_calls {
                    call_names.insert(c.id.as_str(), c.name.as_str());
                }
                let mut msg = json!({"role": "assistant"});
                if let Some(text) = &a.content {
                    msg["content"] = json!(text);
                }
                if !a.tool_calls.is_empty() {
                    msg["tool_calls"] = a
                        .tool_calls
                        .iter()
                        .map(|c| {
                            json!({"type": "function", "function": {
                                "name": c.name, "arguments": json_or_text(&c.input)
                            }})
                        })
                        .collect();
                }
                msg
            }
            Message::Tool {
                tool_call_id,
                content,
            } => {
                let name = call_names.get(tool_call_id.as_str()).ok_or_else(|| {
                    Error::InvalidParameter {
                        parameter: format!("messages[{i}].tool_call_id"),
                        detail: format!(
                            "`{tool_call_id}` is not the id of a tool call in an earlier \
                             assistant message"
                        ),
                    }
                })?;
                let result = serde_json::from_str::<Value>(content)
                    .ok()
                    .filter(Value::is_object)
                    .unwrap_or_else(|| Value::String(content.clone()));
                json!({"role": "tool", "name": name, "content": result})
            }
        });
    }
    Ok((Value::Array(messages), Value::Array(offered_tools(request))))
}

/// JSON text as a value, or as a string when it is not JSON.
fn json_or_text(s: &str) -> Value {
    serde_json::from_str(s).unwrap_or_else(|_| Value::String(s.to_owned()))
}

/// The function declarations the model is shown: none under `tool_choice: none`, only the
/// allowed ones under `allowed_tools`.
fn offered_tools(request: &ChatRequest) -> Vec<Value> {
    let allowed: Option<&[String]> = match &request.tool_choice {
        ToolChoice::None => return Vec::new(),
        ToolChoice::Allowed { tools, .. } => Some(tools),
        _ => None,
    };
    request
        .tools
        .iter()
        .filter(|t| allowed.is_none_or(|a| a.contains(&t.name)))
        .filter_map(|t| match &t.kind {
            ToolKind::Function { parameters, .. } => {
                let mut f = json!({"name": t.name});
                if let Some(d) = &t.description {
                    f["description"] = json!(d);
                }
                if let Some(p) = parameters {
                    f["parameters"] = p.clone();
                }
                Some(json!({"type": "function", "function": f}))
            }
            // Refused by validation: FunctionGemma declares no custom tools.
            ToolKind::Custom { .. } => None,
        })
        .collect()
}

/// How a reply is to be shaped, from its request.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct ReplyPlan {
    /// Most tokens to generate; `None` is whatever the context leaves.
    pub max_tokens: Option<usize>,
    /// Strings that end the reply's text, not included in it.
    pub stop: Vec<String>,
    /// Whether to stop after the first function call.
    pub single_call: bool,
    /// The function a call that leaves out its name means: the only one offered.
    pub sole_function: Option<String>,
}

impl ReplyPlan {
    pub fn for_request(request: &ChatRequest) -> Self {
        let offered = offered_tools(request);
        let sole_function = match offered.as_slice() {
            [only] => only["function"]["name"].as_str().map(str::to_owned),
            _ => None,
        };
        Self {
            max_tokens: request.max_completion_tokens.map(|n| n as usize),
            stop: request.sampling.stop.clone(),
            single_call: request.parallel_tool_calls == Some(false),
            sole_function,
        }
    }
}

const CALL_START: &str = "<start_function_call>";
const CALL_END: &str = "<end_function_call>";

/// Turns FunctionGemma's raw output, as it is decoded, into chat events for choice 0: the
/// text before any function call as content deltas, each function call as one tool-call
/// delta once it is complete, then the finish.
#[derive(Debug)]
pub struct ReplyAssembler {
    plan: ReplyPlan,
    raw: String,
    /// The content sent so far.
    sent: String,
    /// Function calls sent so far.
    calls: usize,
    /// Whether a stop string ended the text.
    stopped: bool,
}

impl ReplyAssembler {
    pub fn new(plan: ReplyPlan) -> Self {
        Self {
            plan,
            raw: String::new(),
            sent: String::new(),
            calls: 0,
            stopped: false,
        }
    }

    /// Adds the next decoded piece. Returns the events it completes, and whether the reply
    /// is over (a stop string, or the one call a single-call request allows).
    pub fn push(&mut self, piece: &str) -> Result<(Vec<ChatEvent>, bool)> {
        self.raw.push_str(piece);
        let mut events = Vec::new();
        if !self.stopped {
            let (text, stop_at) = self.speech();
            let end = match stop_at {
                Some(at) => {
                    self.stopped = true;
                    at
                }
                None => self.safe_end(&text),
            };
            self.send_content(&text[..end], &mut events);
            if self.stopped {
                return Ok((events, true));
            }
        }
        let done = self.send_calls(&mut events)?;
        Ok((events, done))
    }

    /// The events that end the reply: content held back in case it began a stop string,
    /// and the finish. `hit_limit` is whether generation ran out of tokens.
    pub fn finish(mut self, hit_limit: bool) -> Vec<ChatEvent> {
        let mut events = Vec::new();
        if !self.stopped {
            let (text, stop_at) = self.speech();
            let end = stop_at.unwrap_or(text.len());
            self.send_content(&text[..end], &mut events);
        }
        let reason = if self.calls > 0 {
            ChatFinishReason::ToolCalls
        } else if hit_limit && !self.stopped {
            ChatFinishReason::Length
        } else {
            ChatFinishReason::Stop
        };
        events.push(ChatEvent::Finish { index: 0, reason });
        events
    }

    /// The reply's text so far (markers removed, up to the first function call), and
    /// where in it the first stop string begins.
    fn speech(&self) -> (String, Option<usize>) {
        let before_call = self
            .raw
            .find(CALL_START)
            .map_or(&self.raw[..], |i| &self.raw[..i]);
        let mut text = before_call.to_owned();
        for m in ["<end_of_turn>", "<start_function_response>", "<eos>"] {
            text = text.replace(m, "");
        }
        let text = text.trim_start().to_owned();
        let stop_at = self
            .plan
            .stop
            .iter()
            .filter_map(|s| text.find(s.as_str()))
            .min();
        (text, stop_at)
    }

    /// How much of `text` can be sent: all but trailing whitespace (which the reply may
    /// yet end on) and a tail that could be the start of a stop string.
    fn safe_end(&self, text: &str) -> usize {
        let mut end = text.trim_end().len();
        let hold = self
            .plan
            .stop
            .iter()
            .map(|s| s.len().saturating_sub(1))
            .max()
            .unwrap_or(0);
        end = end.min(text.len().saturating_sub(hold));
        while !text.is_char_boundary(end) {
            end -= 1;
        }
        end
    }

    fn send_content(&mut self, text: &str, events: &mut Vec<ChatEvent>) {
        // The text only grows, so what was sent is a prefix of it.
        if text.len() > self.sent.len() && text.starts_with(self.sent.as_str()) {
            let piece = text[self.sent.len()..].to_owned();
            self.sent.push_str(&piece);
            events.push(ChatEvent::Delta {
                index: 0,
                delta: Delta {
                    content: Some(piece),
                    ..Delta::default()
                },
            });
        }
    }

    /// Sends the function calls completed since the last piece. Returns whether a
    /// single-call request now has its call.
    fn send_calls(&mut self, events: &mut Vec<ChatEvent>) -> Result<bool> {
        let complete = self.raw.matches(CALL_END).count();
        if complete <= self.calls {
            return Ok(false);
        }
        let calls = parse_calls(&self.raw).map_err(|e| {
            Error::Inference(format!("the model wrote a malformed function call: {e:#}"))
        })?;
        for (index, call) in calls.into_iter().enumerate().skip(self.calls) {
            let name = if call.name.is_empty() {
                self.plan.sole_function.clone().unwrap_or_default()
            } else {
                call.name
            };
            events.push(ChatEvent::Delta {
                index: 0,
                delta: Delta {
                    tool_calls: vec![ToolCallDelta {
                        index: count(index),
                        id: Some(format!("call_{index}")),
                        name: Some(name),
                        kind: Some(ToolCallKind::Function),
                        input: call.args.to_string(),
                    }],
                    ..Delta::default()
                },
            });
            self.calls = index + 1;
            if self.plan.single_call {
                return Ok(true);
            }
        }
        Ok(false)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chat::{AssistantMessage, Sampling, Tool, ToolCall, collect};

    fn weather() -> Tool {
        Tool {
            name: "get_weather".into(),
            description: Some("Current weather".into()),
            kind: ToolKind::Function {
                parameters: Some(
                    json!({"type": "object", "properties": {"city": {"type": "string"}}}),
                ),
                strict: false,
            },
        }
    }

    fn timer() -> Tool {
        Tool {
            name: "set_timer".into(),
            description: None,
            kind: ToolKind::Function {
                parameters: None,
                strict: false,
            },
        }
    }

    /// Feeds `pieces` through an assembler and collects the events, as the worker does.
    fn assemble(plan: ReplyPlan, pieces: &[&str], hit_limit: bool) -> Vec<ChatEvent> {
        let mut a = ReplyAssembler::new(plan);
        let mut events = Vec::new();
        for p in pieces {
            let (out, done) = a.push(p).unwrap();
            events.extend(out);
            if done {
                break;
            }
        }
        events.extend(a.finish(hit_limit));
        events
    }

    async fn completion(events: Vec<ChatEvent>) -> crate::chat::ChatCompletion {
        let mut events = events;
        events.push(ChatEvent::Usage(ChatUsage::default()));
        let stream: ChatStream = Box::pin(futures::stream::iter(events.into_iter().map(Ok)));
        collect("functiongemma3:270m", 1, stream).await.unwrap()
    }

    #[test]
    fn test_template_inputs_map_a_tool_conversation() {
        let request = ChatRequest {
            messages: vec![
                Message::Developer {
                    content: "You can call functions.".into(),
                    name: None,
                },
                Message::user("Weather in Paris?"),
                Message::Assistant(AssistantMessage {
                    tool_calls: vec![ToolCall {
                        id: "call_0".into(),
                        name: "get_weather".into(),
                        kind: ToolCallKind::Function,
                        input: r#"{"city":"Paris"}"#.into(),
                    }],
                    ..AssistantMessage::default()
                }),
                Message::Tool {
                    tool_call_id: "call_0".into(),
                    content: r#"{"temperature_c": 18}"#.into(),
                },
                Message::Tool {
                    tool_call_id: "call_0".into(),
                    content: "sunny".into(),
                },
            ],
            tools: vec![weather(), timer()],
            ..ChatRequest::default()
        };
        let (messages, tools) = template_inputs(&request).unwrap();
        assert_eq!(
            messages,
            json!([
                {"role": "developer", "content": "You can call functions."},
                {"role": "user", "content": "Weather in Paris?"},
                {"role": "assistant", "tool_calls": [{"type": "function", "function": {
                    "name": "get_weather", "arguments": {"city": "Paris"}}}]},
                {"role": "tool", "name": "get_weather", "content": {"temperature_c": 18}},
                {"role": "tool", "name": "get_weather", "content": "sunny"},
            ])
        );
        assert_eq!(tools[0]["function"]["name"], "get_weather");
        assert_eq!(tools[0]["function"]["description"], "Current weather");
        assert_eq!(
            tools[1],
            json!({"type": "function", "function": {"name": "set_timer"}})
        );
    }

    #[test]
    fn test_template_inputs_refuse_a_result_for_an_unknown_call() {
        let request = ChatRequest {
            messages: vec![
                Message::user("hi"),
                Message::Tool {
                    tool_call_id: "call_9".into(),
                    content: "{}".into(),
                },
            ],
            ..ChatRequest::default()
        };
        let err = template_inputs(&request).unwrap_err();
        assert!(
            matches!(&err, Error::InvalidParameter { parameter, .. } if parameter == "messages[1].tool_call_id"),
            "{err}"
        );
    }

    #[test]
    fn test_tool_choice_decides_which_tools_the_model_sees() {
        let base = ChatRequest {
            messages: vec![Message::user("hi")],
            tools: vec![weather(), timer()],
            ..ChatRequest::default()
        };
        let none = ChatRequest {
            tool_choice: ToolChoice::None,
            ..base.clone()
        };
        assert_eq!(template_inputs(&none).unwrap().1, json!([]));
        let allowed = ChatRequest {
            tool_choice: ToolChoice::Allowed {
                mode: crate::chat::AllowedMode::Auto,
                tools: vec!["set_timer".into()],
            },
            ..base.clone()
        };
        let (_, tools) = template_inputs(&allowed).unwrap();
        assert_eq!(tools.as_array().unwrap().len(), 1);
        // With one function offered, a call without a name means it.
        assert_eq!(
            ReplyPlan::for_request(&allowed).sole_function.as_deref(),
            Some("set_timer")
        );
        assert_eq!(ReplyPlan::for_request(&base).sole_function, None);
    }

    #[tokio::test]
    async fn test_reply_assembler_streams_text_and_finishes() {
        let events = assemble(
            ReplyPlan::default(),
            &["Hel", "lo", " wor", "ld", "!", "<end_of_turn>"],
            false,
        );
        let deltas = events
            .iter()
            .filter(|e| matches!(e, ChatEvent::Delta { .. }))
            .count();
        assert!(deltas > 1, "streamed in pieces: {events:?}");
        let c = completion(events).await;
        assert_eq!(
            c.choices[0].message.content.as_deref(),
            Some("Hello world!")
        );
        assert_eq!(c.choices[0].finish_reason, ChatFinishReason::Stop);
    }

    #[tokio::test]
    async fn test_reply_assembler_stops_at_a_stop_string_split_across_pieces() {
        let plan = ReplyPlan {
            stop: vec!["STOP".into()],
            ..ReplyPlan::default()
        };
        let events = assemble(plan, &["one two S", "TO", "P three"], false);
        let c = completion(events).await;
        assert_eq!(c.choices[0].message.content.as_deref(), Some("one two "));
        assert_eq!(c.choices[0].finish_reason, ChatFinishReason::Stop);
    }

    #[tokio::test]
    async fn test_reply_assembler_reports_the_token_limit() {
        let c = completion(assemble(ReplyPlan::default(), &["partial"], true)).await;
        assert_eq!(c.choices[0].message.content.as_deref(), Some("partial"));
        assert_eq!(c.choices[0].finish_reason, ChatFinishReason::Length);
    }

    #[tokio::test]
    async fn test_reply_assembler_turns_function_calls_into_tool_calls() {
        let plan = ReplyPlan {
            sole_function: Some("get_weather".into()),
            ..ReplyPlan::default()
        };
        let events = assemble(
            plan,
            &[
                CALL_START,
                "call:get_weather{city:<escape>Paris<escape>}",
                CALL_END,
                CALL_START,
                "call{city:<escape>Lima<escape>}",
                CALL_END,
                "<start_function_response>",
            ],
            false,
        );
        let c = completion(events).await;
        let ch = &c.choices[0];
        assert_eq!(ch.finish_reason, ChatFinishReason::ToolCalls);
        assert_eq!(ch.message.content, None);
        let calls: Vec<(&str, &str, &str)> = ch
            .message
            .tool_calls
            .iter()
            .map(|t| (t.id.as_str(), t.name.as_str(), t.input.as_str()))
            .collect();
        assert_eq!(
            calls,
            [
                ("call_0", "get_weather", r#"{"city":"Paris"}"#),
                ("call_1", "get_weather", r#"{"city":"Lima"}"#),
            ]
        );
    }

    #[tokio::test]
    async fn test_reply_assembler_stops_after_one_call_when_asked() {
        let plan = ReplyPlan {
            single_call: true,
            ..ReplyPlan::default()
        };
        let mut a = ReplyAssembler::new(plan);
        let (_, done) = a.push("<start_function_call>call:a{x:1}").unwrap();
        assert!(!done);
        let (events, done) = a.push(CALL_END).unwrap();
        assert!(done);
        assert_eq!(events.len(), 1);
    }

    #[test]
    fn test_reply_assembler_refuses_a_malformed_call() {
        let mut a = ReplyAssembler::new(ReplyPlan::default());
        let err = a
            .push("<start_function_call>call:get_weather<end_function_call>")
            .unwrap_err();
        assert!(matches!(err, Error::Inference(_)), "{err}");
    }

    #[test]
    fn test_reply_plan_follows_the_request() {
        let plan = ReplyPlan::for_request(&ChatRequest {
            messages: vec![Message::user("hi")],
            tools: vec![weather()],
            max_completion_tokens: Some(32),
            parallel_tool_calls: Some(false),
            sampling: Sampling {
                stop: vec!["\n".into()],
                ..Sampling::default()
            },
            ..ChatRequest::default()
        });
        assert_eq!(
            plan,
            ReplyPlan {
                max_tokens: Some(32),
                stop: vec!["\n".into()],
                single_call: true,
                sole_function: Some("get_weather".into()),
            }
        );
    }

    #[test]
    fn test_capabilities_refuse_what_functiongemma_cannot_do() {
        let caps = capabilities(1024);
        let refuse = |r: ChatRequest| crate::chat::validate("functiongemma3:270m", &caps, &r);
        let base = ChatRequest {
            messages: vec![Message::user("hi")],
            tools: vec![weather()],
            ..ChatRequest::default()
        };
        assert!(refuse(base.clone()).is_ok());
        for bad in [
            ChatRequest {
                n: 2,
                ..base.clone()
            },
            ChatRequest {
                tool_choice: ToolChoice::Required,
                ..base.clone()
            },
            ChatRequest {
                sampling: Sampling {
                    temperature: Some(0.7),
                    ..Sampling::default()
                },
                ..base.clone()
            },
        ] {
            assert!(
                matches!(refuse(bad), Err(Error::UnsupportedParameter { .. })),
                "accepted what FunctionGemma cannot do"
            );
        }
    }
}
