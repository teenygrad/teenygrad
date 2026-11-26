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

//! FunctionGemma on teenygrad, on whichever backend the build selects (CUDA, or RISC-V on
//! a board), in one of three modes:
//!
//! ```text
//! cargo run --release -p teeny-llm --example chat -- [--tools tools.json]
//! cargo run --release -p teeny-llm --example chat -- --mode tool \
//!     --prompt "What is the weather in Paris?"
//! cargo run --release -p teeny-llm --example chat -- --mode serve
//! ```
//!
//! `--model` picks the model by tag, `family:size` or the family alone for its default size
//! (`functiongemma3`, the default and so far the only one, is `functiongemma3:270m`). Its
//! weights come from the local Hugging Face cache (`google/functiongemma-270m-it`) unless
//! `--model-dir` names a folder.
//!
//! **`chat`** (the default) is an interactive conversation in the terminal. Type a request.
//! When the model calls a demo tool, the chat runs it: `get_weather` asks Open-Meteo (needs
//! `curl` and internet), `set_timer` counts down and says when it is done, and
//! `send_message` appends to `~/functiongemma-outbox.txt`. For any other tool, or with
//! `--manual-tools`, you play the tool: type its result (JSON, or plain text) and the model
//! answers from it; an empty line skips the call. Commands: `/reset` clears the
//! conversation, `/quit` exits. Each turn re-renders the whole conversation with the
//! model's chat template, and the KV cache keeps every token that still matches, so a turn
//! only processes what is new.
//!
//! **`tool`** answers one `--prompt` and prints the function calls the model chooses,
//! without running them.
//!
//! In both of those, `--tools` is a JSON list of OpenAI-style function declarations
//! (default: the demo tools above).
//!
//! **`serve`** is a web demo: ask about the weather in a browser, and watch the model call
//! the tool while the page paints the city's real sky. It serves one page and one endpoint:
//!
//! - `GET /` the page in `examples/assets/functiongemma/index.html`, compiled into the binary.
//! - `POST /api/chat` takes `{"message": "...", "history": [...]}` and answers with an
//!   event stream: `token` as the reply is generated, `tool_call` and `tool_result`
//!   around the weather lookup, `done` with the timings, or `error`.
//!
//! The only tool it offers is `get_weather` ([`teeny_llm::models::functiongemma::tools`]),
//! which reads from Open-Meteo and writes nothing. The timer and outbox tools are
//! deliberately not offered there: the server is meant to face the open internet, where
//! anything that writes is a liability. The model is one set of weights and one KV cache,
//! so the server generates one reply at a time and answers 503 while it is busy rather than
//! queueing. The conversation lives in the browser and arrives with each request, which
//! keeps visitors from seeing each other's chats; the cost is that each turn re-renders the
//! whole conversation, though the KV cache still keeps whatever prefix matches. The server
//! speaks plain HTTP: put it behind a TLS-terminating reverse proxy before exposing it, and
//! see `--public-url` for what the page reports to visitors.
//!
//! Decoding is greedy and stops at `<start_function_response>` (the model handing a call
//! to the host), `<end_of_turn>` or `<eos>`.

use std::io::{BufRead, BufReader, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Instant;

use clap::{Parser, ValueEnum};
use serde_json::{Value, json};
use teeny_llm::models::Model;
use teeny_llm::models::functiongemma::GemmaConfig;
use teeny_llm::models::functiongemma::chat::{Chat, parse_calls};
use teeny_llm::models::functiongemma::runner::{CompiledStep, Session, Weights, argmax};
use teeny_llm::models::functiongemma::tools;
use teeny_llm::timings::{Recorder, Timings};
use teeny_test::model::runtime;

/// The demo page, compiled in so the board needs no files beside the binary.
const INDEX_HTML: &str = include_str!("assets/functiongemma/index.html");

/// Longest question accepted, in bytes. Long prompts are slow on a board, and the context
/// is small, so this is a kindness to the next visitor as much as a guard.
const MAX_MESSAGE_BYTES: usize = 500;

/// Most turns of history a request may carry. Older turns are dropped by the page, but a
/// handwritten client could send anything.
const MAX_HISTORY_MESSAGES: usize = 20;

/// Longest request body accepted, in bytes.
const MAX_BODY_BYTES: usize = 64 * 1024;

/// `--model`'s parser: the tags of the models in this build, as [`Model`]s.
fn model_parser() -> impl clap::builder::TypedValueParser<Value = Model> {
    use clap::builder::{PossibleValuesParser, TypedValueParser};
    PossibleValuesParser::new(Model::tags()).try_map(|tag| tag.parse::<Model>())
}

/// What the example does with the model.
#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
enum Mode {
    /// An interactive conversation in the terminal that runs the demo tools.
    Chat,
    /// One request: print the function calls the model chooses.
    Tool,
    /// The weather web demo, served over HTTP.
    Serve,
}

#[derive(Parser)]
struct Args {
    /// The model to run, by tag (`family:size`, or the family for its default size); the
    /// choices are the models this build includes.
    #[arg(long, default_value = "functiongemma3", ignore_case = true, value_parser = model_parser())]
    model: Model,
    /// The model's weights folder (for FunctionGemma: config.json, model.safetensors,
    /// tokenizer.json, chat_template.jinja, special_tokens_map.json). Defaults to the
    /// model's snapshot in the local Hugging Face cache.
    #[arg(long)]
    model_dir: Option<PathBuf>,
    /// `chat` for a terminal conversation, `tool` for a single request's function calls,
    /// `serve` for the weather web demo.
    #[arg(long, value_enum, default_value_t = Mode::Chat)]
    mode: Mode,
    /// The request, in `tool` mode.
    #[arg(long, required_if_eq("mode", "tool"))]
    prompt: Option<String>,
    /// In `chat` and `tool` modes, a JSON file with the tool declarations (OpenAI style).
    /// Defaults to a few demo tools: weather, timer, messaging. `serve` offers only the
    /// weather tool.
    #[arg(long)]
    tools: Option<PathBuf>,
    /// In `chat` mode, type each tool's result yourself rather than running the demo
    /// tools for real.
    #[arg(long)]
    manual_tools: bool,
    /// An optional system prompt.
    #[arg(long)]
    system: Option<String>,
    /// Conversation length limit in tokens (the KV cache size).
    #[arg(long, default_value_t = 1024)]
    context: usize,
    /// Most tokens per reply.
    #[arg(long, default_value_t = 128)]
    max_tokens: usize,
    /// Tokens per prompt-processing step.
    #[arg(long, default_value_t = 16)]
    chunk: usize,
    /// In `serve` mode, the address to listen on. The default is loopback: pass
    /// 0.0.0.0:8080 to accept connections from the network, and only behind a reverse proxy
    /// you trust.
    #[arg(long, default_value = "127.0.0.1:8080")]
    listen: String,
    /// In `serve` mode, the URL visitors reach this through, shown on the page. Defaults to
    /// `--listen`.
    #[arg(long)]
    public_url: Option<String>,
    /// In `serve` mode, what the page calls the machine it is running on.
    #[arg(long, default_value = "SpacemiT K3")]
    board: String,
    /// Compile the step graphs and exit. This is how the kernel cache is built on a host
    /// with teenyc, before the binary is copied to a board that has none.
    #[arg(long)]
    compile_only: bool,
}

/// Model text with FunctionGemma's markers and end-of-turn tokens removed.
fn visible(text: &str) -> String {
    let mut t = text.to_string();
    for m in ["<end_of_turn>", "<start_function_response>", "<eos>"] {
        t = t.replace(m, "");
    }
    t.trim().to_string()
}

/// `text` cut to `limit` bytes on a character boundary, for a log line.
fn truncate(text: &str, limit: usize) -> &str {
    if text.len() <= limit {
        return text;
    }
    let mut end = limit;
    while end > 0 && !text.is_char_boundary(end) {
        end -= 1;
    }
    &text[..end]
}

/// The part of a reply meant for a reader. A function call is the model addressing the
/// runtime, not the reader, so it is shown as a tool card (or a `call:` line) instead and
/// never as prose.
fn speech(text: &str) -> String {
    let t = visible(text);
    match t.find("<start_function_call>") {
        Some(i) => t[..i].trim_end().to_string(),
        None => t,
    }
}

/// One generation's measurements, as the page and the terminal modes show them.
trait Report {
    fn to_json(&self, context_used: usize, context_total: usize) -> Value;
    fn report(&self, context_used: usize, context_total: usize) -> String;
}

impl Report for Timings {
    fn to_json(&self, context_used: usize, context_total: usize) -> Value {
        let ms = |d: Option<std::time::Duration>| d.map(|d| d.as_secs_f64() * 1e3);
        json!({
            "prompt_tokens": self.prompt_tokens,
            "prompt_per_s": self.prompt_tokens_per_second(),
            "first_token_s": self.first_token.as_secs_f64(),
            "out_tokens": self.generated_tokens,
            "decode_per_s": self.generation_tokens_per_second(),
            "decode_p50_ms": ms(self.latency_percentile(50.0)),
            "decode_p99_ms": ms(self.latency_percentile(99.0)),
            "context_used": context_used,
            "context_total": context_total,
        })
    }

    /// The same measurements as one line, for the terminal modes.
    fn report(&self, context_used: usize, context_total: usize) -> String {
        format!(
            "  [{} new prompt tokens in {:.2} s ({:.1} tokens/s); first token after {:.2} s; \
             {} tokens out at {:.1} tokens/s decode; context {}/{}]",
            self.prompt_tokens,
            self.prompt.as_secs_f64(),
            self.prompt_tokens_per_second(),
            self.first_token.as_secs_f64(),
            self.generated_tokens,
            self.generation_tokens_per_second(),
            context_used,
            context_total
        )
    }
}

struct Engine<'d> {
    device: &'d runtime::Device<'d>,
    chat: Chat,
    session: Session,
    chunk: CompiledStep<'d>,
    one: CompiledStep<'d>,
    stop: Vec<u32>,
    max_tokens: usize,
}

impl Engine<'_> {
    /// Generates one reply to `messages`, handing each new piece of visible text to
    /// `on_token` as it arrives. Returns the raw text (markers and all) and the timings.
    fn reply(
        &mut self,
        messages: &Value,
        tools: &Value,
        mut on_token: impl FnMut(&str) -> anyhow::Result<()>,
    ) -> anyhow::Result<(String, Timings)> {
        let prompt = self.chat.prompt(messages, tools)?;
        // What `feed` runs: all but the shared prefix, and at least one token.
        let fresh = prompt.len()
            - self
                .session
                .cache
                .shared(&prompt)
                .min(prompt.len().saturating_sub(1));
        let mut timings = Recorder::new();
        timings.begin_prompt();
        let mut logits = self
            .session
            .feed(self.device, &mut self.chunk, &mut self.one, &prompt)?;
        timings.end_prompt(fresh, prompt.len() - fresh);

        let mut out = Vec::new();
        let mut shown = 0;
        // The reply so far, decoded a token at a time rather than re-decoded in full.
        let mut decoder = self.chat.stream_decoder();
        let mut raw = String::new();
        loop {
            let tok = argmax(&logits);
            out.push(tok);
            timings.token();
            if let Some(piece) = decoder.step(tok)? {
                raw.push_str(&piece);
            }
            let text = speech(&raw);
            if text.len() > shown && text.is_char_boundary(shown) {
                on_token(&text[shown..])?;
                shown = text.len();
            }
            if self.stop.contains(&tok) || out.len() >= self.max_tokens {
                break;
            }
            anyhow::ensure!(
                self.session.cache.len() < self.session.cache.capacity(),
                "the conversation reached its {} token limit (--context); start a new one",
                self.session.cache.capacity()
            );
            logits = self.session.step(self.device, &mut self.one, &[tok])?;
        }
        Ok((self.chat.decode(&out)?, timings.finish()))
    }
}

// ---------------------------------------------------------------------------
// A small HTTP/1.1 server
// ---------------------------------------------------------------------------
//
// Hand-written on `std::net` rather than pulling in a web framework: this binary is
// cross-compiled for riscv64 and copied to a board, and the fewer dependencies that have
// to build for that target, the better. It serves exactly the two routes the demo needs.

/// A parsed request: the method, the path and the body.
struct Request {
    method: String,
    path: String,
    body: Vec<u8>,
}

/// Reads one request. `None` if the connection closed before a request line arrived.
fn read_request(stream: &mut BufReader<&TcpStream>) -> anyhow::Result<Option<Request>> {
    let mut line = String::new();
    if stream.read_line(&mut line)? == 0 {
        return Ok(None);
    }
    let mut parts = line.split_whitespace();
    let method = parts.next().unwrap_or_default().to_string();
    let path = parts.next().unwrap_or_default().to_string();

    let mut length = 0usize;
    loop {
        let mut header = String::new();
        if stream.read_line(&mut header)? == 0 {
            break;
        }
        let header = header.trim_end();
        if header.is_empty() {
            break;
        }
        if let Some(v) = header.strip_prefix("Content-Length:") {
            length = v.trim().parse().unwrap_or(0);
        } else if let Some(v) = header.strip_prefix("content-length:") {
            length = v.trim().parse().unwrap_or(0);
        }
    }
    anyhow::ensure!(
        length <= MAX_BODY_BYTES,
        "request body of {length} bytes is over the {MAX_BODY_BYTES} byte limit"
    );
    let mut body = vec![0u8; length];
    stream.read_exact(&mut body)?;
    Ok(Some(Request { method, path, body }))
}

/// Writes a complete response with a body.
fn respond(
    stream: &mut TcpStream,
    status: &str,
    content_type: &str,
    body: &[u8],
) -> std::io::Result<()> {
    write!(
        stream,
        "HTTP/1.1 {status}\r\nContent-Type: {content_type}\r\nContent-Length: {}\r\n\
         Cache-Control: no-store\r\nConnection: close\r\n\r\n",
        body.len()
    )?;
    stream.write_all(body)?;
    stream.flush()
}

/// Starts an event stream, which [`event`] then writes into.
fn begin_stream(stream: &mut TcpStream) -> std::io::Result<()> {
    write!(
        stream,
        "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\n\
         Cache-Control: no-store\r\nConnection: close\r\n\r\n"
    )?;
    stream.flush()
}

/// Writes one server-sent event: a single line of JSON, flushed at once so the page sees
/// each token as the board produces it.
fn event(stream: &mut TcpStream, payload: &Value) -> std::io::Result<()> {
    write!(stream, "data: {payload}\n\n")?;
    stream.flush()
}

/// Pulls `{"message": ..., "history": [...]}` out of a request body, with the limits
/// applied. The history is the page's own record of the conversation.
fn parse_body(body: &[u8]) -> anyhow::Result<(String, Vec<Value>)> {
    let v: Value = serde_json::from_slice(body).map_err(|e| anyhow::anyhow!("bad JSON: {e}"))?;
    let message = v["message"]
        .as_str()
        .ok_or_else(|| anyhow::anyhow!("no message"))?
        .trim()
        .to_string();
    anyhow::ensure!(!message.is_empty(), "the message is empty");
    anyhow::ensure!(
        message.len() <= MAX_MESSAGE_BYTES,
        "the message is {} bytes, over the {MAX_MESSAGE_BYTES} byte limit",
        message.len()
    );
    let history = v["history"].as_array().cloned().unwrap_or_default();
    anyhow::ensure!(
        history.len() <= MAX_HISTORY_MESSAGES,
        "the history has {} messages, over the {MAX_HISTORY_MESSAGES} message limit",
        history.len()
    );
    // Keep only what the template needs, so a client cannot inject other fields.
    //
    // The tool exchange has to survive the round trip. Without it the model sees a past turn
    // where it answered a weather question without ever calling a tool, concludes it has no
    // such tool, and refuses the next one.
    let history: Vec<Value> = history
        .iter()
        .filter_map(|m| match m["role"].as_str()? {
            role @ ("user" | "assistant") if m["content"].is_string() => {
                Some(json!({"role": role, "content": m["content"].as_str()?}))
            }
            "assistant" => {
                let calls: Vec<Value> = m["tool_calls"]
                    .as_array()?
                    .iter()
                    .filter_map(|c| {
                        let f = c.get("function")?;
                        Some(json!({"type": "function", "function": {
                            "name": f["name"].as_str()?, "arguments": f["arguments"].clone(),
                        }}))
                    })
                    .collect();
                (!calls.is_empty()).then(|| json!({"role": "assistant", "tool_calls": calls}))
            }
            "tool" => Some(json!({
                "role": "tool", "name": m["name"].as_str()?, "content": m["content"].clone(),
            })),
            _ => None,
        })
        .collect();
    // A `tool` message with nothing to answer confuses the template, so drop any that lost
    // its call to trimming.
    let mut pruned: Vec<Value> = Vec::with_capacity(history.len());
    for m in history {
        let orphan = m["role"] == "tool"
            && !matches!(pruned.last(), Some(p) if p["tool_calls"].is_array() || p["role"] == "tool");
        if !orphan {
            pruned.push(m);
        }
    }
    Ok((message, pruned))
}

/// Runs one conversation turn, streaming events to the browser: the model's reply, and if
/// it calls `get_weather`, the call, the result and the reply it then gives.
fn handle_chat(
    engine: &mut Engine<'_>,
    system: &Option<String>,
    body: &[u8],
    stream: &mut TcpStream,
) -> anyhow::Result<()> {
    let (message, history) = parse_body(body)?;
    let tools = json!([tools::declaration()]);

    let mut messages: Vec<Value> = Vec::new();
    if let Some(system) = system {
        messages.push(json!({"role": "system", "content": system}));
    }
    messages.extend(history);
    messages.push(json!({"role": "user", "content": message}));

    // Up to this many tool rounds before we stop and show whatever the model last said,
    // so a model that keeps calling the tool cannot hold the board forever.
    const MAX_ROUNDS: usize = 3;

    let mut raw;
    let mut stats;
    let (mut used, mut total);
    let mut round = 0;
    // The last reading the tool returned, kept to check the model's words against.
    let mut reading: Option<Value> = None;
    loop {
        let (text, s) = engine.reply(&json!(messages), &tools, |piece| {
            event(stream, &json!({"type": "token", "text": piece}))?;
            Ok(())
        })?;
        raw = text;
        stats = s;
        used = engine.session.cache.len();
        total = engine.session.cache.capacity();

        let calls = parse_calls(&raw)?;
        if calls.is_empty() || round == MAX_ROUNDS {
            break;
        }
        round += 1;

        // The model's turn was the call itself, so it goes back as `tool_calls` rather
        // than as text, and each result as a `tool` message. This is the shape the chat
        // template expects; anything else and the model just calls the tool again.
        let tool_calls: Vec<Value> = calls
            .iter()
            .map(|c| {
                let name = if c.name.is_empty() {
                    "get_weather"
                } else {
                    c.name.as_str()
                };
                json!({"type": "function", "function": {"name": name, "arguments": c.args}})
            })
            .collect();
        messages.push(json!({"role": "assistant", "tool_calls": tool_calls}));

        for call in &calls {
            let args = &call.args;
            // An empty name means the model called without saying what. Only one function is
            // declared, so there is no ambiguity about which it meant.
            let name = if call.name.is_empty() {
                "get_weather"
            } else {
                call.name.as_str()
            };
            event(
                stream,
                &json!({"type": "tool_call", "name": name, "args": args}),
            )?;
            let result = if name == "get_weather" {
                let city = args["city"].as_str().unwrap_or_default().to_string();
                let mut w =
                    tools::weather(&city).unwrap_or_else(|e| json!({"error": e.to_string()}));
                // The model drops syllables: "Novosibirsk" went out as "Novibirsk". When no
                // such place exists, try the name the visitor actually typed, which is better
                // evidence of what was meant than the model's transcription of it.
                if w["error"]
                    .as_str()
                    .is_some_and(|e| e.starts_with("no place called"))
                    && let Some(meant) = tools::nearest_from(&message, &city)
                    && let Ok(second) = tools::weather(&meant)
                {
                    w = second;
                    // Say so, rather than quietly answering about somewhere else.
                    w["asked_for"] = json!(city);
                }
                reading = Some(w.clone());
                w
            } else {
                // Only the weather tool is declared, so this is the model inventing one.
                json!({"error": format!("no tool called {name}")})
            };
            event(
                stream,
                &json!({"type": "tool_result", "name": name, "result": result}),
            )?;
            // The browser got the whole result above, scene and all; the model gets the
            // visitor-facing fields only.
            messages.push(json!({
                "role": "tool", "name": name,
                "content": tools::model_view(&result),
            }));
        }
        event(stream, &json!({"type": "reply_start"}))?;
    }

    // A reply that was nothing but a tool call leaves the visitor with no words at all.
    let words = speech(&raw);
    if words.is_empty() {
        // The model meant to call something and the parser would not have it. Nothing of this
        // reaches the browser, because the markup is special tokens that decode to nothing,
        // so without this line the failure is invisible and unfixable.
        eprintln!(
            "unparsable reply ({} bytes): {:?}",
            raw.len(),
            truncate(&raw, 400)
        );
        event(
            stream,
            &json!({"type": "token",
                    "text": "I looked that up, but could not put it into words."}),
        )?;
    } else if reading
        .as_ref()
        .is_some_and(|r| tools::contradicts(&words, r) || tools::misquotes(&words, r))
    {
        // The words have already been streamed, so the browser is told to take them back.
        // The reading itself stands: it is the part that was never in doubt.
        event(stream, &json!({"type": "retract"}))?;
    }
    event(
        stream,
        &json!({"type": "done", "stats": stats.to_json(used, total)}),
    )?;
    Ok(())
}

fn demo_tools() -> Value {
    json!([
        {"type": "function", "function": {
            "name": "get_weather", "description": "Get current weather for a city",
            "parameters": {"type": "object",
                "properties": {"city": {"type": "string", "description": "City name"}},
                "required": ["city"]}}},
        {"type": "function", "function": {
            "name": "set_timer", "description": "Start a countdown timer",
            "parameters": {"type": "object",
                "properties": {
                    "minutes": {"type": "integer", "description": "Length in minutes"},
                    "label": {"type": "string", "description": "What the timer is for"}},
                "required": ["minutes"]}}},
        {"type": "function", "function": {
            "name": "send_message", "description": "Send a text message to a contact",
            "parameters": {"type": "object",
                "properties": {
                    "to": {"type": "string", "description": "Contact name"},
                    "text": {"type": "string", "description": "Message body"}},
                "required": ["to", "text"]}}}
    ])
}

/// Runs a demo tool for real: `get_weather` asks Open-Meteo (through `curl`), `set_timer`
/// starts a countdown that prints when it ends, and `send_message` appends to
/// `~/functiongemma-outbox.txt` (there is no messaging service to send through). `None` for
/// any other tool, which the user then answers.
fn run_tool(name: &str, args: &Value) -> Option<Value> {
    let text = |key: &str| {
        args.get(key)
            .map(|v| v.as_str().map_or(v.to_string(), str::to_string))
    };
    Some(match name {
        "get_weather" => tools::weather(&text("city")?)
            .map(|r| tools::model_view(&r))
            .unwrap_or_else(|e| json!({"error": e.to_string()})),
        "set_timer" => {
            let minutes: f64 = text("minutes")?.trim().parse().ok()?;
            let label = text("label").unwrap_or_else(|| "timer".into());
            let secs = (minutes * 60.0).max(0.0) as u64;
            let ends = std::time::SystemTime::now() + std::time::Duration::from_secs(secs);
            let ends = ends.duration_since(std::time::UNIX_EPOCH).ok()?.as_secs() % 86400;
            let ends = format!("{:02}:{:02} UTC", ends / 3600, ends / 60 % 60);
            let l = label.clone();
            std::thread::spawn(move || {
                std::thread::sleep(std::time::Duration::from_secs(secs));
                println!("\n  [timer '{l}' ({minutes} min) is done]");
            });
            json!({"status": "started", "minutes": minutes, "label": label, "ends_at": ends})
        }
        "send_message" => {
            let (to, body) = (text("to")?, text("text").unwrap_or_default());
            let path = std::env::var_os("HOME")
                .map(PathBuf::from)
                .unwrap_or_default()
                .join("functiongemma-outbox.txt");
            let saved = std::fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(&path)
                .and_then(|mut f| writeln!(f, "to {to}: {body}"));
            match saved {
                Ok(()) => json!({"status": "sent", "to": to, "text": body}),
                Err(e) => json!({"status": "failed", "error": e.to_string()}),
            }
        }
        _ => return None,
    })
}

fn read_line(prompt: &str) -> anyhow::Result<Option<String>> {
    print!("{prompt}");
    std::io::stdout().flush()?;
    let mut line = String::new();
    if std::io::stdin().lock().read_line(&mut line)? == 0 {
        return Ok(None);
    }
    Ok(Some(line.trim_end_matches(['\n', '\r']).to_string()))
}

/// One reply in a terminal mode: streamed to stdout after `model> `, with its timings on
/// stderr.
fn terminal_reply(
    engine: &mut Engine<'_>,
    messages: &Value,
    tools: &Value,
) -> anyhow::Result<String> {
    print!("model> ");
    std::io::stdout().flush()?;
    let (reply, stats) = engine.reply(messages, tools, |piece| {
        print!("{piece}");
        std::io::stdout().flush()?;
        Ok(())
    })?;
    println!();
    eprintln!(
        "{}",
        stats.report(engine.session.cache.len(), engine.session.cache.capacity())
    );
    Ok(reply)
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    anyhow::ensure!(
        args.mode == Mode::Tool || args.prompt.is_none(),
        "--prompt is for --mode tool; in chat mode, type the request at the prompt"
    );
    anyhow::ensure!(
        args.mode != Mode::Serve || args.tools.is_none(),
        "--tools is for --mode chat and tool; serve offers only the weather tool"
    );
    // The modes below are FunctionGemma's: its chat template, tools and step graphs.
    anyhow::ensure!(
        matches!(args.model, Model::FunctionGemma3),
        "this example runs FunctionGemma only, not {}",
        args.model
    );
    let model_dir = args.model.locate(args.model_dir.as_deref())?;
    let parameters = args.model.parameter_count(&model_dir)?;
    let cfg = GemmaConfig::load(&model_dir.join("config.json"))?;
    let chat = Chat::load(&model_dir)?;
    let tools: Value = match (&args.tools, args.mode) {
        (_, Mode::Serve) => json!([tools::declaration()]),
        (Some(p), _) => serde_json::from_str(&std::fs::read_to_string(p)?)?,
        (None, _) => demo_tools(),
    };
    let mut stop = cfg.eos_token_id.ids();
    // <eos>, <start_function_response>, <end_of_turn>.
    stop.extend([1, 50, 106]);

    let t0 = Instant::now();
    let weights = Weights::load(&model_dir.join("model.safetensors"))?;
    let device = runtime::open()?;
    let session = Session::new(cfg.clone(), args.context);
    let chunk = CompiledStep::new(
        &device,
        &cfg,
        &weights,
        args.chunk,
        args.context,
        &session.past(),
    )?;
    let one = CompiledStep::new(&device, &cfg, &weights, 1, args.context, &session.past())?;
    drop(weights);
    let names: Vec<&str> = tools
        .as_array()
        .map(|a| {
            a.iter()
                .filter_map(|t| t["function"]["name"].as_str())
                .collect()
        })
        .unwrap_or_default();
    let spec = args.model.spec();
    eprintln!(
        "{} ({}, {} parameters, {}) on {} (ready in {:.1} s). Tools: {}.{}",
        spec.display_name,
        args.model,
        parameters,
        spec.architecture,
        runtime::BACKEND_NAME,
        t0.elapsed().as_secs_f64(),
        names.join(", "),
        if args.mode == Mode::Chat {
            " /reset, /quit."
        } else {
            ""
        }
    );

    let mut engine = Engine {
        device: &device,
        chat,
        session,
        chunk,
        one,
        stop,
        max_tokens: args.max_tokens,
    };
    if args.compile_only {
        println!("kernels compiled (--compile-only)");
        return Ok(());
    }
    let base: Vec<Value> = args
        .system
        .iter()
        .map(|s| json!({"role": "system", "content": s}))
        .collect();

    match args.mode {
        Mode::Chat => chat_loop(&mut engine, &tools, &base, args.manual_tools),
        Mode::Tool => {
            let mut messages = base;
            messages.push(
                json!({"role": "user", "content": args.prompt.as_deref().unwrap_or_default()}),
            );
            let reply = terminal_reply(&mut engine, &Value::Array(messages), &tools)?;
            for call in parse_calls(&reply)? {
                println!("call: {}({})", call.name, call.args);
            }
            Ok(())
        }
        Mode::Serve => serve(&mut engine, &args),
    }
}

/// The weather web demo of `serve` mode: answers HTTP requests until the process is
/// stopped.
fn serve(engine: &mut Engine<'_>, args: &Args) -> anyhow::Result<()> {
    let listener = TcpListener::bind(&args.listen)?;
    let public = args
        .public_url
        .clone()
        .unwrap_or(format!("http://{}", args.listen));
    println!("listening on {} ({})", args.listen, public);

    // One reply at a time: the model is a single set of weights and one KV cache. A second
    // visitor is told the board is busy rather than made to wait behind an unknown queue.
    let busy = AtomicBool::new(false);

    for stream in listener.incoming() {
        let mut stream = match stream {
            Ok(s) => s,
            Err(e) => {
                eprintln!("accept failed: {e}");
                continue;
            }
        };
        let request = {
            let mut reader = BufReader::new(&stream);
            match read_request(&mut reader) {
                Ok(Some(r)) => r,
                Ok(None) => continue,
                Err(e) => {
                    let body = json!({"error": e.to_string()}).to_string();
                    let _ = respond(
                        &mut stream,
                        "400 Bad Request",
                        "application/json",
                        body.as_bytes(),
                    );
                    continue;
                }
            }
        };

        match (request.method.as_str(), request.path.as_str()) {
            ("GET", "/") | ("GET", "/index.html") => {
                let page = INDEX_HTML
                    .replace("{{BOARD}}", &args.board)
                    .replace("{{PUBLIC_URL}}", &public);
                let _ = respond(
                    &mut stream,
                    "200 OK",
                    "text/html; charset=utf-8",
                    page.as_bytes(),
                );
            }
            ("GET", "/api/health") => {
                let body = json!({
                    "status": "ok",
                    "board": args.board,
                    "busy": busy.load(Ordering::SeqCst),
                })
                .to_string();
                let _ = respond(&mut stream, "200 OK", "application/json", body.as_bytes());
            }
            ("POST", "/api/chat") => {
                if busy.swap(true, Ordering::SeqCst) {
                    let body = json!({"error": "the board is answering someone else; try again in a moment"})
                        .to_string();
                    let _ = respond(
                        &mut stream,
                        "503 Service Unavailable",
                        "application/json",
                        body.as_bytes(),
                    );
                    continue;
                }
                let result = begin_stream(&mut stream)
                    .map_err(anyhow::Error::from)
                    .and_then(|()| handle_chat(engine, &args.system, &request.body, &mut stream));
                if let Err(e) = result {
                    eprintln!("chat failed: {e}");
                    let _ = event(
                        &mut stream,
                        &json!({"type": "error", "message": e.to_string()}),
                    );
                }
                busy.store(false, Ordering::SeqCst);
            }
            _ => {
                let body = json!({"error": "not found"}).to_string();
                let _ = respond(
                    &mut stream,
                    "404 Not Found",
                    "application/json",
                    body.as_bytes(),
                );
            }
        }
    }
    Ok(())
}

/// The interactive conversation of `chat` mode.
fn chat_loop(
    engine: &mut Engine<'_>,
    tools: &Value,
    base: &[Value],
    manual_tools: bool,
) -> anyhow::Result<()> {
    let mut messages = base.to_vec();
    while let Some(line) = read_line("you> ")? {
        let line = line.trim();
        match line {
            "" => continue,
            "/quit" | "/exit" => break,
            "/reset" => {
                messages = base.to_vec();
                eprintln!("  [conversation cleared]");
                continue;
            }
            _ => {}
        }
        messages.push(json!({"role": "user", "content": line}));
        let mut reply = terminal_reply(engine, &Value::Array(messages.clone()), tools)?;

        // Function calls: run the tool (or the user plays it), then the model answers.
        let mut rounds = 0;
        loop {
            let calls = parse_calls(&reply)?;
            if calls.is_empty() || rounds == 4 {
                messages.push(json!({"role": "assistant", "content": visible(&reply)}));
                break;
            }
            rounds += 1;
            let tool_calls: Vec<Value> = calls
                .iter()
                .map(|c| json!({"type": "function", "function": {"name": c.name, "arguments": c.args}}))
                .collect();
            messages.push(json!({"role": "assistant", "tool_calls": tool_calls}));
            let mut answered = false;
            for c in &calls {
                println!("  call: {}({})", c.name, c.args);
                if let Some(result) = (!manual_tools)
                    .then(|| run_tool(&c.name, &c.args))
                    .flatten()
                {
                    println!("  {} -> {result}", c.name);
                    messages.push(json!({"role": "tool", "name": c.name, "content": result}));
                    answered = true;
                    continue;
                }
                let Some(result) = read_line(&format!("  {} result> ", c.name))? else {
                    return Ok(());
                };
                if result.trim().is_empty() {
                    continue;
                }
                let content = serde_json::from_str::<Value>(&result)
                    .ok()
                    .filter(Value::is_object)
                    .unwrap_or(Value::String(result));
                messages.push(json!({"role": "tool", "name": c.name, "content": content}));
                answered = true;
            }
            if !answered {
                break;
            }
            reply = terminal_reply(engine, &Value::Array(messages.clone()), tools)?;
        }
    }
    Ok(())
}
