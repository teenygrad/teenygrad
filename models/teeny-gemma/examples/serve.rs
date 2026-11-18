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

//! A web demo: ask FunctionGemma about the weather, and watch it call the tool.
//!
//! ```text
//! cargo run --release -p teeny-gemma --example serve -- --model <hf-model-folder>
//! ```
//!
//! Serves one page and one endpoint:
//!
//! - `GET /` the page in `assets/index.html`, compiled into the binary.
//! - `POST /api/chat` takes `{"message": "...", "history": [...]}` and answers with an
//!   event stream: `token` as the reply is generated, `tool_call` and `tool_result`
//!   around the weather lookup, `done` with the timings, or `error`.
//!
//! The only tool offered is `get_weather` ([`teeny_gemma::tools`]), which reads from
//! Open-Meteo and writes nothing. The chat example's timer and outbox tools are
//! deliberately not here: this server is meant to face the open internet, where anything
//! that writes is a liability.
//!
//! The model is one set of weights and one KV cache, so the server generates one reply at
//! a time and answers 503 while it is busy rather than queueing. The conversation lives in
//! the browser and arrives with each request, which keeps visitors from seeing each
//! other's chats; the cost is that each turn re-renders the whole conversation, though the
//! KV cache still keeps whatever prefix matches.
//!
//! The server speaks plain HTTP. Put it behind a TLS-terminating reverse proxy before
//! exposing it, and see `--public-url` for what the page reports to visitors.

use std::io::{BufRead, BufReader, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Instant;

use clap::Parser;
use serde_json::{Value, json};
use teeny_gemma::GemmaConfig;
use teeny_gemma::chat::{Chat, parse_calls};
use teeny_gemma::runner::{CompiledStep, Session, Weights, argmax};
use teeny_test::model::runtime;

/// The demo page, compiled in so the board needs no files beside the binary.
const INDEX_HTML: &str = include_str!("../assets/index.html");

/// Longest question accepted, in bytes. Long prompts are slow on a board, and the context
/// is small, so this is a kindness to the next visitor as much as a guard.
const MAX_MESSAGE_BYTES: usize = 500;

/// Most turns of history a request may carry. Older turns are dropped by the page, but a
/// handwritten client could send anything.
const MAX_HISTORY_MESSAGES: usize = 20;

/// Longest request body accepted, in bytes.
const MAX_BODY_BYTES: usize = 64 * 1024;

#[derive(Parser)]
struct Args {
    /// The Hugging Face model folder (config.json, model.safetensors, tokenizer.json,
    /// chat_template.jinja, special_tokens_map.json).
    #[arg(long)]
    model: PathBuf,
    /// Address to listen on. The default is loopback: pass 0.0.0.0:8080 to accept
    /// connections from the network, and only behind a reverse proxy you trust.
    #[arg(long, default_value = "127.0.0.1:8080")]
    listen: String,
    /// The URL visitors reach this through, shown on the page. Defaults to `--listen`.
    #[arg(long)]
    public_url: Option<String>,
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
    /// What the page calls the machine it is running on.
    #[arg(long, default_value = "SpacemiT K3")]
    board: String,
    /// Compile the step graphs and exit, without listening. This is how the kernel cache
    /// is built on a host with teenyc, before the binary is copied to a board that has none.
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

/// The part of a reply meant for a reader. A function call is the model addressing the
/// runtime, not the visitor, so it is shown as a tool card instead and never as prose.
fn speech(text: &str) -> String {
    let t = visible(text);
    match t.find("<start_function_call>") {
        Some(i) => t[..i].trim_end().to_string(),
        None => t,
    }
}

/// One generation's measurements, reported to the page so it can show them.
struct Stats {
    prompt_tokens: usize,
    prompt_s: f64,
    first_token_s: f64,
    out_tokens: usize,
    decode_per_s: f64,
}

impl Stats {
    fn to_json(&self, context_used: usize, context_total: usize) -> Value {
        json!({
            "prompt_tokens": self.prompt_tokens,
            "prompt_per_s": self.prompt_tokens as f64 / self.prompt_s.max(1e-9),
            "first_token_s": self.first_token_s,
            "out_tokens": self.out_tokens,
            "decode_per_s": self.decode_per_s,
            "context_used": context_used,
            "context_total": context_total,
        })
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
    ) -> anyhow::Result<(String, Stats)> {
        let prompt = self.chat.prompt(messages, tools)?;
        let fresh = prompt.len()
            - self
                .session
                .tokens
                .iter()
                .zip(&prompt)
                .take_while(|(a, b)| a == b)
                .count();
        let t0 = Instant::now();
        let mut logits = self
            .session
            .feed(self.device, &mut self.chunk, &mut self.one, &prompt)?;
        let prompt_s = t0.elapsed().as_secs_f64();

        let t1 = Instant::now();
        let mut out = Vec::new();
        let mut shown = 0;
        let mut first_at: Option<f64> = None;
        loop {
            let tok = argmax(&logits);
            out.push(tok);
            first_at.get_or_insert_with(|| t1.elapsed().as_secs_f64());
            let text = speech(&self.chat.decode(&out)?);
            if text.len() > shown && text.is_char_boundary(shown) {
                on_token(&text[shown..])?;
                shown = text.len();
            }
            if self.stop.contains(&tok) || out.len() >= self.max_tokens {
                break;
            }
            anyhow::ensure!(
                self.session.cache.len < self.session.cache.capacity,
                "the conversation reached its {} token limit; start a new one",
                self.session.cache.capacity
            );
            logits = self.session.step(self.device, &mut self.one, &[tok])?;
        }
        let gen_s = t1.elapsed().as_secs_f64();
        let first = first_at.unwrap_or(0.0);
        let stats = Stats {
            prompt_tokens: fresh,
            prompt_s,
            first_token_s: prompt_s + first,
            out_tokens: out.len(),
            decode_per_s: if out.len() > 1 {
                (out.len() - 1) as f64 / (gen_s - first).max(1e-9)
            } else {
                0.0
            },
        };
        Ok((self.chat.decode(&out)?, stats))
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
    let history = history
        .iter()
        .filter_map(|m| {
            let role = m["role"].as_str()?;
            let content = m["content"].as_str()?;
            matches!(role, "user" | "assistant").then(|| json!({"role": role, "content": content}))
        })
        .collect();
    Ok((message, history))
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
    let tools = json!([teeny_gemma::tools::declaration()]);

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
    loop {
        let (text, s) = engine.reply(&json!(messages), &tools, |piece| {
            event(stream, &json!({"type": "token", "text": piece}))?;
            Ok(())
        })?;
        raw = text;
        stats = s;
        used = engine.session.cache.len;
        total = engine.session.cache.capacity;

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
            .map(|c| json!({"type": "function", "function": {"name": c.name, "arguments": c.args}}))
            .collect();
        messages.push(json!({"role": "assistant", "tool_calls": tool_calls}));

        for call in &calls {
            let args = &call.args;
            event(
                stream,
                &json!({"type": "tool_call", "name": call.name, "args": args}),
            )?;
            let result = if call.name == "get_weather" {
                let city = args["city"].as_str().unwrap_or_default().to_string();
                teeny_gemma::tools::weather(&city)
                    .unwrap_or_else(|e| json!({"error": e.to_string()}))
            } else {
                // Only the weather tool is declared, so this is the model inventing one.
                json!({"error": format!("no tool called {}", call.name)})
            };
            event(
                stream,
                &json!({"type": "tool_result", "name": call.name, "result": result}),
            )?;
            messages.push(json!({"role": "tool", "name": call.name, "content": result}));
        }
        event(stream, &json!({"type": "reply_start"}))?;
    }

    // A reply that was nothing but a tool call leaves the visitor with no words at all.
    if speech(&raw).is_empty() {
        event(
            stream,
            &json!({"type": "token",
                    "text": "I looked that up, but could not put it into words."}),
        )?;
    }
    event(
        stream,
        &json!({"type": "done", "stats": stats.to_json(used, total)}),
    )?;
    Ok(())
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();

    let cfg = GemmaConfig::load(&args.model.join("config.json"))?;
    let chat = Chat::load(&args.model)?;
    let mut stop = cfg.eos_token_id.ids();
    // <eos>, <start_function_response>, <end_of_turn>.
    stop.extend([1, 50, 106]);

    let t0 = Instant::now();
    let weights = Weights::load(&args.model.join("model.safetensors"))?;
    let device = runtime::open()?;
    let session = Session::new(cfg.clone(), args.context);
    let chunk = CompiledStep::new(
        &device,
        &cfg,
        &weights,
        args.chunk,
        args.context,
        &session.past,
    )?;
    let one = CompiledStep::new(&device, &cfg, &weights, 1, args.context, &session.past)?;
    drop(weights);
    eprintln!(
        "FunctionGemma on {} (ready in {:.1} s)",
        runtime::BACKEND_NAME,
        t0.elapsed().as_secs_f64()
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
        println!("kernels compiled; not listening (--compile-only)");
        return Ok(());
    }

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
                    .and_then(|()| {
                        handle_chat(&mut engine, &args.system, &request.body, &mut stream)
                    });
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
