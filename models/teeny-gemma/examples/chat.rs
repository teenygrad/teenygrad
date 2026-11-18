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

//! An interactive chat with FunctionGemma on teenygrad, on whichever backend the
//! build selects (CUDA, or RISC-V on a board).
//!
//! ```text
//! cargo run --release -p teeny-gemma --example chat -- --model <hf-model-folder> [--tools tools.json]
//! ```
//!
//! Type a request. When the model calls a demo tool, the chat runs it: `get_weather`
//! asks Open-Meteo (needs `curl` and internet), `set_timer` counts down and says when it
//! is done, and `send_message` appends to `~/teeny-gemma-outbox.txt`. For any other tool,
//! or with `--manual-tools`, you play the tool: type its result (JSON, or plain text) and
//! the model answers from it; an empty line skips the call. Commands: `/reset` clears the
//! conversation, `/quit` exits.
//!
//! Each turn re-renders the whole conversation with the model's chat template, and the
//! KV cache keeps every token that still matches, so a turn only processes what is new.

use std::io::{BufRead, Write};
use std::path::PathBuf;
use std::time::Instant;

use clap::Parser;
use serde_json::{Value, json};
use teeny_gemma::GemmaConfig;
use teeny_gemma::chat::{Chat, parse_calls};
use teeny_gemma::runner::{CompiledStep, Session, Weights, argmax};
use teeny_test::model::runtime;

#[derive(Parser)]
struct Args {
    /// The Hugging Face model folder (config.json, model.safetensors, tokenizer.json,
    /// chat_template.jinja, special_tokens_map.json).
    #[arg(long)]
    model: PathBuf,
    /// JSON file with the tool declarations (OpenAI style). Defaults to a few demo
    /// tools: weather, timer, messaging.
    #[arg(long)]
    tools: Option<PathBuf>,
    /// Type each tool's result yourself, rather than running the demo tools for real.
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
/// `~/teeny-gemma-outbox.txt` (there is no messaging service to send through). `None` for
/// any other tool, which the user then answers.
fn run_tool(name: &str, args: &Value) -> Option<Value> {
    let text = |key: &str| {
        args.get(key)
            .map(|v| v.as_str().map_or(v.to_string(), str::to_string))
    };
    Some(match name {
        "get_weather" => teeny_gemma::tools::weather(&text("city")?)
            .map(|r| teeny_gemma::tools::model_view(&r))
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
                .join("teeny-gemma-outbox.txt");
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

/// Model text with FunctionGemma's markers and end-of-turn tokens removed.
fn visible(text: &str) -> String {
    let mut t = text.to_string();
    for m in ["<end_of_turn>", "<start_function_response>", "<eos>"] {
        t = t.replace(m, "");
    }
    t.trim().to_string()
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
    /// Generates the model's reply to `messages`, streaming it to stdout.
    fn reply(&mut self, messages: &Value, tools: &Value) -> anyhow::Result<String> {
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
        // When the first token is out: the rate after it is pure decode (one token per step),
        // which is what streaming speed feels like.
        let mut first_at: Option<f64> = None;
        print!("model> ");
        loop {
            let tok = argmax(&logits);
            out.push(tok);
            first_at.get_or_insert_with(|| t1.elapsed().as_secs_f64());
            let text = visible(&self.chat.decode(&out)?);
            if text.len() > shown && text.is_char_boundary(shown) {
                print!("{}", &text[shown..]);
                std::io::stdout().flush()?;
                shown = text.len();
            }
            if self.stop.contains(&tok) || out.len() >= self.max_tokens {
                break;
            }
            anyhow::ensure!(
                self.session.cache.len < self.session.cache.capacity,
                "conversation reached --context {} tokens; /reset to start over",
                self.session.cache.capacity
            );
            logits = self.session.step(self.device, &mut self.one, &[tok])?;
        }
        let gen_s = t1.elapsed().as_secs_f64();
        println!();
        let first = first_at.unwrap_or(0.0);
        let decode_rate = if out.len() > 1 {
            (out.len() - 1) as f64 / (gen_s - first).max(1e-9)
        } else {
            0.0
        };
        eprintln!(
            "  [{} new prompt tokens in {:.2} s ({:.1} tokens/s); first token after {:.2} s; \
             {} tokens out at {:.1} tokens/s ({:.1} decode); context {}/{}]",
            fresh,
            prompt_s,
            fresh as f64 / prompt_s.max(1e-9),
            prompt_s + first,
            out.len(),
            out.len() as f64 / gen_s.max(1e-9),
            decode_rate,
            self.session.cache.len,
            self.session.cache.capacity
        );
        self.chat.decode(&out)
    }
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    let cfg = GemmaConfig::load(&args.model.join("config.json"))?;
    let chat = Chat::load(&args.model)?;
    let tools: Value = match &args.tools {
        Some(p) => serde_json::from_str(&std::fs::read_to_string(p)?)?,
        None => demo_tools(),
    };
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
    let names: Vec<&str> = tools
        .as_array()
        .map(|a| {
            a.iter()
                .filter_map(|t| t["function"]["name"].as_str())
                .collect()
        })
        .unwrap_or_default();
    eprintln!(
        "FunctionGemma on {} (ready in {:.1} s). Tools: {}. /reset, /quit.",
        runtime::BACKEND_NAME,
        t0.elapsed().as_secs_f64(),
        names.join(", ")
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
    let base: Vec<Value> = args
        .system
        .iter()
        .map(|s| json!({"role": "system", "content": s}))
        .collect();
    let mut messages = base.clone();

    while let Some(line) = read_line("you> ")? {
        let line = line.trim();
        match line {
            "" => continue,
            "/quit" | "/exit" => break,
            "/reset" => {
                messages = base.clone();
                eprintln!("  [conversation cleared]");
                continue;
            }
            _ => {}
        }
        messages.push(json!({"role": "user", "content": line}));
        let mut reply = engine.reply(&Value::Array(messages.clone()), &tools)?;

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
                if let Some(result) = (!args.manual_tools)
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
            reply = engine.reply(&Value::Array(messages.clone()), &tools)?;
        }
    }
    Ok(())
}
