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

//! FunctionGemma on teenygrad in the terminal, on whichever backend the build selects
//! (CUDA, or RISC-V on a board), in one of two modes:
//!
//! ```text
//! cargo run --release -p teeny-llm --example chat -- [--tools tools.json]
//! cargo run --release -p teeny-llm --example chat -- --mode tool \
//!     --prompt "What is the weather in Paris?"
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
//! In both, `--tools` is a JSON list of OpenAI-style function declarations
//! (default: the demo tools above).
//!
//! Decoding is greedy and stops at `<start_function_response>` (the model handing a call
//! to the host), `<end_of_turn>` or `<eos>`.

use std::io::{BufRead, Write};
use std::path::PathBuf;
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
    /// `chat` for a conversation, `tool` for a single request's function calls.
    #[arg(long, value_enum, default_value_t = Mode::Chat)]
    mode: Mode,
    /// The request, in `tool` mode.
    #[arg(long, required_if_eq("mode", "tool"))]
    prompt: Option<String>,
    /// A JSON file with the tool declarations (OpenAI style). Defaults to a few demo tools:
    /// weather, timer, messaging.
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

/// The part of a reply meant for a reader. A function call is the model addressing the
/// runtime, not the reader, so it is shown as a `call:` line instead and never as prose.
fn speech(text: &str) -> String {
    let t = visible(text);
    match t.find("<start_function_call>") {
        Some(i) => t[..i].trim_end().to_string(),
        None => t,
    }
}

/// One generation's measurements as one line: prompt and decode speed, and how much of
/// the context the conversation uses.
fn report(timings: &Timings, context_used: usize, context_total: usize) -> String {
    format!(
        "  [{} new prompt tokens in {:.2} s ({:.1} tokens/s); first token after {:.2} s; \
         {} tokens out at {:.1} tokens/s decode; context {}/{}]",
        timings.prompt_tokens,
        timings.prompt.as_secs_f64(),
        timings.prompt_tokens_per_second(),
        timings.first_token.as_secs_f64(),
        timings.generated_tokens,
        timings.generation_tokens_per_second(),
        context_used,
        context_total
    )
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
        report(
            &stats,
            engine.session.cache.len(),
            engine.session.cache.capacity()
        )
    );
    Ok(reply)
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    anyhow::ensure!(
        args.mode == Mode::Tool || args.prompt.is_none(),
        "--prompt is for --mode tool; in chat mode, type the request at the prompt"
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
    let tools: Value = match &args.tools {
        Some(p) => serde_json::from_str(&std::fs::read_to_string(p)?)?,
        None => demo_tools(),
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
    }
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
