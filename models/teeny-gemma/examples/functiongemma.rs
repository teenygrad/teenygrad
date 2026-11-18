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

//! FunctionGemma on teenygrad: asks the model a question with a set of tools and
//! prints the function call it chooses.
//!
//! ```text
//! cargo run --release -p teeny-gemma --example functiongemma -- \
//!     --model ~/.cache/huggingface/hub/models--google--functiongemma-270m-it/snapshots/<rev> \
//!     --prompt "What is the weather in Paris?" [--tools tools.json]
//! ```
//!
//! `--tools` is a JSON list of OpenAI-style function declarations; without it a demo
//! `get_weather(city)` tool is offered. Decoding is greedy and stops at
//! `<start_function_response>` (the model handing a call to the host), `<end_of_turn>`
//! or `<eos>`.

use std::io::Write;
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
    /// chat_template.jinja).
    #[arg(long)]
    model: PathBuf,
    /// The user's request.
    #[arg(long)]
    prompt: String,
    /// JSON file with the tool declarations.
    #[arg(long)]
    tools: Option<PathBuf>,
    /// Most tokens to generate.
    #[arg(long, default_value_t = 64)]
    max_tokens: usize,
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    let cfg = GemmaConfig::load(&args.model.join("config.json"))?;
    let chat = Chat::load(&args.model)?;
    let tools: Value = match &args.tools {
        Some(p) => serde_json::from_str(&std::fs::read_to_string(p)?)?,
        None => json!([{
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get current weather",
                "parameters": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"],
                },
            },
        }]),
    };
    let messages = json!([{"role": "user", "content": args.prompt}]);
    let prompt = chat.prompt(&messages, &tools)?;
    // <eos>, <start_function_response>, <end_of_turn>, and whatever config.json adds.
    let mut stop = cfg.eos_token_id.ids();
    stop.extend([1, 50, 106]);

    let t0 = Instant::now();
    let weights = Weights::load(&args.model.join("model.safetensors"))?;
    let device = runtime::open()?;
    let capacity = (prompt.len() + args.max_tokens).next_multiple_of(64);
    let mut session = Session::new(cfg.clone(), capacity);
    let mut prefill = CompiledStep::new(
        &device,
        &cfg,
        &weights,
        prompt.len(),
        capacity,
        &session.past,
    )?;
    let mut decode = CompiledStep::new(&device, &cfg, &weights, 1, capacity, &session.past)?;
    drop(weights);
    eprintln!(
        "[{}] loaded and compiled in {:.1} s; prompt {} tokens",
        runtime::BACKEND_NAME,
        t0.elapsed().as_secs_f64(),
        prompt.len()
    );

    let t1 = Instant::now();
    let mut logits = session.step(&device, &mut prefill, &prompt)?;
    let prefill_s = t1.elapsed().as_secs_f64();

    let t2 = Instant::now();
    let mut out = Vec::new();
    let mut shown = 0;
    while out.len() < args.max_tokens {
        let tok = argmax(&logits);
        out.push(tok);
        let text = chat.decode(&out)?;
        print!("{}", &text[shown..]);
        std::io::stdout().flush()?;
        shown = text.len();
        if stop.contains(&tok) {
            break;
        }
        logits = session.step(&device, &mut decode, &[tok])?;
    }
    let decode_s = t2.elapsed().as_secs_f64();
    println!();

    eprintln!(
        "prompt: {:.0} tokens/s ({:.2} s); generation: {} tokens, {:.1} tokens/s",
        prompt.len() as f64 / prefill_s,
        prefill_s,
        out.len(),
        out.len() as f64 / decode_s
    );
    for call in parse_calls(&chat.decode(&out)?)? {
        println!("call: {}({})", call.name, call.args);
    }
    Ok(())
}
