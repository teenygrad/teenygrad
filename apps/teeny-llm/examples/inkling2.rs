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

//! A terminal chat with Inkling-2, one token at a time.
//!
//! Loading is the slow part and happens once: around 22 GB of parameters go into 24
//! layer graphs plus the embedding, the messages and the head. After that a token is 48
//! layer executions, a few seconds on a RISC-V board.
//!
//! The reply opens with a thinking block: the checkpoint's own chat template injects a
//! "Thinking effort level" system turn, so the model reasons before it answers. Pass
//! `--effort none` to ask for the least of it.

use std::io::Write;
use std::path::PathBuf;
use std::time::Instant;

use clap::Parser;
use serde_json::json;
use teeny_llm::models::inkling2::Inkling2Config;
use teeny_llm::models::inkling2::chat::Chat;
use teeny_llm::models::inkling2::engine::Engine;
use teeny_llm::models::inkling2::weights::Checkpoint;
use teeny_test::model::runtime;

#[derive(Parser, Debug)]
#[command(about = "Chat with Inkling-2 in the terminal")]
struct Args {
    /// The checkpoint folder: config.json, the safetensors shards, tokenizer.json.
    /// Defaults to $INKLING2_MODEL_DIR.
    #[arg(long)]
    model: Option<PathBuf>,
    /// Stop after this many generated tokens. Without it a reply runs until the
    /// model ends it, or the context is full.
    #[arg(long)]
    max_tokens: Option<usize>,
    /// Room for the whole conversation, in tokens. The default is the model's
    /// sliding window, within which every layer sees the whole conversation.
    #[arg(long, default_value_t = 1024)]
    context: usize,
    /// One prompt, answered and then exit. Without it, the chat is interactive.
    #[arg(long)]
    prompt: Option<String>,
    /// How much the model should think: none, minimal, low, medium, high, xhigh, max.
    #[arg(long)]
    effort: Option<String>,
    /// Prompt tokens per engine step, up to the engine's maximum: a chunk reads each
    /// weight once for all its tokens. 1 feeds the prompt a token at a time.
    #[arg(long, default_value_t = teeny_llm::models::inkling2::engine::MAX_TOKENS)]
    prompt_chunk: usize,
    /// Check rollback: step each generated token together with two junk tokens, keep
    /// the first row's logits and truncate the junk away. The output must be the
    /// same as without it.
    #[arg(long)]
    rewind_check: bool,
    /// A system turn to open the conversation with.
    #[arg(long)]
    system: Option<String>,
}

fn read_line(prompt: &str) -> anyhow::Result<Option<String>> {
    print!("{prompt}");
    std::io::stdout().flush()?;
    let mut line = String::new();
    if std::io::stdin().read_line(&mut line)? == 0 {
        return Ok(None);
    }
    Ok(Some(line.trim_end().to_string()))
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    let dir = match args.model.clone() {
        Some(d) => d,
        None => PathBuf::from(
            std::env::var("INKLING2_MODEL_DIR")
                .map_err(|_| anyhow::anyhow!("pass --model or set INKLING2_MODEL_DIR"))?,
        ),
    };
    let chat = Chat::load(&dir)?;
    let full = Inkling2Config::load(&dir.join("config.json"))?;
    let cfg = full.text_config.clone();
    let mut ck = Checkpoint::open(&dir)?;

    let device = runtime::open()?;
    eprint!("loading {} layers ... ", cfg.num_hidden_layers);
    std::io::stderr().flush()?;
    let t0 = Instant::now();
    // The reference applies no muP scaling to the logits: it is the unembedding then
    // the softcap, so the multiplier is 1.
    let mut engine = Engine::load(&device, cfg, 1.0, &mut ck, args.context)?;
    eprintln!(
        "{:.1} s, {} executions per token",
        t0.elapsed().as_secs_f64(),
        engine.executions()
    );

    let mut messages: Vec<serde_json::Value> = Vec::new();
    if let Some(system) = &args.system {
        messages.push(json!({"role": "system", "content": system}));
    }
    let mut one_shot = args.prompt.clone();

    loop {
        let line = match one_shot.take() {
            Some(p) => p,
            None => match read_line("\nyou> ")? {
                None => break,
                Some(l) if l.is_empty() => continue,
                Some(l) if l == "/quit" => break,
                Some(l) => l,
            },
        };
        messages.push(json!({"role": "user", "content": line}));
        let prompt = chat.prompt(&json!(messages), args.effort.as_deref())?;

        // Everything already in the cache that still matches is kept; the rest is new.
        let fresh = prompt.len()
            - engine
                .tokens
                .iter()
                .zip(&prompt)
                .take_while(|(a, b)| *a == *b)
                .count();
        anyhow::ensure!(
            engine.tokens.len() + fresh <= args.context,
            "conversation needs more than --context {} tokens",
            args.context
        );
        anyhow::ensure!(
            fresh == prompt.len() - engine.tokens.len(),
            "the prompt diverges from what is cached; this demo cannot rewind, restart it"
        );

        let t_prompt = Instant::now();
        let mut logits = Vec::new();
        let chunk = args.prompt_chunk.clamp(1, teeny_llm::models::inkling2::engine::MAX_TOKENS);
        let mut done = 0usize;
        for tokens in prompt[engine.tokens.len()..].chunks(chunk) {
            logits = engine.step_many(&device, tokens)?;
            done += tokens.len();
            eprint!("\r  prompt {done}/{fresh}");
            std::io::stderr().flush()?;
        }
        let prompt_s = t_prompt.elapsed().as_secs_f64();
        eprintln!(
            "\r  prompt {fresh} tokens in {prompt_s:.1} s ({:.2} tokens/s)",
            fresh as f64 / prompt_s.max(1e-9)
        );

        print!("model> ");
        std::io::stdout().flush()?;
        let t_gen = Instant::now();
        let mut out: Vec<u32> = Vec::new();
        let mut shown = 0usize;
        loop {
            let next = engine.pick(&logits);
            out.push(next);
            let text = chat.decode(&out)?;
            if text.len() > shown && text.is_char_boundary(shown) {
                print!("{}", &text[shown..]);
                std::io::stdout().flush()?;
                shown = text.len();
            }
            if next == chat.eos || args.max_tokens.is_some_and(|m| out.len() >= m) {
                break;
            }
            if engine.len() >= args.context {
                eprintln!("\n  [out of context]");
                break;
            }
            if args.rewind_check {
                engine.keep_undo(true);
                let mut rows = engine.step_many_logits(&device, &[next, 13, 42])?;
                engine.truncate(engine.len() - 2)?;
                logits = rows.swap_remove(0);
            } else {
                logits = engine.step(&device, next)?;
            }
        }
        let gen_s = t_gen.elapsed().as_secs_f64();
        println!();
        eprintln!(
            "  [{} tokens in {gen_s:.1} s, {:.2} tokens/s; context {}/{}]",
            out.len(),
            out.len() as f64 / gen_s.max(1e-9),
            engine.len(),
            args.context
        );
        messages.push(json!({"role": "assistant", "content": chat.decode_visible(&out)?}));
        if args.prompt.is_some() {
            break;
        }
    }
    Ok(())
}
