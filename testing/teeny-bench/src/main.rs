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

//! `teeny-bench`: benchmark a teenygrad chat model's prompt processing and generation.

use std::sync::Arc;
use std::time::Instant;

use anyhow::Context;
use clap::{Parser, ValueEnum};
use teeny_bench::{ApiTarget, LocalTarget, Target, Test, Workload};
use teeny_llm::chat::EchoChatModel;

#[derive(Debug, Clone, Copy, ValueEnum)]
enum Output {
    /// A markdown table, llama-bench style.
    Md,
    /// Everything, every run included.
    Json,
}

#[derive(Debug, Parser)]
#[command(version, about)]
struct Args {
    /// A model to load and benchmark in this process, by tag. The choices are the models
    /// this build includes (`--features functiongemma3,cuda`).
    #[arg(long, short, ignore_case = true, value_parser = model_parser(), conflicts_with_all = ["echo", "api_url"])]
    model: Option<teeny_llm::models::Model>,

    /// The weights folder for `--model`. Defaults to its snapshot in the Hugging Face cache.
    #[arg(long, requires = "model")]
    model_dir: Option<std::path::PathBuf>,

    /// The longest conversation, in tokens, the model keeps (its KV cache).
    #[arg(long, requires = "model")]
    context: Option<u32>,

    /// Benchmark the echo chat model in this process (to try the tool; its numbers mean
    /// nothing).
    #[arg(long, conflicts_with = "api_url")]
    echo: bool,

    /// Benchmark a model a running teeny-api serves (`http://127.0.0.1:8000`).
    #[arg(long)]
    api_url: Option<String>,

    /// The model to ask teeny-api for. Defaults to the first it lists.
    #[arg(long, requires = "api_url")]
    api_model: Option<String>,

    /// Prompt processing tests: the prompt lengths, in tokens (multiples of 16 suit
    /// FunctionGemma, which runs prompts in 16-token chunks).
    #[arg(long, short, value_delimiter = ',', default_value = "64,256")]
    prompt: Vec<u32>,

    /// Generation tests: the tokens to generate.
    #[arg(long = "gen", short = 'n', value_delimiter = ',', default_value = "64")]
    generate: Vec<u32>,

    /// Measured runs per test.
    #[arg(long, short, default_value_t = 5)]
    repetitions: u32,

    /// Unmeasured runs per test first.
    #[arg(long, default_value_t = 1)]
    warmup: u32,

    /// How to report.
    #[arg(long, short, value_enum, default_value_t = Output::Md)]
    output: Output,
}

fn model_parser() -> impl clap::builder::TypedValueParser<Value = teeny_llm::models::Model> {
    use clap::builder::{PossibleValuesParser, TypedValueParser};
    PossibleValuesParser::new(teeny_llm::models::Model::tags())
        .try_map(|tag| tag.parse::<teeny_llm::models::Model>())
}

/// The backend this build runs models on.
fn backend() -> &'static str {
    if cfg!(feature = "qemu") {
        "riscv under qemu"
    } else if cfg!(feature = "riscv") {
        "riscv"
    } else if cfg!(feature = "cuda") {
        "cuda"
    } else {
        "no backend"
    }
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    let mut load_ms = None;
    let target: Box<dyn Target> = if let Some(url) = &args.api_url {
        Box::new(ApiTarget::connect(url, args.api_model.as_deref()).await?)
    } else if let Some(model) = args.model {
        let mut options = teeny_llm::models::LoadOptions::default()
            .with_progress(move |what| eprintln!("[{model}] {what}"));
        if let Some(dir) = &args.model_dir {
            options = options.with_model_dir(dir.clone());
        }
        if let Some(n) = args.context {
            options = options.with_context_length(n);
        }
        let started = Instant::now();
        let loaded = tokio::task::spawn_blocking(move || model.load(&options))
            .await
            .context("the loading thread panicked")?
            .with_context(|| format!("loading {model}"))?;
        load_ms = Some(started.elapsed().as_secs_f64() * 1e3);
        let chat = loaded
            .chat
            .with_context(|| format!("{model} serves no chat API"))?;
        Box::new(LocalTarget::new(
            chat,
            &format!("in-process, {}", backend()),
        ))
    } else if args.echo {
        Box::new(LocalTarget::new(Arc::new(EchoChatModel), "in-process"))
    } else {
        anyhow::bail!(
            "nothing to benchmark: pass --model <tag> (this build has: {}), --echo, or --api-url",
            match teeny_llm::models::Model::tags() {
                tags if tags.is_empty() =>
                    "none; rebuild with e.g. --features functiongemma3,cuda".to_owned(),
                tags => tags.join(", "),
            }
        );
    };

    let tests = args
        .prompt
        .iter()
        .map(|&n| Test::Prompt(n))
        .chain(args.generate.iter().map(|&n| Test::Generation(n)))
        .collect();
    let workload = Workload {
        tests,
        repetitions: args.repetitions,
        warmup: args.warmup,
    };
    let report = teeny_bench::run(target.as_ref(), &workload, load_ms, |what| {
        eprintln!("{what}")
    })
    .await?;
    match args.output {
        Output::Md => print!("{}", report.markdown()),
        Output::Json => println!("{}", serde_json::to_string_pretty(&report)?),
    }
    Ok(())
}
