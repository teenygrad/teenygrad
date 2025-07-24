/*
 * Copyright (c) 2025 Teenygrad. All rights reserved.
 *
 * This program is free software: you can redistribute it and/or modify it
 * under the terms of the GNU General Public License as published by the
 * Free Software Foundation, either version 3 of the License, or (at your
 * option) any later version.
 *
 * This program is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program. If not, see <https://www.gnu.org/licenses/>.
 */

#[allow(dead_code)]
use std::path::PathBuf;
use std::{path::Path, sync::Arc};

use clap::{Parser, ValueEnum};
use ndarray::Array1;

use teeny_core::graph::tensor;
#[allow(unused_imports)]
use teeny_hf::{
    error::Error,
    transformer::{
        self,
        model::qwen::qwen3::qwen3_config::Qwen3Config,
        tokenizer::{self, tokenizer_config::TokenizerConfig},
        util::template,
    },
    util::download::{DownloadConfig, download_model},
};
#[allow(unused_imports)]
use teeny_nlp::tokenizer::Message;

use teeny_hf::{
    error::Result, transformer::model::qwen::qwen2::qwen2_model::QwenModelInputsBuilder,
};
use tracing::info;
use tracing_subscriber::{self, EnvFilter};

#[derive(Parser)]
#[command(
    name = "hf",
    about = "Run huggingface models",
    version,
    long_about = "A CLI tool for downloading and running huggingface models."
)]
struct Cli {
    /// The model identifier (e.g., "Qwen/Qwen3-1.7B", "bert-base-uncased")
    #[arg(value_name = "MODEL", default_value = "Qwen/Qwen3-1.7B")]
    model: String,

    /// Directory to cache/download the model files
    #[arg(value_name = "CACHE_DIR")]
    cache_dir: Option<PathBuf>,

    /// Maximum concurrent downloads
    #[arg(short, long, default_value = "1")]
    max_concurrent: usize,

    /// Components to download
    #[arg(short, long, value_enum, default_values_t = vec![Component::Tokenizer, Component::Config, Component::Weights])]
    components: Vec<Component>,

    /// The revision/branch to download
    #[arg(short, long, default_value = "main")]
    revision: String,

    /// Disable progress display
    #[arg(long, default_value = "true")]
    show_progress: bool,

    /// The log level
    #[arg(value_enum, short = 'l', long = "log-level", default_value = "info")]
    log_level: LogLevel,
}

#[derive(ValueEnum, Clone, Debug, PartialEq)]
enum Component {
    Tokenizer,
    Config,
    Weights,
}

#[derive(Debug, Clone, ValueEnum)]
enum LogLevel {
    Error,
    Warn,
    Info,
    Debug,
    Trace,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();

    let log_level = match cli.log_level {
        LogLevel::Error => "error",
        LogLevel::Warn => "warn",
        LogLevel::Info => "info",
        LogLevel::Debug => "debug",
        LogLevel::Trace => "trace",
    };

    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new(log_level)),
        )
        .init();

    // Determine which components to include based on CLI arguments
    let include_tokenizer = cli.components.contains(&Component::Tokenizer);
    let include_config = cli.components.contains(&Component::Config);
    let include_weights = cli.components.contains(&Component::Weights);

    // Use auth token from CLI or environment variable
    let auth_token = std::env::var("HF_TOKEN").ok();
    let cache_dir = cli.cache_dir.clone().unwrap_or_else(|| {
        let home = std::env::var("HOME").unwrap_or_else(|_| ".".to_string());
        PathBuf::from(format!("{home}/.teeny/cache/models"))
    });

    let config = DownloadConfig {
        model_id: cli.model.clone(),
        cache_dir: cache_dir.as_path(),
        include_tokenizer,
        include_config,
        include_weights,
        revision: cli.revision,
        auth_token,
        max_concurrent: cli.max_concurrent,
        show_progress: cli.show_progress,
    };

    info!("Downloading model: {}", cli.model);
    download_model(config).await?;

    info!("Running model: {}", cli.model);
    run_model(&cli.model, cache_dir.as_path()).await?;

    Ok(())
}

async fn run_model(model_id: &str, cache_dir: &Path) -> Result<()> {
    let tokenizer_config = TokenizerConfig::from_pretrained(model_id, cache_dir)?;
    let tokenizer = tokenizer::from_pretrained(model_id, cache_dir)?;
    let model = transformer::model::from_pretrained(model_id, cache_dir)?;
    let _config = Qwen3Config::from_pretrained(model_id, cache_dir)?;

    let prompt = "Give me a short introduction to large language model.";
    let messages = [Message::new("user", prompt)];

    let text = template::apply_chat_template(&tokenizer_config.chat_template, &messages, &[]);
    let encoded_inputs = tokenizer
        .encode(text, false)
        .map_err(Error::TokenizerError)?;
    let encoded_ids = tensor(
        Array1::from(
            encoded_inputs
                .get_ids()
                .iter()
                .map(|x| *x as usize)
                .collect::<Vec<_>>(),
        )
        .into_dyn(),
    );

    let model_inputs = QwenModelInputsBuilder::default()
        .input_ids(Some(encoded_ids))
        .build()
        .map_err(|e| Error::BuilderError(Arc::new(e)))?;

    let generated_ids = model
        .forward(model_inputs)?
        .realize()?
        .iter()
        .map(|x| *x as u32)
        .collect::<Vec<_>>();

    let thinking_content = tokenizer
        .decode(&generated_ids[..], false)
        .map_err(Error::TokenizerError)?;

    let content = tokenizer
        .decode(&generated_ids[thinking_content.len()..], false)
        .map_err(Error::TokenizerError)?;

    println!("thinking content: {thinking_content}");
    println!("content: {content}");

    todo!("run model");
}
