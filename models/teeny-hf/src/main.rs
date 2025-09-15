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

use std::path::Path;
use std::path::PathBuf;

use clap::{Parser, ValueEnum};
use serde::Deserialize;
use teeny_core::types::bf16::bf16;
use teeny_hf::transformer;
use tracing::info;
use tracing_subscriber::{self, EnvFilter};

use teeny_hf::{
    error::{Error, Result},
    util::download::{DownloadConfig, download_model},
};

#[derive(Parser)]
#[command(
    name = "thf",
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

#[derive(Debug, Deserialize)]
#[serde(rename_all = "lowercase")]
enum ModelType {
    Qwen3,
}

#[derive(Debug, Deserialize)]
struct ModelConfig {
    pub model_type: ModelType,
    pub torch_dtype: String,
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
    let config = read_model_config(model_id, cache_dir).await?;
    match config.model_type {
        ModelType::Qwen3 => {
            assert_eq!(config.torch_dtype, "bfloat16");
            transformer::model::run_qwen3::<bf16>(model_id, cache_dir)?;
        }
    }

    Ok(())
}

async fn read_model_config(model_id: &str, cache_dir: &Path) -> Result<ModelConfig> {
    let config_path = format!("{}/{model_id}/config.json", cache_dir.to_string_lossy());
    let config_str = std::fs::read_to_string(config_path).map_err(Error::IoError)?;
    let config: ModelConfig = serde_json::from_str(&config_str).map_err(Error::SerdeError)?;

    Ok(config)
}
