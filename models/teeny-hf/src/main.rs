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

use std::path::PathBuf;

use clap::{Parser, ValueEnum};
use teeny_hf::{
    model, tokenizer,
    util::model::download::{DownloadConfig, default_progress_callback, download_model},
};
use teeny_nlp::tokenizer::Message;

#[derive(Parser)]
#[command(
    name = "hf",
    about = "Run huggingface models",
    version,
    long_about = "A CLI tool for downloading and running huggingface models."
)]
struct Cli {
    /// The model identifier (e.g., "Qwen/Qwen3-1.7B", "bert-base-uncased")
    #[arg(value_name = "MODEL", required = true)]
    model: String,

    /// Directory to cache/download the model files
    #[arg(value_name = "CACHE_DIR", default_value = "/tmp/models")]
    cache_dir: PathBuf,
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
}

#[derive(ValueEnum, Clone, Debug, PartialEq)]
enum Component {
    Tokenizer,
    Config,
    Weights,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cli = Cli::parse();

    // Determine which components to include based on CLI arguments
    let include_tokenizer = cli.components.contains(&Component::Tokenizer);
    let include_config = cli.components.contains(&Component::Config);
    let include_weights = cli.components.contains(&Component::Weights);

    // Use auth token from CLI or environment variable
    let auth_token = std::env::var("HF_TOKEN").ok();

    let config = DownloadConfig {
        model_id: cli.model.clone(),
        output_dir: cli.cache_dir.clone(),
        include_tokenizer,
        include_config,
        include_weights,
        revision: cli.revision,
        auth_token,
        max_concurrent: cli.max_concurrent,
        progress_callback: if cli.show_progress {
            Some(default_progress_callback())
        } else {
            None
        },
    };

    println!("Downloading model: {}", cli.model);

    download_model(config).await?;

    run_model(&cli.model, cli.cache_dir.to_str().unwrap()).await?;

    Ok(())
}

async fn run_model(model_id: &str, cache_dir: &str) -> Result<(), Box<dyn std::error::Error>> {
    let tokenizer = tokenizer::from_pretrained(model_id, cache_dir)?;
    let model = model::from_pretrained(model_id, cache_dir)?;

    let prompt = "Give me a short introduction to large language model.";
    let messages = [Message::new("user", prompt)];

    let text = tokenizer.apply_chat_template(&messages, false, true, true);
    let model_inputs = tokenizer.encode(&[text]);
    let generated_ids = model.generate(&model_inputs, 32768);

    let thinking_content = tokenizer.decode(&generated_ids[..]);
    let content = tokenizer.decode(&generated_ids[thinking_content.len()..]);

    println!("thinking content: {}", thinking_content);
    println!("content: {}", content);

    Ok(())
}
