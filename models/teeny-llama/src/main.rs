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

use std::{path::PathBuf, time::Duration};

use teeny_data::hf::{DownloadConfig, download_huggingface_model};

#[tokio::main]
async fn main() {
    println!("Hello, world!");

    let config = DownloadConfig {
        model_id: "Qwen/Qwen3-1.7B".to_string(),
        output_dir: PathBuf::from("/tmp/downloaded_llama"),
        include_tokenizer: true,
        include_config: true,
        include_weights: true,
        revision: "main".to_string(),
        auth_token: std::env::var("HF_TOKEN").ok(),
        timeout: Duration::from_secs(300),
        max_concurrent: 4,
    };

    let result = download_huggingface_model(config).await;
    if let Err(e) = result {
        eprintln!("Error: {:?}", e);
    }
}
