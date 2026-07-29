/*
 * Copyright (c) 2026 Teenygrad.
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

//! `teeny-quant inspect`: lists a checkpoint's tensors (and, if present, its
//! `quantization_config`).

use std::path::PathBuf;

use anyhow::{Context, Result};
use clap::Args;

use crate::format;
use crate::read::read_metadata;

/// `teeny-quant inspect` arguments.
#[derive(Args, Debug)]
pub struct InspectArgs {
    /// `.safetensors` file to inspect.
    pub path: PathBuf,
}

/// Runs `teeny-quant inspect`.
pub fn run(args: InspectArgs) -> Result<()> {
    let mapped = teeny_data::safetensors::SafeTensors::from_pretrained(&args.path)
        .with_context(|| format!("failed to open '{}'", args.path.display()))?;
    let tensors = mapped
        .tensors()
        .with_context(|| format!("failed to read tensor headers from '{}'", args.path.display()))?;

    let mut names = tensors.names();
    names.sort();

    println!("{:<55} {:<10} {:>14}  shape", "name", "dtype", "bytes");
    let mut total_bytes = 0usize;
    for name in &names {
        let view = tensors
            .tensor(name)
            .with_context(|| format!("reading tensor '{name}'"))?;
        let nbytes = view.data().len();
        total_bytes += nbytes;
        println!(
            "{:<55} {:<10} {:>14}  {:?}",
            name,
            format!("{:?}", view.dtype()),
            nbytes,
            view.shape()
        );
    }
    println!(
        "\n{} tensor(s), {:.2} MiB total",
        names.len(),
        total_bytes as f64 / (1024.0 * 1024.0)
    );

    let metadata = read_metadata(&args.path)?;
    if let Some(config) = format::config_from_metadata(&metadata)? {
        println!("\nquantization_config:\n{}", serde_json::to_string_pretty(&config)?);
    }

    Ok(())
}
