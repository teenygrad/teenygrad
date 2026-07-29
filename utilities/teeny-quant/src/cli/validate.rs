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

//! `teeny-quant validate`: per-tensor quantization error metrics (see [`crate::validate`]).

use std::path::PathBuf;

use anyhow::{Context, Result};
use clap::Args;

use crate::validate::validate_checkpoint;

/// `teeny-quant validate` arguments.
#[derive(Args, Debug)]
pub struct ValidateArgs {
    /// The original (unquantized) `.safetensors` checkpoint.
    #[arg(long)]
    pub original: PathBuf,

    /// The quantized `.safetensors` checkpoint (produced by `teeny-quant quantize`).
    #[arg(long)]
    pub quantized: PathBuf,

    /// Exit non-zero if any tensor's max absolute error exceeds this.
    #[arg(long)]
    pub max_abs_error_threshold: Option<f32>,
}

/// Runs `teeny-quant validate`.
pub fn run(args: ValidateArgs) -> Result<()> {
    let reports = validate_checkpoint(&args.original, &args.quantized).with_context(|| {
        format!(
            "failed to validate '{}' against '{}'",
            args.quantized.display(),
            args.original.display()
        )
    })?;

    println!(
        "{:<55} {:>14} {:>14} {:>10}",
        "tensor", "max_abs_err", "mean_abs_err", "sqnr_db"
    );
    let mut flagged = 0usize;
    for r in &reports {
        println!(
            "{:<55} {:>14.6} {:>14.6} {:>10.2}",
            r.name, r.max_abs_error, r.mean_abs_error, r.sqnr_db
        );
        if args.max_abs_error_threshold.is_some_and(|t| r.max_abs_error > t) {
            flagged += 1;
        }
    }
    println!("\n{} tensor(s) checked", reports.len());

    if let Some(threshold) = args.max_abs_error_threshold
        && flagged > 0
    {
        anyhow::bail!("{flagged} tensor(s) exceeded max-abs-error threshold {threshold}");
    }

    Ok(())
}
