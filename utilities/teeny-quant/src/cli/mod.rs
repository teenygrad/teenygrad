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

//! The `teeny-quant` binary's command-line interface.

pub mod inspect;
pub mod quantize;
pub mod validate;

use clap::{Parser, Subcommand};

/// Quantize `.safetensors` model checkpoints for deployment.
#[derive(Parser, Debug)]
#[command(
    name = "teeny-quant",
    about = "Quantize .safetensors model checkpoints for deployment."
)]
pub struct Cli {
    /// The subcommand to run.
    #[command(subcommand)]
    pub command: Command,
}

/// `teeny-quant` subcommands.
#[derive(Subcommand, Debug)]
pub enum Command {
    /// Quantize a checkpoint.
    Quantize(quantize::QuantizeArgs),
    /// List a checkpoint's tensors (and, if present, its `quantization_config`).
    Inspect(inspect::InspectArgs),
    /// Compare a quantized checkpoint against its original for per-tensor error.
    Validate(validate::ValidateArgs),
}

impl Cli {
    /// Runs the selected subcommand.
    pub fn run(self) -> anyhow::Result<()> {
        match self.command {
            Command::Quantize(args) => quantize::run(args),
            Command::Inspect(args) => inspect::run(args),
            Command::Validate(args) => validate::run(args),
        }
    }
}
