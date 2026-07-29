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

//! Smoke test / reference implementation for `teeny-cli`: AOT-compiles
//! `teeny-vision`'s LeNet-5 against the CUDA backend. This is the pattern a
//! downstream project (e.g. `vision-rs`) copies into its own binary —
//! `#[command(flatten)]` [`teeny_cli::AotArgs`] into your own `clap` struct,
//! build your model + sample input, and call [`teeny_cli::aot_compile`].
//!
//! Needs a `teenyc` binary, resolved via `teeny_compiler::compiler::find_teenyc`
//! (`TEENYC_PATH`, or a `rustup`-linked toolchain otherwise). Typically driven via
//! `cargo teeny aot --bin teeny-cli --device cuda --options "capability=sm_90"`
//! (from `cargo-teeny`), but can also be run directly.

use anyhow::Result;
use clap::Parser;
use teeny_core::graph::DtypeRepr;
use teeny_core::model::LoweringMode;
use teeny_kernels::graph::TritonLowering;
use teeny_vision::mnist::mnist_lenet5;

#[derive(Parser)]
#[command(
    name = "teeny-cli",
    about = "Ahead-of-time compile a model's kernels for a given device/config."
)]
struct Cli {
    #[command(flatten)]
    aot: teeny_cli::AotArgs,
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    let model = mnist_lenet5::<f32>();
    let lowering = TritonLowering::new();

    teeny_cli::aot_compile(
        &model,
        DtypeRepr::F32,
        vec![None, Some(1), Some(28), Some(28)],
        &lowering,
        LoweringMode::Inference,
        &cli.aot,
    )?;

    println!(
        "AOT compile complete (device={}, cache={})",
        cli.aot.device,
        cli.aot.resolve_cache_dir().display()
    );
    Ok(())
}
