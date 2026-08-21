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

//! Shared `--device`/`--options` CLI configuration for ahead-of-time kernel
//! compilation, meant to be `#[command(flatten)]`-ed into a downstream
//! project's own binary (e.g. a `vision-rs` demo) and driven end-to-end by
//! `cargo teeny aot` in `cargo-teeny`.
//!
//! `--device` selects a backend; everything after that (including
//! `--options`) is opaque here and handed to that backend's own parser —
//! today that's `teeny_cuda::compiler::options::Options::parse` for
//! `--device cuda`. Adding a new backend means adding a match arm in
//! [`aot_compile`], not touching [`AotArgs`].

#![warn(missing_docs)]

#[cfg(feature = "cuda")]
mod hardware_profiles;

use std::path::PathBuf;

use anyhow::{Result, anyhow};
use teeny_core::graph::{DtypeRepr, Shape, SymTensor};
use teeny_core::model::{Lowering, LoweringMode};
use teeny_core::nn::Layer;

/// Default fallback cache directory when neither `--cache-dir` nor
/// `$TEENYC_CACHE_DIR` is set. Matches the default used by the runtime JIT
/// compile path (`teeny_cuda::compiler::compile_kernel`).
const DEFAULT_CACHE_DIR: &str = "/tmp/teenyc_cache";

/// Shared AOT-compile CLI arguments. Flatten this into your own `clap`
/// struct: `#[command(flatten)] aot: teeny_cli::AotArgs`.
#[derive(clap::Args, Clone, Debug)]
pub struct AotArgs {
    /// Target backend to compile for (e.g. `cuda`).
    #[arg(long)]
    pub device: String,

    /// Backend-specific compiler options as comma-separated `key=value`
    /// pairs (e.g. `"capability=sm_90,maxnreg=16"`). Parsed by whichever
    /// backend `--device` selects.
    #[arg(long)]
    pub options: Option<String>,

    /// Directory to write/read compiled kernels. Falls back to
    /// `$TEENYC_CACHE_DIR`, then `/tmp/teenyc_cache` if neither is set.
    #[arg(long)]
    pub cache_dir: Option<PathBuf>,

    /// Recompile even if a cached artifact already exists.
    #[arg(long)]
    pub force: bool,
}

impl AotArgs {
    /// Resolve the effective cache directory: `--cache-dir` > `$TEENYC_CACHE_DIR` > default.
    pub fn resolve_cache_dir(&self) -> PathBuf {
        self.cache_dir
            .clone()
            .or_else(|| std::env::var_os("TEENYC_CACHE_DIR").map(PathBuf::from))
            .unwrap_or_else(|| PathBuf::from(DEFAULT_CACHE_DIR))
    }
}

/// Trace `model` with a symbolic sample input of `input_dtype`/`input_shape`,
/// then ahead-of-time compile every kernel the traced graph references for
/// the backend/config selected by `args`.
///
/// The compiled artifacts are written to `args.resolve_cache_dir()` as a
/// side effect; this returns `()` rather than a backend-specific model type
/// so that callers (and `--device` dispatch here) don't need to unify return
/// types across backends. Callers that need the compiled model in-process
/// (e.g. to immediately `.load()` it onto a live device) should call the
/// backend's own compile entry point directly instead.
#[cfg_attr(not(feature = "cuda"), allow(unused_variables))]
pub fn aot_compile<'a, M, L>(
    model: &M,
    input_dtype: DtypeRepr,
    input_shape: Shape,
    lowering: &L,
    mode: LoweringMode,
    args: &AotArgs,
) -> Result<()>
where
    M: Layer<SymTensor>,
    L: Lowering<'a>,
{
    match args.device.as_str() {
        #[cfg(feature = "cuda")]
        "cuda" => {
            let (input, graph) = SymTensor::input(input_dtype, input_shape);
            let _ = model.call(input);
            let graph = graph.borrow();

            let options = teeny_cuda::compiler::options::Options::parse(
                args.options.as_deref().unwrap_or(""),
            )?;

            // Match the runtime inference path (e.g. `build_infer_fn` in vision-rs's
            // parking-garage demo, and examples/yolo26.rs): lower first, then run
            // Anduin as a separate optimization step over the already-lowered DAG —
            // lowering has no knowledge of optimization, and neither does
            // `teeny_cuda::compiler::aot`, which only knows how to compile an
            // already-lowered `(Dag, Vec<usize>)` pair (`compile_lowered_graph`).
            // No live device is open here (AOT compilation may target a device
            // other than the one doing the compiling), so the hardware profile
            // Anduin schedules against comes from the packaged per-capability
            // defaults in `hardware_profiles.json` instead of a query.
            let (op_dag, graph_to_dag, lowered_graph) =
                lowering.lower_with_mapping(&graph, mode)?;

            use teeny_kernels::graph::{Anduin, GraphOptimizer};
            let hardware = hardware_profiles::hardware_profile_for(
                options.gpu_name,
                options.sm_count,
            )?;
            let (op_dag, graph_to_dag) = Anduin.optimize(op_dag, graph_to_dag, &hardware)?;

            let cache_dir = args.resolve_cache_dir();
            std::fs::create_dir_all(&cache_dir)?;

            teeny_cuda::compiler::aot::compile_lowered_graph(
                op_dag,
                graph_to_dag,
                &lowered_graph,
                lowering,
                &options,
                cache_dir.to_string_lossy().as_ref(),
                args.force,
            )?;

            Ok(())
        }
        other => Err(anyhow!(
            "unsupported --device '{other}'; supported backends: {}",
            supported_backends()
        )),
    }
}

fn supported_backends() -> &'static str {
    #[cfg(feature = "cuda")]
    {
        "cuda"
    }
    #[cfg(not(feature = "cuda"))]
    {
        "(none enabled; rebuild teeny-cli with --features cuda)"
    }
}
