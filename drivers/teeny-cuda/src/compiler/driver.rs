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

use teeny_compiler::compiler::backend::llvm::compiler::{LlvmCompiler, LogLevel};
use teeny_core::compiler::Compiler;
use teeny_core::device::program::Kernel;
use teeny_core::graph::Graph;
use teeny_core::model::{Lowering, LoweringMode};

use crate::compiler::graph::CudaGraphCompiler;
use crate::compiler::target::{Capability, Target};
use crate::errors::Result;
use crate::model::CudaModel;

/// Highest SM version our Triton MLIR codegen has validated support for.
///
/// LLVM's NVPTX backend emits `.target sm_NNNa` for each architecture:
///   sm_90  → sm_90a  (Hopper:   TMA, wgmma)
///   sm_100 → sm_100a (Blackwell DC: B100/B200/GB200)
///   sm_120 → sm_120a (Blackwell consumer: RTX 50xx)
///
/// Architecture-specific PTX (the `a` suffix) runs only on its own
/// architecture and forward via native execution — NOT via driver JIT
/// cross-architecture.  Each GPU therefore needs code compiled for its own SM
/// version to avoid `ptxas fatal: ... cannot be compiled to future architecture`.
///
/// The teenyc backend (LLVM 20+) validates all SM versions up to sm_120.
/// If a future architecture is released before the backend adds support,
/// extend this constant and the match arm below.
#[allow(dead_code)]
const MAX_CODEGEN_CAPABILITY: Capability = Capability::Sm120;

/// Compiles `kernel` for `target` via the LLVM backend, using the `teenyc` binary resolved by
/// [`teeny_compiler::compiler::find_teenyc`] and the default cache directory.
///
/// - `force`: recompile even if a cached artifact exists.
/// - `debug`: when `true`, enable `teenyc` pipeline-stage logging (`ttir` / `ttgpuir` / `llir` /
///   `llvmir` / `ptx`) to stderr via a thread-local `tracing` subscriber. Pair with
///   `--nocapture` (and usually `force = true`, otherwise a cache hit emits nothing), e.g.:
///
///   ```text
///   cargo test -p teeny-kernels --test test_relu --features cuda \
///     test_relu -- --nocapture 2>pipeline.log
///   ```
pub fn compile_kernel(
    kernel: &impl Kernel,
    target: &Target,
    force: bool,
    debug: bool,
) -> Result<String> {
    let compiler = make_llvm_compiler(target, debug)?;
    with_pipeline_logging(debug, || compiler.compile(kernel, target, force))
}

/// Compiles a lowered `graph` to a [`CudaModel`] via [`CudaGraphCompiler`], using the `teenyc`
/// binary resolved by [`teeny_compiler::compiler::find_teenyc`] and the default cache directory.
///
/// This is the graph-level counterpart to [`compile_kernel`]: it handles `ExecutableOp` →
/// `Kernel` adaptation internally so callers do not need an `ExecKernel`-style wrapper.
///
/// - `force`: recompile even if a cached artifact exists.
/// - `debug`: when `true`, enable the same pipeline-stage logging as [`compile_kernel`].
pub fn compile_cuda_graph<'a, L: Lowering<'a>>(
    graph: &Graph,
    lowering: &L,
    target: &Target,
    mode: LoweringMode,
    force: bool,
    debug: bool,
) -> Result<CudaModel<'a>> {
    let compiler = make_llvm_compiler(target, debug)?;
    let graph_compiler = CudaGraphCompiler::new(compiler);
    with_pipeline_logging(debug, || {
        graph_compiler.compile_model(graph, lowering, target, mode, force)
    })
}

fn make_llvm_compiler(target: &Target, debug: bool) -> Result<LlvmCompiler> {
    let teenyc_path = teeny_compiler::compiler::find_teenyc()?;
    let cache_dir = teeny_compiler::compiler::default_cache_dir();
    let effective_cpu = clamp_capability(target.capability).to_string();
    let mut compiler = LlvmCompiler::new(teenyc_path, cache_dir)?.with_target_cpu(effective_cpu);
    if debug {
        compiler = compiler.with_log_level(LogLevel::Debug);
    }
    Ok(compiler)
}

/// When `debug` is set, install a thread-local stderr tracing subscriber for the duration of `f`
/// so `teenyc` pipeline-stage IR is visible under `--nocapture`.
fn with_pipeline_logging<T>(debug: bool, f: impl FnOnce() -> Result<T>) -> Result<T> {
    if !debug {
        return f();
    }
    // Scoped to this thread only, so it doesn't clobber a subscriber another
    // test running concurrently in this binary may have installed.
    let subscriber = tracing_subscriber::fmt()
        .with_max_level(tracing::Level::TRACE)
        .with_ansi(false)
        .with_writer(std::io::stderr)
        .finish();
    tracing::subscriber::with_default(subscriber, f)
}

/// Clamp `cap` to `MAX_CODEGEN_CAPABILITY` for any architecture newer than
/// what the backend supports.  All SM versions up to sm_120 are natively
/// supported, so no clamping is needed for current hardware.  If a new SM
/// version is added to `Capability` before teenyc validates it, add a match
/// arm here that maps it to `MAX_CODEGEN_CAPABILITY`.
fn clamp_capability(cap: Capability) -> Capability {
    cap
}
