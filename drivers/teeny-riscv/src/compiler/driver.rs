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
use teeny_core::compiler::{Compiler, Target as _};
use teeny_core::device::program::Kernel;

use crate::compiler::TARGET_TRIPLE;
use crate::compiler::target::Target;
use crate::errors::Result;

/// Compiles `kernel` for `target` via the LLVM backend's RISC-V path (`riscv64-generic`), using
/// the `teenyc` binary resolved by [`teeny_compiler::compiler::find_teenyc`] and the default
/// cache directory. Same signature as `teeny_cuda::compiler::compile_kernel` -- both back the
/// same `LlvmCompiler` -- so `teeny-runtime` can re-export whichever backend's `compile_kernel`
/// is active without callers caring which one it is.
///
/// - `force`: recompile even if a cached artifact exists.
/// - `debug`: when `true`, enable the same `teenyc` pipeline-stage logging as
///   `teeny_cuda::compiler::compile_kernel`'s `debug` parameter.
///
/// The underlying backend (`RiscvBackend`) is still a stub -- every kernel compiles to the same
/// placeholder no-argument `void @<name>()` function regardless of `kernel`'s actual body (see
/// the crate README) -- so this only proves the compile pipeline (real LLVM RISC-V codegen,
/// linked via `ld.lld`) produces a genuine RISC-V shared object, not that it runs `kernel`'s
/// logic.
pub fn compile_kernel(
    kernel: &impl Kernel,
    target: &Target,
    force: bool,
    debug: bool,
) -> Result<String> {
    let teenyc_path = teeny_compiler::compiler::find_teenyc()?;
    let cache_dir = teeny_compiler::compiler::default_cache_dir();
    let mut compiler = LlvmCompiler::new(teenyc_path, cache_dir)?
        .with_target_triple(TARGET_TRIPLE)
        .with_target_cpu(
            target
                .target_cpu()
                .expect("Target::target_cpu is always Some"),
        );
    if debug {
        compiler = compiler.with_log_level(LogLevel::Debug);
    }

    with_pipeline_logging(debug, || compiler.compile(kernel, target, force))
}

/// When `debug` is set, install a thread-local stderr tracing subscriber for the duration of `f`
/// so `teenyc` pipeline-stage IR is visible under `--nocapture`. Mirrors
/// `teeny_cuda::compiler::driver`'s helper of the same name.
fn with_pipeline_logging<T>(debug: bool, f: impl FnOnce() -> Result<T>) -> Result<T> {
    if !debug {
        return f();
    }
    let subscriber = tracing_subscriber::fmt()
        .with_max_level(tracing::Level::TRACE)
        .with_ansi(false)
        .with_writer(std::io::stderr)
        .finish();
    tracing::subscriber::with_default(subscriber, f)
}
