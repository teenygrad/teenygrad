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

use teeny_compiler::compiler::backend::llvm::compiler::LlvmCompiler;
use teeny_core::graph::Graph;
use teeny_core::model::{Lowering, LoweringMode};

use crate::compiler::graph::CudaGraphCompiler;
use crate::compiler::options::Options;
use crate::compiler::target::Target;
use crate::errors::Result;
use crate::model::CudaModel;

/// Ahead-of-time compile an already-traced `graph` to PTX files on disk,
/// driven by an already-parsed [`Options`] (see [`Options::parse`]).
///
/// This is the same `LlvmCompiler` + `CudaGraphCompiler` sequence used for
/// JIT compilation at runtime (e.g. in `models/teeny-vision/examples/mnist.rs`),
/// just parameterized by CLI-driven `options`/`cache_dir` instead of hardcoded
/// values. No live CUDA device/context is required — only `.load()` on the
/// returned `CudaModel` needs one.
///
/// `cache_dir` is where compiled PTX is written/read. Passing the same
/// directory a later run resolves via `TEENYC_CACHE_DIR` pre-warms that
/// runtime JIT cache. The `teenyc` binary is resolved via
/// [`teeny_compiler::compiler::find_teenyc`], matching the existing JIT compile path.
pub fn compile_graph<'a, L: Lowering<'a>>(
    graph: &Graph,
    lowering: &L,
    mode: LoweringMode,
    options: &Options,
    cache_dir: &str,
    force: bool,
) -> Result<CudaModel<'a>> {
    let teenyc_path = teeny_compiler::compiler::find_teenyc()?;

    let mut compiler = LlvmCompiler::new(teenyc_path, cache_dir.to_string())?;
    if let Some(ptx_version) = options.ptx_version {
        compiler = compiler.with_ptx_version(ptx_version);
    }
    if let Some(log_level) = options.log_level {
        compiler = compiler.with_log_level(log_level);
    }
    let graph_compiler = CudaGraphCompiler::new(compiler);
    let target = Target::new(options.gpu_name);

    graph_compiler.compile_model(graph, lowering, &target, mode, force)
}
