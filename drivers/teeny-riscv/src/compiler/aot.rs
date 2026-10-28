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

use teeny_compiler::compiler::backend::llvm::compiler::LlvmCompiler;
use teeny_core::graph::Graph;
use teeny_core::model::{Lowering, LoweringMode};

use crate::compiler::graph::{CompiledKernel, RiscvGraphCompiler};
use crate::compiler::options::Options;
use crate::compiler::target::Target;
use crate::errors::Result;

/// Ahead-of-time compile an already-traced `graph` to RISC-V kernel shared libraries on disk,
/// driven by an already-parsed [`Options`] (see [`Options::parse`]).
///
/// The RISC-V counterpart of `teeny_cuda::compiler::aot::compile_graph`. `cache_dir` is where
/// compiled kernels are written/read; kernels are keyed by kernel id and `capability`, so
/// passing the directory a later run resolves via `TEENYC_CACHE_DIR` pre-warms its cache. The
/// `teenyc` binary is resolved via [`teeny_compiler::compiler::find_teenyc`].
pub fn compile_graph<'a, L: Lowering<'a>>(
    graph: &Graph,
    lowering: &L,
    mode: LoweringMode,
    options: &Options,
    cache_dir: &str,
    force: bool,
) -> Result<Vec<CompiledKernel>> {
    let teenyc_path = teeny_compiler::compiler::find_teenyc()?;

    let mut compiler = LlvmCompiler::new(teenyc_path, cache_dir.to_string())?;
    if let Some(log_level) = options.log_level {
        compiler = compiler.with_log_level(log_level);
    }
    let graph_compiler = RiscvGraphCompiler::new(compiler);
    let target = Target::new(options.capability);

    graph_compiler.compile_graph(graph, lowering, &target, mode, force)
}
