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
use teeny_core::{
    compiler::{Compiler, CompilerOptions, Target as _},
    device::program::Kernel,
    graph::Graph,
    model::{ExecutableOp, Lowering, LoweringMode},
};

use crate::compiler::TARGET_TRIPLE;
use crate::compiler::target::Target;
use crate::errors::Result;

/// Adapts a `&dyn ExecutableOp` to the `Kernel` trait so that `LlvmCompiler`
/// can compile the forward kernel without knowing the concrete argument types.
struct ForwardKernelAdapter<'a>(&'a dyn ExecutableOp);

impl<'a> Kernel for ForwardKernelAdapter<'a> {
    /// Argument types are not needed at compile time; `()` satisfies the bound.
    type Args<'b> = ();

    fn name(&self) -> &str {
        self.0.name()
    }

    fn source(&self) -> &str {
        self.0.forward_kernel_source()
    }

    fn kernel_source(&self) -> &str {
        self.0.forward_kernel_source()
    }

    fn entry_point_source(&self) -> &str {
        ""
    }
}

/// A kernel compiled by [`RiscvGraphCompiler::compile_graph`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CompiledKernel {
    /// The op's kernel name.
    pub name: String,
    /// Path of the compiled RISC-V ELF shared library in the cache directory.
    pub library_path: String,
}

/// Compiles every kernel a `teeny-core` [`Graph`] references for a RISC-V [`Target`].
///
/// Unlike `teeny_cuda::compiler::graph::CudaGraphCompiler`, this doesn't assemble a runnable
/// model -- there is no RISC-V model runtime yet -- it only populates the kernel cache. Only
/// forward kernels are compiled.
#[derive(Debug, Clone)]
pub struct RiscvGraphCompiler {
    compiler: LlvmCompiler,
}

impl RiscvGraphCompiler {
    /// Wraps an [`LlvmCompiler`] as a graph compiler; the RISC-V target triple is set here.
    pub fn new(compiler: LlvmCompiler) -> Self {
        Self {
            compiler: compiler.with_target_triple(TARGET_TRIPLE),
        }
    }

    /// Lowers `graph` and compiles each non-input op's forward kernel for `target`, returning
    /// the compiled kernels in DAG order.
    pub fn compile_graph<'a, L: Lowering<'a>>(
        &self,
        graph: &Graph,
        lowering: &L,
        target: &Target,
        mode: LoweringMode,
        force: bool,
    ) -> Result<Vec<CompiledKernel>> {
        let (op_dag, _graph_to_dag) = lowering.lower_with_mapping(graph, mode)?;

        let compiler = self.compiler.clone().with_target_cpu(
            target
                .target_cpu()
                .expect("Target::target_cpu is always Some"),
        );
        // Emit both outputs, matching `driver::compile_kernel`: the cache only counts as warm
        // for an output that was requested, so a runtime compile asking for the assembly too
        // would otherwise recompile every kernel compiled here.
        let options = CompilerOptions {
            force,
            emit_asm: true,
            emit_bin: true,
        };

        let mut compiled = Vec::new();
        for i in 0..op_dag.len() {
            let op = op_dag.node(i).value.as_ref();
            if op.is_input() {
                continue;
            }
            if op.forward_kernel_source().is_empty() {
                return Err(anyhow::anyhow!(
                    "no forward kernel source for op {}",
                    op.name()
                ));
            }
            let library_path = compiler.compile(&ForwardKernelAdapter(op), target, &options)?;
            compiled.push(CompiledKernel {
                name: op.name().to_string(),
                library_path,
            });
        }
        Ok(compiled)
    }
}
