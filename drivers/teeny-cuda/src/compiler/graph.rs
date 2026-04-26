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
use teeny_core::{
    compiler::{Compiler, Target},
    device::program::Kernel,
    graph::{Graph, compiler::GraphCompiler},
    model::{ExecutableOp, Lowering, Model},
    utils::dag::Dag,
};

use crate::{
    errors::Result,
    model::{CompiledNode, CudaModel},
};

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

    fn entry_point(&self) -> &str {
        self.0.forward_kernel_entry_point()
    }
}

#[derive(Debug, Clone)]
pub struct CudaGraphCompiler {
    compiler: LlvmCompiler,
}

impl CudaGraphCompiler {
    pub fn new(compiler: LlvmCompiler) -> Self {
        Self { compiler }
    }

    /// Compile a graph to a `CudaModel`, returning the concrete type directly.
    /// Use this when you need access to the compiled DAG (e.g. in tests).
    pub fn compile_model<'a, L: Lowering<'a>, T: Target>(
        &self,
        graph: &Graph,
        lowering: &L,
        target: &T,
        force: bool,
    ) -> Result<CudaModel<'a>> {
        self.compile_inner(graph, lowering, target, force)
    }

    fn compile_inner<'a, L: Lowering<'a>, T: Target>(
        &self,
        graph: &Graph,
        lowering: &L,
        target: &T,
        force: bool,
    ) -> Result<CudaModel<'a>> {
        let op_dag: Dag<Box<dyn ExecutableOp>> = lowering.lower(graph)?;

        let compiler = match target.target_cpu() {
            Some(cpu) => self.compiler.clone().with_target_cpu(cpu),
            None => self.compiler.clone(),
        };

        let mut compiled_dag: Dag<CompiledNode> = Dag::new();

        for i in 0..op_dag.len() {
            let op = op_dag.node(i).value.as_ref();
            let ptx_path = if op.is_input() {
                String::new()
            } else if op.forward_kernel_source().is_empty() {
                return Err(anyhow::anyhow!(
                    "no forward kernel source for op {}",
                    op.name()
                ));
            } else {
                let adapter = ForwardKernelAdapter(op);
                compiler.compile(&adapter, target, force)?
            };

            compiled_dag.add_node(CompiledNode {
                ptx_path,
                entry_point: op.forward_kernel_entry_point().to_string(),
                output_shape: op.output_shape().clone(),
                output_dtype: op.output_dtype(),
            });
        }

        for i in 0..op_dag.len() {
            for &child in &op_dag.node(i).children {
                compiled_dag.add_edge(i, child);
            }
        }

        CudaModel::new(compiled_dag)
    }
}

impl GraphCompiler for CudaGraphCompiler {
    fn compile<'a, L: Lowering<'a>, T: Target>(
        &self,
        graph: &Graph,
        lowering: &L,
        target: &T,
        force: bool,
    ) -> Result<impl Model<'a>> {
        self.compile_inner(graph, lowering, target, force)
    }
}
