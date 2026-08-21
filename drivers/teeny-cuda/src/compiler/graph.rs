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

use std::collections::HashMap;

use teeny_compiler::compiler::backend::llvm::compiler::LlvmCompiler;
use teeny_core::{
    compiler::{Compiler, Target},
    device::program::Kernel,
    graph::{Graph, compiler::GraphCompiler},
    model::{ExecutableOp, Lowering, LoweringMode, Model},
    utils::dag::Dag,
};

use crate::{
    errors::Result,
    model::{CompiledNode, CudaModel},
};

/// Adapts a `&dyn ExecutableOp` to the `Kernel` trait so that `LlvmCompiler`
/// can compile the forward kernel without knowing the concrete argument types.
struct ForwardKernelAdapter<'a>(&'a dyn ExecutableOp);

/// Adapts a `&dyn ExecutableOp` backward kernel source to the `Kernel` trait.
#[cfg(feature = "training")]
struct BackwardKernelAdapter<'a>(&'a dyn ExecutableOp);

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

#[cfg(feature = "training")]
impl<'a> Kernel for BackwardKernelAdapter<'a> {
    type Args<'b> = ();

    fn name(&self) -> &str {
        self.0.name()
    }

    fn source(&self) -> &str {
        self.0.backward_kernel_source()
    }

    fn kernel_source(&self) -> &str {
        self.0.backward_kernel_source()
    }

    fn entry_point_source(&self) -> &str {
        ""
    }
}

/// Compiles a `teeny-core` [`Graph`] into a runnable CUDA model.
#[derive(Debug, Clone)]
pub struct CudaGraphCompiler {
    compiler: LlvmCompiler,
}

impl CudaGraphCompiler {
    /// Wraps an [`LlvmCompiler`] as a graph compiler.
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
        mode: LoweringMode,
        force: bool,
    ) -> Result<CudaModel<'a>> {
        let (op_dag, graph_to_dag, lowered_graph) = lowering.lower_with_mapping(graph, mode)?;
        self.compile_lowered(op_dag, graph_to_dag, &lowered_graph, lowering, target, force)
    }

    /// Compile an already-lowered `op_dag`/`graph_to_dag` pair (as produced by
    /// [`Lowering::lower_with_mapping`]) to a `CudaModel`.
    ///
    /// Use this instead of [`Self::compile_model`] when a caller-side
    /// optimization pass (e.g. `teeny_kernels::graph::Anduin`, run via its
    /// own `GraphOptimizer` trait — not a dependency of this crate) has
    /// rewritten the lowered DAG first: optimization is a separate step run
    /// *after* lowering, not something `lowering` or this crate knows
    /// about, so its output can't be fed back through a `Graph`-typed entry
    /// point. `lowered_graph` and `lowering` are still needed here — not to
    /// re-lower, but to resolve DAG node names
    /// ([`Lowering::extra_dag_names`]) for placing pretrained weights.
    pub fn compile_lowered<'a, L: Lowering<'a>, T: Target>(
        &self,
        op_dag: Dag<Box<dyn ExecutableOp>>,
        graph_to_dag: Vec<usize>,
        lowered_graph: &Graph,
        lowering: &L,
        target: &T,
        force: bool,
    ) -> Result<CudaModel<'a>> {
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

            #[cfg(feature = "training")]
            let backward_ptx_path = if op.is_input() || op.backward_kernel_source().is_empty() {
                None
            } else {
                let adapter = BackwardKernelAdapter(op);
                Some(compiler.compile(&adapter, target, force)?)
            };

            compiled_dag.add_node(CompiledNode {
                ptx_path,
                entry_point: op.forward_kernel_entry_point().to_string(),
                output_shape: op.output_shape().clone(),
                output_dtype: op.output_dtype(),
                runtime_op: op.runtime_op(),
                #[cfg(feature = "training")]
                backward_ptx_path,
                #[cfg(feature = "training")]
                backward_entry_point: op.backward_kernel_entry_point().to_string(),
            });
        }

        // Rebuild edges using parent lists (not children) to preserve the insertion
        // order that the lowering recorded. If we iterated children (add_edge(i, child)
        // for each i in 0..N), parents with smaller DAG indices would be appended first,
        // destroying the logical input ordering required by ops like ChannelCat.
        for i in 0..op_dag.len() {
            for &parent in &op_dag.node(i).parents {
                compiled_dag.add_edge(parent, i);
            }
        }

        // Propagate graph-level node names (from name_scope annotations) into the
        // compiled DAG using the graph_node → dag_node index mapping. `graph_to_dag`
        // is indexed against `lowered_graph` (the graph the lowering actually built
        // the DAG from — post-optimization when an optimizer runs), not the caller's
        // original `graph`, which may have a different node count/ordering — see
        // `Lowering::lower_with_mapping`'s doc.
        let mut dag_names: HashMap<usize, String> = lowered_graph
            .names
            .iter()
            .filter_map(|(&graph_idx, name)| {
                let dag_idx = *graph_to_dag.get(graph_idx)?;
                Some((dag_idx, name.clone()))
            })
            .collect();

        // Lowerings that split one graph node into multiple DAG nodes (e.g.
        // Conv2d-with-bias → Conv2d + NchwBiasAdd) expose the extra mappings
        // here so that every DAG node with parameters can resolve its name.
        for (dag_idx, name) in lowering.extra_dag_names(lowered_graph, &graph_to_dag) {
            dag_names.entry(dag_idx).or_insert(name);
        }

        CudaModel::with_names(compiled_dag, dag_names)
    }
}

impl GraphCompiler for CudaGraphCompiler {
    fn compile<'a, L: Lowering<'a>, T: Target>(
        &self,
        graph: &Graph,
        lowering: &L,
        target: &T,
        mode: LoweringMode,
        force: bool,
    ) -> Result<impl Model<'a>> {
        self.compile_model(graph, lowering, target, mode, force)
    }
}
