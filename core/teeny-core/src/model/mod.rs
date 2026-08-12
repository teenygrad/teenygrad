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

use alloc::{boxed::Box, string::String, sync::Arc, vec, vec::Vec};

use crate::{
    device::program::ArgVisitor,
    errors::Result,
    graph::{DtypeRepr, Graph, Shape},
    utils::dag::Dag,
};

/// A node index within a compiled model's DAG.
pub type NodeId = usize;
/// Raw device pointer alias used by runtime arg-packing.
pub type RawPtr = *mut core::ffi::c_void;

/// A concrete, dtype-resolved kernel produced by a `#[kernel(dtypes = [...])]`
/// dispatcher (or a `kernel_group!`).
///
/// It is deliberately crate-agnostic: it carries only the pieces needed to
/// assemble a compilable unit (`teeny-kernels`' `KernelExecutable`), while
/// living in `teeny-core` so the `#[kernel]` macro can reference it from any
/// consuming crate without a dependency on `teeny-kernels`.
pub struct KernelInstance {
    /// Forward kernel name (used to derive the entry-point symbol).
    pub name: String,
    /// Combined forward kernel source (`kernel_source + entry_point`).
    pub source: String,
    /// Forward kernel body only (no C-ABI entry wrapper) — used when composing
    /// fused entries that synthesize their own wrapper.
    pub kernel_body: String,
    /// Runtime dispatch object for arg-packing and launch config.
    pub runtime_op: Arc<dyn RuntimeOp>,
    /// `Some(BLOCK_SIZE)` when this kernel passes the pointwise-fuse probe
    /// (unary In/Out + `n_elements` CTA map); otherwise `None`.
    pub pointwise_fuse_block_size: Option<i32>,
    /// Backward kernel, present only when the kernel declares one.
    pub backward: Option<KernelInstanceBackward>,
}

/// The backward half of a [`KernelInstance`], when a paired backward kernel is
/// declared via `#[kernel(..., backward = ...)]`.
pub struct KernelInstanceBackward {
    /// Backward kernel name (used to derive the entry-point symbol).
    pub name: String,
    /// Combined backward kernel source.
    pub source: String,
}

/// A runtime execution context (device/stream handles, etc), threaded through arg-packing.
pub trait RuntimeContext<'a> {}

/// Encapsulates the runtime-dispatch behaviour for a compiled op node:
/// how many activation inputs it takes, what parameter buffers it needs,
/// how to pack kernel arguments, and how to compute the launch grid.
///
/// CUDA thread counts are not part of this trait: the executor reads them from
/// compiled PTX metadata (`.reqntid` / `num_warps`) at launch time.
///
/// Implementations live in `teeny-kernels` alongside the kernel structs so that
/// each kernel owns its arg layout. The trait is defined here in `teeny-core`
/// so that both `teeny-kernels` (impl) and `teeny-cuda` (consumer) can share it
/// without a circular dependency.
pub trait RuntimeOp: Send + Sync {
    /// Number of activation tensors taken from predecessor DAG nodes.
    fn n_activation_inputs(&self) -> usize;

    /// Shapes of additional parameter buffers (weights, biases) needed by this
    /// op. Called at `LoadedModel::load()` time to pre-allocate device buffers.
    /// `input_shapes` / `output_shape` are concrete (batch dim resolved).
    fn param_shapes(&self, input_shapes: &[&[usize]], output_shape: &[usize]) -> Vec<Vec<usize>>;

    /// Names of parameter slots returned by [`param_shapes`], in the same order.
    /// Used as the suffix in the dotted key `{node_name}.{slot_name}`.
    /// Return an empty slice for ops that have no named parameters.
    fn param_names(&self) -> &'static [&'static str] {
        &[]
    }

    /// Returns the required row stride (in elements) for the output buffer of
    /// this op's forward kernel.  The default is the natural row-major stride
    /// (`output_shape[-1]`).  Kernels using TMA must round up to satisfy the
    /// 16-byte alignment constraint (e.g. 4 elements for f32).
    fn forward_output_row_stride(&self, output_shape: &[usize]) -> usize {
        output_shape.last().copied().unwrap_or(1)
    }

    /// Returns raw (little-endian) bytes to pre-populate parameter slot `param_idx`
    /// immediately after device buffer allocation.  Return `None` to leave the
    /// slot zero-initialised (the default for trained parameters).
    /// Byte count must equal `param_shapes()[param_idx].iter().product() * dtype_bytes`.
    fn param_init_data(&self, _param_idx: usize) -> Option<Vec<u8>> {
        None
    }

    /// Override to compute the true concrete output shape from concrete input shapes.
    ///
    /// Called during both `load()` (for param allocation) and `forward()` (for buffer
    /// allocation). The default just returns the `resolved` shape (from `resolve_shape`).
    ///
    /// Override this when the output's first dimension is a multiple of the batch size
    /// (e.g. `B * H` for attention pack/unpack ops) so that `resolve_shape`'s simple
    /// `None → batch_size` substitution would under-allocate the buffer.
    fn compute_concrete_output_shape(
        &self,
        _input_shapes: &[&[usize]],
        resolved: &[usize],
    ) -> Vec<usize> {
        resolved.to_vec()
    }

    /// Pack all kernel arguments into `visitor` in the correct order.
    /// - `inputs`            — (ptr, concrete_shape) per activation input
    /// - `params`            — raw pointers to pre-allocated param buffers
    /// - `output`            — raw pointer to the output buffer for this node
    /// - `output_shape`      — concrete output shape (batch dim resolved)
    /// - `output_row_stride` — actual memory row stride (elements) of the
    ///   output buffer (may be padded for TMA alignment)
    fn pack_args(
        &self,
        inputs: &[(RawPtr, &[usize])],
        params: &[RawPtr],
        output: RawPtr,
        output_shape: &[usize],
        output_row_stride: i32,
        visitor: &mut dyn ArgVisitor,
    );

    /// Number of CTAs to launch (x, y, z), given the concrete output shape.
    fn grid(&self, output_shape: &[usize]) -> [u32; 3];

    /// Number of sequential kernel launches this op requires.
    ///
    /// Ops like channel-cat scatter N input chunks into one output buffer and
    /// need one kernel call per chunk. The executor loops `n_launches()` times,
    /// calling `pack_args_for_launch` and `grid_for_launch` on each iteration.
    /// The default is 1, which delegates to `pack_args` / `grid`.
    fn n_launches(&self) -> usize {
        1
    }

    /// Pack kernel arguments for launch `i` (0-indexed).
    ///
    /// Only called by the executor when `n_launches() > 1`. The default
    /// delegates to `pack_args`, ignoring `launch_idx`.
    // Kernel-launch argument list; each parameter corresponds to a distinct piece of the
    // launch ABI (input/param/output buffers, shape, stride, launch index) and bundling them
    // into a struct wouldn't be clearer at call sites.
    #[allow(clippy::too_many_arguments)]
    fn pack_args_for_launch(
        &self,
        launch_idx: usize,
        inputs: &[(RawPtr, &[usize])],
        params: &[RawPtr],
        output: RawPtr,
        output_shape: &[usize],
        output_row_stride: i32,
        visitor: &mut dyn ArgVisitor,
    ) {
        let _ = launch_idx;
        self.pack_args(
            inputs,
            params,
            output,
            output_shape,
            output_row_stride,
            visitor,
        );
    }

    /// Grid for launch `i`. Receives concrete input shapes so that per-chunk
    /// grids can be computed without storing them in the op.
    ///
    /// Only called when `n_launches() > 1`. The default delegates to `grid`.
    fn grid_for_launch(
        &self,
        launch_idx: usize,
        input_shapes: &[&[usize]],
        output_shape: &[usize],
    ) -> [u32; 3] {
        let _ = (launch_idx, input_shapes);
        self.grid(output_shape)
    }

    /// Returns true if this op has a backward (gradient) kernel.
    #[cfg(feature = "training")]
    fn has_backward(&self) -> bool {
        false
    }

    /// Returns the required row stride (in elements) for the grad_output buffer
    /// passed to `pack_backward_args`. The default is the natural row-major
    /// stride (`output_shape[-1]`). Kernels using TMA must round up to satisfy
    /// the 16-byte alignment constraint (e.g. 4 elements for f32).
    #[cfg(feature = "training")]
    fn backward_grad_output_row_stride(&self, output_shape: &[usize]) -> usize {
        output_shape.last().copied().unwrap_or(1)
    }

    /// Pack backward kernel arguments.
    ///
    /// - `inputs`       — (ptr, shape) per forward activation input (from cache)
    /// - `params`       — raw ptrs to forward param buffers (weights, biases)
    /// - `output`       — forward output buffer (activation cache)
    /// - `output_shape` — concrete forward output shape
    /// - `grad_output`  — incoming gradient dL/dy from the consumer node
    /// - `grad_output_row_stride` — actual memory row stride (elements) of the
    ///   grad_output buffer (may be padded for TMA alignment)
    /// - `grad_inputs`  — output gradient buffers: dL/dx per activation parent
    /// - `grad_params`  — output gradient buffers: dL/dw, dL/db, etc.
    #[cfg(feature = "training")]
    #[allow(clippy::too_many_arguments)]
    fn pack_backward_args(
        &self,
        inputs: &[(RawPtr, &[usize])],
        params: &[RawPtr],
        output: RawPtr,
        output_shape: &[usize],
        grad_output: RawPtr,
        grad_output_row_stride: i32,
        grad_inputs: &[RawPtr],
        grad_params: &[RawPtr],
        visitor: &mut dyn ArgVisitor,
    ) {
        let _ = (
            inputs,
            params,
            output,
            output_shape,
            grad_output,
            grad_output_row_stride,
            grad_inputs,
            grad_params,
            visitor,
        );
    }

    /// Number of CTAs for the backward kernel.
    ///
    /// `input_shapes[i]` is the concrete shape of the i-th activation input.
    #[cfg(feature = "training")]
    fn backward_grid(&self, input_shapes: &[&[usize]], output_shape: &[usize]) -> [u32; 3] {
        let _ = (input_shapes, output_shape);
        [0, 0, 0]
    }

    /// Number of sequential kernel launches for the backward pass.
    ///
    /// For ops like channel-cat where the backward must write separate gradient
    /// buffers per input chunk, this should return `n_inputs`. Default is 1.
    #[cfg(feature = "training")]
    fn n_backward_launches(&self) -> usize {
        1
    }

    /// Pack backward kernel arguments for launch `i` (0-indexed).
    ///
    /// Only called when `n_backward_launches() > 1`. The default delegates to
    /// `pack_backward_args`, ignoring `launch_idx`.
    #[cfg(feature = "training")]
    #[allow(clippy::too_many_arguments)]
    fn pack_backward_args_for_launch(
        &self,
        launch_idx: usize,
        inputs: &[(RawPtr, &[usize])],
        params: &[RawPtr],
        output: RawPtr,
        output_shape: &[usize],
        grad_output: RawPtr,
        grad_output_row_stride: i32,
        grad_inputs: &[RawPtr],
        grad_params: &[RawPtr],
        visitor: &mut dyn ArgVisitor,
    ) {
        let _ = launch_idx;
        self.pack_backward_args(
            inputs,
            params,
            output,
            output_shape,
            grad_output,
            grad_output_row_stride,
            grad_inputs,
            grad_params,
            visitor,
        );
    }

    /// Grid for backward launch `i`.
    ///
    /// Only called when `n_backward_launches() > 1`. The default delegates to
    /// `backward_grid`.
    #[cfg(feature = "training")]
    fn backward_grid_for_launch(
        &self,
        launch_idx: usize,
        input_shapes: &[&[usize]],
        output_shape: &[usize],
    ) -> [u32; 3] {
        let _ = launch_idx;
        self.backward_grid(input_shapes, output_shape)
    }
}

/// An op that has been lowered to a compilable kernel representation.
///
/// Holds enough information for a caller (who has access to `teeny-compiler`)
/// to compile the kernel for a given target. Dispatch/execution is deferred.
pub trait ExecutableOp {
    /// This op's name.
    fn name(&self) -> &str;
    /// Returns `true` for `Input` placeholder nodes, which carry no kernel.
    fn is_input(&self) -> bool {
        false
    }
    /// This op's forward kernel source.
    fn forward_kernel_source(&self) -> &str;
    /// This op's forward kernel entry-point symbol name.
    fn forward_kernel_entry_point(&self) -> &str;
    /// This op's output shape.
    fn output_shape(&self) -> &Shape;
    /// This op's output dtype.
    fn output_dtype(&self) -> DtypeRepr;
    /// Returns the runtime dispatch object for this op, or `None` for Input nodes.
    fn runtime_op(&self) -> Option<Arc<dyn RuntimeOp>> {
        None
    }

    /// Downcast support for backend-specific executable types.
    fn as_any(&self) -> &dyn core::any::Any;

    /// Returns the backward kernel source, or `""` if no backward is available.
    #[cfg(feature = "training")]
    fn backward_kernel_source(&self) -> &str {
        ""
    }

    /// Returns the backward kernel entry point name.
    #[cfg(feature = "training")]
    fn backward_kernel_entry_point(&self) -> &str {
        "entry_point"
    }
}

/// Whether a graph is being lowered for inference or training (training lowerings additionally
/// wire up backward kernels).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum LoweringMode {
    /// Inference only; no backward kernels.
    #[default]
    Inference,
    /// Training; backward kernels are wired up.
    Training,
}

/// Lowers a [`Graph`] into a DAG of [`ExecutableOp`]s, ready to compile/execute.
pub trait Lowering<'a> {
    /// Lowers `graph` for `mode`.
    fn lower(&self, graph: &Graph, mode: LoweringMode) -> Result<Dag<Box<dyn ExecutableOp>>>;

    /// Like [`lower`] but also returns a `graph_node_idx → dag_node_idx` mapping
    /// so that graph-level metadata (e.g. names) can be propagated into the compiled DAG.
    ///
    /// The default implementation assumes a 1-to-1 identity mapping between graph
    /// topological order and DAG node indices — valid for graphs that are already in
    /// topological order (which is always true for models built by sequential recording)
    /// and lowerings that do not reorder or split nodes.  Override this method in
    /// lowerings that reorder or split nodes to return the correct mapping.
    #[allow(clippy::type_complexity)]
    fn lower_with_mapping(
        &self,
        graph: &Graph,
        mode: LoweringMode,
    ) -> Result<(Dag<Box<dyn ExecutableOp>>, Vec<usize>)> {
        let dag = self.lower(graph, mode)?;
        let topo = graph.topological_sort();
        // topo[dag_idx] = graph_node_idx  →  graph_to_dag[graph_node_idx] = dag_idx
        let mut graph_to_dag = vec![0usize; graph.nodes.len()];
        for (dag_idx, graph_idx) in topo.into_iter().enumerate() {
            graph_to_dag[graph_idx] = dag_idx;
        }
        Ok((dag, graph_to_dag))
    }

    /// Returns extra (dag_idx, name) pairs beyond those derivable from the
    /// 1-to-1 graph→dag mapping.  Used for lowerings that split one graph node
    /// into multiple DAG nodes (e.g. Conv2d-with-bias → Conv2d + NchwBiasAdd);
    /// the "extra" DAG nodes would otherwise have no name and their weight
    /// parameters would not be loaded.
    fn extra_dag_names(&self, _graph: &Graph, _graph_to_dag: &[usize]) -> Vec<(usize, String)> {
        Vec::new()
    }

    /// Returns the next lowering in a middleware chain, or `None` if this is
    /// the final lowering.  A custom lowering can call `self.base_lowering()`
    /// to delegate ops it does not handle.
    fn base_lowering(&self) -> Option<&dyn Lowering<'a>> {
        None
    }
}

/// A compiled, runnable model.
pub trait Model<'a> {
    /// This model's input type.
    type Input;
    /// This model's output type.
    type Output;

    /// Runs inference, producing `Output` from `input`.
    fn forward(&self, input: Self::Input) -> Result<Self::Output>;
}
