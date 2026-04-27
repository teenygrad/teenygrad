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

use alloc::{boxed::Box, sync::Arc, vec::Vec};

use crate::{device::program::ArgVisitor, errors::Result, graph::{DtypeRepr, Graph, Shape}, utils::dag::Dag};

pub type NodeId = usize;
/// Raw device pointer alias used by runtime arg-packing.
pub type RawPtr = *mut core::ffi::c_void;

pub trait RuntimeContext<'a> {}

/// Encapsulates the runtime-dispatch behaviour for a compiled op node:
/// how many activation inputs it takes, what parameter buffers it needs,
/// how to pack kernel arguments, and how to compute the launch grid.
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

    /// Pack all kernel arguments into `visitor` in the correct order.
    /// - `inputs`       — (ptr, concrete_shape) per activation input
    /// - `params`       — raw pointers to pre-allocated param buffers (weights, biases)
    /// - `output`       — raw pointer to the output buffer for this node
    /// - `output_shape` — concrete output shape (batch dim resolved)
    fn pack_args(
        &self,
        inputs: &[(RawPtr, &[usize])],
        params: &[RawPtr],
        output: RawPtr,
        output_shape: &[usize],
        visitor: &mut dyn ArgVisitor,
    );

    /// Threads-per-CTA for this kernel (x, y, z).
    fn block(&self) -> [u32; 3];

    /// Number of CTAs to launch (x, y, z), given the concrete output shape.
    fn grid(&self, output_shape: &[usize]) -> [u32; 3];

    /// Returns true if this op has a backward (gradient) kernel.
    #[cfg(feature = "training")]
    fn has_backward(&self) -> bool { false }

    /// Pack backward kernel arguments.
    ///
    /// - `inputs`       — (ptr, shape) per forward activation input (from cache)
    /// - `params`       — raw ptrs to forward param buffers (weights, biases)
    /// - `output`       — forward output buffer (activation cache)
    /// - `output_shape` — concrete forward output shape
    /// - `grad_output`  — incoming gradient dL/dy from the consumer node
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
        grad_inputs: &[RawPtr],
        grad_params: &[RawPtr],
        visitor: &mut dyn ArgVisitor,
    ) {
        let _ = (inputs, params, output, output_shape, grad_output, grad_inputs, grad_params, visitor);
    }

    /// Threads-per-CTA for the backward kernel.
    #[cfg(feature = "training")]
    fn backward_block(&self) -> [u32; 3] { [128, 1, 1] }

    /// Number of CTAs for the backward kernel.
    ///
    /// `input_shapes[i]` is the concrete shape of the i-th activation input.
    #[cfg(feature = "training")]
    fn backward_grid(&self, input_shapes: &[&[usize]], output_shape: &[usize]) -> [u32; 3] {
        let _ = (input_shapes, output_shape);
        [0, 0, 0]
    }
}

/// An op that has been lowered to a compilable kernel representation.
///
/// Holds enough information for a caller (who has access to `teeny-compiler`)
/// to compile the kernel for a given target. Dispatch/execution is deferred.
pub trait ExecutableOp {
    fn name(&self) -> &str;
    /// Returns `true` for `Input` placeholder nodes, which carry no kernel.
    fn is_input(&self) -> bool {
        false
    }
    fn forward_kernel_source(&self) -> &str;
    fn forward_kernel_entry_point(&self) -> &str;
    fn output_shape(&self) -> &Shape;
    fn output_dtype(&self) -> DtypeRepr;
    /// Returns the runtime dispatch object for this op, or `None` for Input nodes.
    fn runtime_op(&self) -> Option<Arc<dyn RuntimeOp>> {
        None
    }

    /// Returns the backward kernel source, or `""` if no backward is available.
    #[cfg(feature = "training")]
    fn backward_kernel_source(&self) -> &str { "" }

    /// Returns the backward kernel entry point name.
    #[cfg(feature = "training")]
    fn backward_kernel_entry_point(&self) -> &str { "entry_point" }
}

pub trait Lowering<'a> {
    fn lower(&self, graph: &Graph) -> Result<Dag<Box<dyn ExecutableOp>>>;
}

pub trait Model<'a> {
    type Input;
    type Output;

    fn forward(&self, input: Self::Input) -> Result<Self::Output>;
}
