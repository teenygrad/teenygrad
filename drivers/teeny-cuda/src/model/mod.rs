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

use std::{marker::PhantomData, sync::Arc};

use anyhow::anyhow;
use teeny_core::{
    graph::{DtypeRepr, Shape},
    model::{Model, RuntimeOp},
    utils::dag::Dag,
};

use crate::{
    cuda,
    device::{
        CudaArgPacker, CudaDevice, CudaLaunchConfig,
        mem::{self, DevicePtr},
        program::{CudaProgram, ErasedKernel},
    },
    errors::Result,
};

// ---------------------------------------------------------------------------
// TensorRef — a device buffer pointer with a concrete runtime shape
// ---------------------------------------------------------------------------

/// A reference to a device-side tensor: raw device pointer + concrete shape.
///
/// `shape` is always fully concrete (no `None` dims).
#[derive(Clone, Debug)]
pub struct TensorRef {
    pub ptr: DevicePtr,
    pub shape: Vec<usize>,
}

impl TensorRef {
    pub fn new(ptr: DevicePtr, shape: Vec<usize>) -> Self {
        Self { ptr, shape }
    }

    pub fn n_elements(&self) -> usize {
        self.shape.iter().product()
    }
}

fn dtype_bytes(dtype: DtypeRepr) -> usize {
    match dtype {
        DtypeRepr::Bool | DtypeRepr::I8 | DtypeRepr::U8 => 1,
        DtypeRepr::I16 | DtypeRepr::U16 | DtypeRepr::F16 | DtypeRepr::BF16 => 2,
        DtypeRepr::I32 | DtypeRepr::U32 | DtypeRepr::F32 => 4,
        DtypeRepr::I64 | DtypeRepr::U64 | DtypeRepr::F64 => 8,
    }
}

fn resolve_shape(shape: &Shape, batch_size: usize) -> Vec<usize> {
    shape.iter().map(|d| d.unwrap_or(batch_size)).collect()
}

// ---------------------------------------------------------------------------
// CompiledNode — one PTX-compiled graph node
// ---------------------------------------------------------------------------

pub struct CompiledNode {
    /// Path to the compiled `.o` PTX file. Empty for `Input` placeholder nodes.
    pub ptx_path: String,
    pub entry_point: String,
    pub output_shape: Shape,
    pub output_dtype: DtypeRepr,
    /// Runtime dispatch: arg-packing + grid computation. `None` for Input nodes.
    pub runtime_op: Option<Arc<dyn RuntimeOp>>,
}

// ---------------------------------------------------------------------------
// CudaModel — compiled but not yet loaded into GPU memory
// ---------------------------------------------------------------------------

pub struct CudaModel<'a> {
    pub dag: Dag<CompiledNode>,
    _marker: PhantomData<&'a ()>,
}

impl<'a> Model<'a> for CudaModel<'a> {
    type Input = TensorRef;
    type Output = TensorRef;

    fn forward(&self, _input: Self::Input) -> teeny_core::errors::Result<Self::Output> {
        Err(anyhow!("call CudaModel::load() first, then LoadedModel::forward()").into())
    }
}

impl<'a> CudaModel<'a> {
    pub fn new(dag: Dag<CompiledNode>) -> Result<Self> {
        Ok(Self { dag, _marker: PhantomData })
    }

    /// Load all compiled PTX kernels into GPU memory and pre-allocate
    /// zero-initialised parameter buffers, producing a `LoadedModel` ready
    /// for inference.
    ///
    /// `batch_size` resolves dynamic (`None`) shape dimensions when computing
    /// parameter buffer sizes.
    pub fn load(self, _device: &CudaDevice<'_>, batch_size: usize) -> Result<LoadedModel> {
        let n = self.dag.len();
        let topo = self.dag.topological_sort();

        // Snapshot parent lists before consuming the dag.
        let parents: Vec<Vec<usize>> = (0..n)
            .map(|i| self.dag.node(i).parents.clone())
            .collect();

        // Consume the dag into (parents, CompiledNode) pairs.
        let compiled: Vec<CompiledNode> = self.dag.into_iter().map(|node| node.value).collect();

        let mut loaded_nodes: Vec<Option<LoadedNode>> = (0..n).map(|_| None).collect();

        for &idx in &topo {
            let cn = &compiled[idx];
            let Some(rop) = cn.runtime_op.as_ref() else {
                continue; // Input placeholder — no kernel, no params
            };

            // Gather concrete input shapes from parent nodes' CompiledNode shapes.
            let parent_shapes: Vec<Vec<usize>> = parents[idx].iter()
                .map(|&p| resolve_shape(&compiled[p].output_shape, batch_size))
                .collect();
            let parent_shape_refs: Vec<&[usize]> =
                parent_shapes.iter().map(|s| s.as_slice()).collect();
            let output_shape = resolve_shape(&cn.output_shape, batch_size);

            // Allocate and zero-init device buffers for each parameter slot.
            let p_shapes = rop.param_shapes(&parent_shape_refs, &output_shape);
            let mut param_bufs: Vec<DevicePtr> = Vec::with_capacity(p_shapes.len());
            for ps in &p_shapes {
                let n_elems: usize = ps.iter().product();
                let byte_size = n_elems * dtype_bytes(cn.output_dtype);
                let ptr = mem::alloc(byte_size)?;
                unsafe { cuda::cuMemsetD8_v2(ptr, 0, byte_size) };
                param_bufs.push(ptr);
            }

            // JIT-compile the PTX via the CUDA driver.
            let ptx = std::fs::read(&cn.ptx_path)
                .map_err(|e| anyhow!("failed to read PTX for node {idx}: {e}"))?;
            let program = CudaProgram::<ErasedKernel>::try_from_ptx(&ptx, &cn.entry_point)?;

            loaded_nodes[idx] = Some(LoadedNode {
                program,
                output_shape: cn.output_shape.clone(),
                output_dtype: cn.output_dtype,
                runtime_op: Arc::clone(rop),
                param_bufs,
                param_shapes: p_shapes,
            });
        }

        Ok(LoadedModel { nodes: loaded_nodes, parents })
    }
}

// ---------------------------------------------------------------------------
// LoadedNode — kernel + param buffers, fully loaded in GPU memory
// ---------------------------------------------------------------------------

struct LoadedNode {
    program: CudaProgram<'static, ErasedKernel>,
    output_shape: Shape,
    output_dtype: DtypeRepr,
    runtime_op: Arc<dyn RuntimeOp>,
    /// Zero-initialised device buffers for model parameters (weights, biases).
    param_bufs: Vec<DevicePtr>,
    /// Concrete shape of each param buffer — stored so callers can initialise weights.
    param_shapes: Vec<Vec<usize>>,
}

impl Drop for LoadedNode {
    fn drop(&mut self) {
        for &ptr in &self.param_bufs {
            if let Err(e) = mem::free(ptr) {
                eprintln!("LoadedNode: failed to free param buffer: {e}");
            }
        }
    }
}

// ---------------------------------------------------------------------------
// LoadedModel — eager-loaded model ready for inference
// ---------------------------------------------------------------------------

pub struct LoadedModel {
    /// Per-DAG-node loaded kernel. `None` for `Input` placeholder nodes.
    nodes: Vec<Option<LoadedNode>>,
    /// Parent node indices per node (same topology as the compiled DAG).
    parents: Vec<Vec<usize>>,
}

impl LoadedModel {
    /// Iterate over every node that has parameter buffers.
    ///
    /// Yields `(node_idx, param_shapes)` where `param_shapes[i]` is the concrete
    /// shape of parameter slot `i` (e.g. `[out_features, in_features]` for a
    /// weight matrix). Use `load_param_f32(node_idx, i, data)` to upload values.
    pub fn param_info(&self) -> impl Iterator<Item = (usize, &[Vec<usize>])> {
        self.nodes.iter().enumerate().filter_map(|(idx, node)| {
            node.as_ref().filter(|n| !n.param_shapes.is_empty())
                .map(|n| (idx, n.param_shapes.as_slice()))
        })
    }

    /// Copy `f32` parameter data into a node's pre-allocated device buffer.
    ///
    /// `node_idx`  — the DAG node index.
    /// `param_idx` — which parameter slot (0 = weight, 1 = bias, …).
    /// `data`      — host `f32` slice; must match the buffer element count exactly.
    pub fn load_param_f32(
        &mut self,
        node_idx: usize,
        param_idx: usize,
        data: &[f32],
    ) -> Result<()> {
        let node = self.nodes[node_idx].as_ref()
            .ok_or_else(|| anyhow!("node {node_idx} is an Input placeholder"))?;
        let ptr = *node.param_bufs.get(param_idx)
            .ok_or_else(|| anyhow!("node {node_idx} has no param at index {param_idx}"))?;
        unsafe { mem::copy_h_to_d(ptr, data.as_ptr(), data.len()) }
    }

    /// Run a single forward pass through the loaded model.
    ///
    /// `device`     — the CUDA device context.
    /// `batch_size` — concrete value for dynamic (`None`) batch dimensions.
    /// `inputs`     — device tensors matched to `Input` nodes in topological order.
    ///
    /// Returns the `TensorRef` of the last DAG node. Intermediate output buffers
    /// are allocated per-call and freed when the returned `TensorRef` is dropped
    /// (caller owns the final buffer; all intermediate ones are freed at the end).
    pub fn forward(
        &self,
        device: &CudaDevice<'_>,
        batch_size: usize,
        inputs: &[TensorRef],
    ) -> Result<TensorRef> {
        let n = self.nodes.len();

        // Rebuild topological order from parent lists.
        let topo = {
            let mut in_deg: Vec<usize> = (0..n).map(|i| self.parents[i].len()).collect();
            let mut dependents: Vec<Vec<usize>> = vec![vec![]; n];
            for i in 0..n {
                for &p in &self.parents[i] {
                    dependents[p].push(i);
                }
            }
            let mut stack: Vec<usize> = (0..n).filter(|&i| in_deg[i] == 0).collect();
            let mut order = Vec::with_capacity(n);
            while let Some(id) = stack.pop() {
                order.push(id);
                for &dep in &dependents[id] {
                    in_deg[dep] -= 1;
                    if in_deg[dep] == 0 { stack.push(dep); }
                }
            }
            order
        };

        // ctx[i] = TensorRef for node i once it has been computed.
        let mut ctx: Vec<Option<TensorRef>> = vec![None; n];
        // Intermediate output buffers that we own and must free.
        let mut intermediate_ptrs: Vec<DevicePtr> = Vec::new();
        let mut input_cursor = 0usize;

        for &idx in &topo {
            if self.nodes[idx].is_none() {
                // Input placeholder — assign from caller-provided inputs.
                let tr = inputs.get(input_cursor)
                    .ok_or_else(|| anyhow!("too few inputs: needed >{input_cursor}"))?
                    .clone();
                ctx[idx] = Some(tr);
                input_cursor += 1;
                continue;
            }

            let loaded = self.nodes[idx].as_ref().unwrap();
            let output_shape = resolve_shape(&loaded.output_shape, batch_size);

            // Gather activation inputs from the context.
            let parent_refs: Vec<&TensorRef> = self.parents[idx].iter()
                .map(|&p| ctx[p].as_ref().expect("parent must be computed before child"))
                .collect();

            // Allocate output buffer.
            let n_elems: usize = output_shape.iter().product();
            let byte_size = n_elems * dtype_bytes(loaded.output_dtype);
            let out_ptr = mem::alloc(byte_size)?;
            intermediate_ptrs.push(out_ptr);

            // Build arg inputs: (raw ptr, concrete shape slice).
            let act_inputs: Vec<(teeny_core::model::RawPtr, &[usize])> = parent_refs.iter()
                .map(|tr| (tr.ptr as *mut core::ffi::c_void, tr.shape.as_slice()))
                .collect();

            let param_ptrs: Vec<teeny_core::model::RawPtr> = loaded.param_bufs.iter()
                .map(|&p| p as *mut core::ffi::c_void)
                .collect();

            // Pack arguments via RuntimeOp.
            let mut packer = CudaArgPacker::new();
            loaded.runtime_op.pack_args(
                &act_inputs,
                &param_ptrs,
                out_ptr as *mut core::ffi::c_void,
                &output_shape,
                &mut packer,
            );

            // Compute launch config.
            let grid = loaded.runtime_op.grid(&output_shape);
            let block = loaded.runtime_op.block();
            let cfg = CudaLaunchConfig { grid, block, cluster: [1, 1, 1] };

            device.launch_with_packer(&loaded.program, &cfg, &mut packer)?;

            ctx[idx] = Some(TensorRef::new(out_ptr, output_shape));
        }

        let last_idx = *topo.last().ok_or_else(|| anyhow!("empty model"))?;
        let result = ctx[last_idx].clone()
            .ok_or_else(|| anyhow!("last node produced no output"))?;

        // Free all intermediate buffers except the output of the last node.
        // The last node's buffer is returned to the caller (who must free it).
        for ptr in intermediate_ptrs {
            if ptr != result.ptr {
                let _ = mem::free(ptr).map_err(|e| {
                    eprintln!("LoadedModel::forward: failed to free intermediate buffer: {e}");
                });
            }
        }

        Ok(result)
    }
}
