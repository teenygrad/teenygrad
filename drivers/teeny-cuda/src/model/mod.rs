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

use std::{collections::HashMap, marker::PhantomData, sync::Arc};

use anyhow::anyhow;
use teeny_core::{
    device::program::ArgVisitor,
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
    errors::{Error, Result},
};

// ---------------------------------------------------------------------------
// Inline PTX for GPU-side f32 gradient accumulation: dst[i] += src[i]
// ---------------------------------------------------------------------------

#[cfg(feature = "training")]
const GRAD_ACCUM_F32_PTX: &[u8] = b"\
// meta:name=grad_accum_f32\n\
// meta:num_warps=4\n\
.version 7.0\n\
.target sm_60\n\
.address_size 64\n\
\n\
.visible .entry grad_accum_f32(\n\
    .param .u64 param0,\n\
    .param .u64 param1,\n\
    .param .u32 param2,\n\
    .param .u64 param3,\n\
    .param .u64 param4\n\
)\n\
{\n\
    .reg .pred %p0;\n\
    .reg .u32 %r<5>;\n\
    .reg .u64 %rd<5>;\n\
    .reg .f32 %f<3>;\n\
    ld.param.u64 %rd0, [param0];\n\
    ld.param.u64 %rd1, [param1];\n\
    ld.param.u32 %r0,  [param2];\n\
    mov.u32 %r1, %ctaid.x;\n\
    mov.u32 %r2, %ntid.x;\n\
    mov.u32 %r3, %tid.x;\n\
    mad.lo.u32 %r4, %r1, %r2, %r3;\n\
    setp.ge.u32 %p0, %r4, %r0;\n\
    @%p0 bra $L__return;\n\
    mul.wide.u32 %rd2, %r4, 4;\n\
    add.u64 %rd3, %rd0, %rd2;\n\
    add.u64 %rd4, %rd1, %rd2;\n\
    ld.global.f32 %f0, [%rd3];\n\
    ld.global.f32 %f1, [%rd4];\n\
    add.f32 %f2, %f0, %f1;\n\
    st.global.f32 [%rd3], %f2;\n\
$L__return:\n\
    ret;\n\
}\n\
";

// ---------------------------------------------------------------------------
// TensorRef — a device buffer pointer with a concrete runtime shape
// ---------------------------------------------------------------------------

/// A reference to a device-side tensor: raw device pointer + concrete shape.
///
/// `shape` is always fully concrete (no `None` dims).
#[derive(Clone, Debug)]
pub struct TensorRef {
    /// The device pointer to this tensor's data.
    pub ptr: DevicePtr,
    /// This tensor's concrete shape.
    pub shape: Vec<usize>,
}

impl TensorRef {
    /// Wraps an existing device pointer and shape as a `TensorRef`.
    pub fn new(ptr: DevicePtr, shape: Vec<usize>) -> Self {
        Self { ptr, shape }
    }

    /// Total number of elements (product of `shape`).
    pub fn n_elements(&self) -> usize {
        self.shape.iter().product()
    }

    /// Allocate a device buffer, copy `data` to it, and return a `TensorRef`.
    ///
    /// `data.len()` must equal `shape.iter().product()`.
    /// The caller owns the allocation; call [`TensorRef::free`] when done.
    pub fn from_host_f32(data: &[f32], shape: Vec<usize>) -> Result<Self> {
        assert_eq!(
            data.len(),
            shape.iter().product::<usize>(),
            "data length must match shape product"
        );
        let ptr = mem::alloc(std::mem::size_of_val(data))?;
        unsafe { mem::copy_h_to_d(ptr, data.as_ptr(), data.len()) }?;
        Ok(Self { ptr, shape })
    }

    /// Copy the device buffer contents to a host `Vec<f32>`.
    pub fn to_host_f32(&self) -> Result<Vec<f32>> {
        let n = self.n_elements();
        let mut out = vec![0.0_f32; n];
        unsafe { mem::copy_d_to_h(out.as_mut_ptr(), self.ptr, n) }?;
        Ok(out)
    }

    /// Free the underlying device buffer.
    ///
    /// Only call this on `TensorRef`s that own their allocation (created via
    /// [`TensorRef::from_host_f32`] or [`TensorRef::new`] with a freshly
    /// allocated pointer).  Do **not** call this on refs borrowed from an
    /// [`ActivationCache`] — the cache frees them on drop.
    pub fn free(self) -> Result<()> {
        mem::free(self.ptr)
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

/// One PTX-compiled graph node: where its compiled kernel lives on disk, its output
/// shape/dtype, and how to dispatch it at runtime.
pub struct CompiledNode {
    /// Path to the compiled `.o` PTX file. Empty for `Input` placeholder nodes.
    pub ptx_path: String,
    /// The kernel's entry point symbol name.
    pub entry_point: String,
    /// This node's output shape.
    pub output_shape: Shape,
    /// This node's output dtype.
    pub output_dtype: DtypeRepr,
    /// Runtime dispatch: arg-packing + grid computation. `None` for Input nodes.
    pub runtime_op: Option<Arc<dyn RuntimeOp>>,
    /// Path to compiled backward PTX. `None` if no backward kernel for this op.
    #[cfg(feature = "training")]
    pub backward_ptx_path: Option<String>,
    /// Entry point name for the backward kernel.
    #[cfg(feature = "training")]
    pub backward_entry_point: String,
}

// ---------------------------------------------------------------------------
// CudaModel — compiled but not yet loaded into GPU memory
// ---------------------------------------------------------------------------

/// A compiled model: a DAG of [`CompiledNode`]s, not yet loaded into GPU memory. Call
/// [`CudaModel::load`] to load it and get a runnable `LoadedModel`.
pub struct CudaModel<'a> {
    /// The compiled node DAG.
    pub dag: Dag<CompiledNode>,
    /// DAG node index → dotted name (e.g. `"model.0.conv"`), populated from the
    /// source `Graph::names` field during compilation.
    pub names: HashMap<usize, String>,
    _marker: PhantomData<&'a ()>,
}

impl<'a> Model<'a> for CudaModel<'a> {
    type Input = TensorRef;
    type Output = TensorRef;

    fn forward(&self, _input: Self::Input) -> teeny_core::errors::Result<Self::Output> {
        Err(anyhow!(
            "call CudaModel::load() first, then LoadedModel::forward()"
        ))
    }
}

impl<'a> CudaModel<'a> {
    /// Wraps a compiled node DAG as a `CudaModel`, with no node names.
    pub fn new(dag: Dag<CompiledNode>) -> Result<Self> {
        Ok(Self {
            dag,
            names: HashMap::new(),
            _marker: PhantomData,
        })
    }

    /// Wraps a compiled node DAG as a `CudaModel`, with the given node-index → name mapping.
    pub fn with_names(dag: Dag<CompiledNode>, names: HashMap<usize, String>) -> Result<Self> {
        Ok(Self {
            dag,
            names,
            _marker: PhantomData,
        })
    }

    /// Load all compiled PTX kernels into GPU memory and pre-allocate
    /// zero-initialised parameter buffers, producing a `LoadedModel` ready
    /// for inference.
    ///
    /// `batch_size` resolves dynamic (`None`) shape dimensions when computing
    /// parameter buffer sizes.
    pub fn load(self, _device: &CudaDevice<'_>, batch_size: usize) -> Result<LoadedModel> {
        let names = self.names;
        let n = self.dag.len();
        let topo = self.dag.topological_sort();

        // Snapshot parent lists before consuming the dag.
        let parents: Vec<Vec<usize>> = (0..n).map(|i| self.dag.node(i).parents.clone()).collect();

        // Consume the dag into (parents, CompiledNode) pairs.
        let compiled: Vec<CompiledNode> = self.dag.into_iter().map(|node| node.value).collect();

        let mut loaded_nodes: Vec<Option<LoadedNode>> = (0..n).map(|_| None).collect();

        // Track correctly-computed concrete shapes for each node so that ops whose
        // first dimension is `k * batch_size` (e.g. attention pack/unpack) propagate
        // the true shape rather than the naive `batch_size` substitution.
        let mut concrete_shapes: Vec<Vec<usize>> = compiled
            .iter()
            .map(|cn| resolve_shape(&cn.output_shape, batch_size))
            .collect();

        for &idx in &topo {
            let cn = &compiled[idx];
            let Some(rop) = cn.runtime_op.as_ref() else {
                // Input placeholder: shape is already correct in concrete_shapes.
                continue;
            };

            // Gather concrete input shapes, using the correctly-propagated shapes.
            let parent_shapes: Vec<Vec<usize>> = parents[idx]
                .iter()
                .map(|&p| concrete_shapes[p].clone())
                .collect();
            let parent_shape_refs: Vec<&[usize]> =
                parent_shapes.iter().map(|s| s.as_slice()).collect();
            let raw_output_shape = resolve_shape(&cn.output_shape, batch_size);
            let output_shape =
                rop.compute_concrete_output_shape(&parent_shape_refs, &raw_output_shape);
            concrete_shapes[idx] = output_shape.clone();

            // Allocate and zero-init device buffers for each parameter slot.
            let p_shapes = rop.param_shapes(&parent_shape_refs, &output_shape);
            let mut param_bufs: Vec<DevicePtr> = Vec::with_capacity(p_shapes.len());
            for (pi, ps) in p_shapes.iter().enumerate() {
                let n_elems: usize = ps.iter().product();
                let byte_size = n_elems * dtype_bytes(cn.output_dtype);
                let ptr = mem::alloc(byte_size)?;
                unsafe { cuda::cuMemsetD8_v2(ptr, 0, byte_size) };
                if let Some(cpu_bytes) = rop.param_init_data(pi) {
                    unsafe { mem::copy_h_to_d::<u8>(ptr, cpu_bytes.as_ptr(), cpu_bytes.len())? };
                }
                param_bufs.push(ptr);
            }

            // JIT-compile the PTX via the CUDA driver.
            let ptx = std::fs::read(&cn.ptx_path)
                .map_err(|e| anyhow!("failed to read PTX for node {idx}: {e}"))?;
            let program = CudaProgram::<ErasedKernel>::try_from_ptx(&ptx)?;

            #[cfg(feature = "training")]
            let backward_program = if let Some(ref bwd_path) = cn.backward_ptx_path {
                let bwd_ptx = std::fs::read(bwd_path)
                    .map_err(|e| anyhow!("failed to read backward PTX for node {idx}: {e}"))?;
                Some(CudaProgram::<ErasedKernel>::try_from_ptx(&bwd_ptx)?)
            } else {
                None
            };

            // Allocate zero-initialised gradient + optimizer state buffers per param.
            #[cfg(feature = "training")]
            let (grad_param_bufs, optim_m_bufs, optim_v_bufs) = {
                let mut grads = Vec::with_capacity(p_shapes.len());
                let mut ms = Vec::with_capacity(p_shapes.len());
                let mut vs = Vec::with_capacity(p_shapes.len());
                for ps in &p_shapes {
                    let n_elems: usize = ps.iter().product();
                    let byte_size = n_elems * dtype_bytes(cn.output_dtype);
                    let gp = mem::alloc(byte_size)?;
                    let mp = mem::alloc(byte_size)?;
                    let vp = mem::alloc(byte_size)?;
                    unsafe {
                        cuda::cuMemsetD8_v2(gp, 0, byte_size);
                        cuda::cuMemsetD8_v2(mp, 0, byte_size);
                        cuda::cuMemsetD8_v2(vp, 0, byte_size);
                    }
                    grads.push(gp);
                    ms.push(mp);
                    vs.push(vp);
                }
                (grads, ms, vs)
            };

            loaded_nodes[idx] = Some(LoadedNode {
                program,
                output_shape: cn.output_shape.clone(),
                output_dtype: cn.output_dtype,
                runtime_op: Arc::clone(rop),
                param_bufs,
                param_shapes: p_shapes,
                #[cfg(feature = "training")]
                backward_program,
                #[cfg(feature = "training")]
                grad_param_bufs,
                #[cfg(feature = "training")]
                optim_m_bufs,
                #[cfg(feature = "training")]
                optim_v_bufs,
            });
        }

        Ok(LoadedModel {
            nodes: loaded_nodes,
            parents,
            names,
            #[cfg(feature = "training")]
            optim_step: 0,
            #[cfg(feature = "training")]
            accum_program: None,
        })
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
    /// Compiled backward kernel. `None` if this op has no backward.
    #[cfg(feature = "training")]
    backward_program: Option<CudaProgram<'static, ErasedKernel>>,
    /// Per-parameter gradient buffers (dW, db …), same shapes as `param_bufs`.
    #[cfg(feature = "training")]
    grad_param_bufs: Vec<DevicePtr>,
    /// AdamW first-moment (exp_avg) per parameter, same shapes as `param_bufs`.
    #[cfg(feature = "training")]
    optim_m_bufs: Vec<DevicePtr>,
    /// AdamW second-moment (exp_avg_sq) per parameter, same shapes as `param_bufs`.
    #[cfg(feature = "training")]
    optim_v_bufs: Vec<DevicePtr>,
}

impl Drop for LoadedNode {
    fn drop(&mut self) {
        for &ptr in &self.param_bufs {
            if let Err(e) = mem::free(ptr) {
                eprintln!("LoadedNode: failed to free param buffer: {e}");
            }
        }
        #[cfg(feature = "training")]
        for &ptr in &self.grad_param_bufs {
            if let Err(e) = mem::free(ptr) {
                eprintln!("LoadedNode: failed to free grad param buffer: {e}");
            }
        }
        #[cfg(feature = "training")]
        for &ptr in &self.optim_m_bufs {
            if let Err(e) = mem::free(ptr) {
                eprintln!("LoadedNode: failed to free optim m buffer: {e}");
            }
        }
        #[cfg(feature = "training")]
        for &ptr in &self.optim_v_bufs {
            if let Err(e) = mem::free(ptr) {
                eprintln!("LoadedNode: failed to free optim v buffer: {e}");
            }
        }
    }
}

// ---------------------------------------------------------------------------
// LoadedModel — eager-loaded model ready for inference
// ---------------------------------------------------------------------------

/// A [`CudaModel`] with all kernels loaded into GPU memory and parameter buffers allocated,
/// ready to run inference (and, with the `training` feature, training steps).
pub struct LoadedModel {
    /// Per-DAG-node loaded kernel. `None` for `Input` placeholder nodes.
    nodes: Vec<Option<LoadedNode>>,
    /// Parent node indices per node (same topology as the compiled DAG).
    parents: Vec<Vec<usize>>,
    /// DAG node index → dotted name (e.g. `"model.0.conv"`).
    names: HashMap<usize, String>,
    /// AdamW step counter for bias correction (incremented each `adamw_step` call).
    #[cfg(feature = "training")]
    optim_step: u32,
    /// Lazily-compiled f32 gradient accumulation kernel (`dst[i] += src[i]`).
    #[cfg(feature = "training")]
    accum_program: Option<CudaProgram<'static, ErasedKernel>>,
}

impl LoadedModel {
    /// Iterate over every node that has parameter buffers.
    ///
    /// Yields `(node_idx, param_shapes)` where `param_shapes[i]` is the concrete
    /// shape of parameter slot `i` (e.g. `[out_features, in_features]` for a
    /// weight matrix). Use `load_param_f32(node_idx, i, data)` to upload values.
    pub fn param_info(&self) -> impl Iterator<Item = (usize, &[Vec<usize>])> {
        self.nodes.iter().enumerate().filter_map(|(idx, node)| {
            node.as_ref()
                .filter(|n| !n.param_shapes.is_empty())
                .map(|n| (idx, n.param_shapes.as_slice()))
        })
    }

    /// Iterate over every named parameter slot.
    ///
    /// Yields `(full_key, node_idx, param_idx)` where `full_key` is the dotted
    /// safetensors key (e.g. `"model.0.conv.weight"`), built by joining the
    /// node name from the graph with the slot name from the runtime op.
    /// Nodes without a name or without parameters are skipped.
    pub fn param_info_named(&self) -> impl Iterator<Item = (String, usize, usize)> + '_ {
        self.nodes
            .iter()
            .enumerate()
            .filter_map(|(node_idx, node)| {
                let n = node.as_ref().filter(|n| !n.param_shapes.is_empty())?;
                let node_name = self.names.get(&node_idx)?;
                let slot_names = n.runtime_op.param_names();
                Some((node_idx, node_name, n, slot_names))
            })
            .flat_map(|(node_idx, node_name, n, slot_names)| {
                (0..n.param_shapes.len()).filter_map(move |param_idx| {
                    let slot = slot_names.get(param_idx).copied().unwrap_or("");
                    if slot.is_empty() {
                        return None;
                    }
                    let key = format!("{node_name}.{slot}");
                    Some((key, node_idx, param_idx))
                })
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
        let node = self.nodes[node_idx]
            .as_ref()
            .ok_or_else(|| anyhow!("node {node_idx} is an Input placeholder"))?;
        let ptr = *node
            .param_bufs
            .get(param_idx)
            .ok_or_else(|| anyhow!("node {node_idx} has no param at index {param_idx}"))?;
        unsafe { mem::copy_h_to_d(ptr, data.as_ptr(), data.len()) }
    }

    /// Copy the accumulated parameter gradient (dL/dParam) back to host as `f32`.
    ///
    /// Call after `backward` and before `zero_grad`.
    #[cfg(feature = "training")]
    pub fn read_param_grad_f32(&self, node_idx: usize, param_idx: usize) -> Result<Vec<f32>> {
        let node = self.nodes[node_idx]
            .as_ref()
            .ok_or_else(|| anyhow!("node {node_idx} is an Input placeholder"))?;
        let &ptr = node
            .grad_param_bufs
            .get(param_idx)
            .ok_or_else(|| anyhow!("node {node_idx} has no grad param at index {param_idx}"))?;
        let n_elems: usize = node.param_shapes[param_idx].iter().product();
        let mut out = vec![0.0_f32; n_elems];
        unsafe { mem::copy_d_to_h(out.as_mut_ptr(), ptr, n_elems) }?;
        Ok(out)
    }

    /// Return the parent node indices for a DAG node.
    pub fn node_parents(&self, idx: usize) -> &[usize] {
        self.parents.get(idx).map(|v| v.as_slice()).unwrap_or(&[])
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
        let topo = self.topo_sort();

        // ctx[i] = TensorRef for node i once it has been computed.
        let mut ctx: Vec<Option<TensorRef>> = vec![None; n];
        // Intermediate output buffers that we own and must free.
        let mut intermediate_ptrs: Vec<DevicePtr> = Vec::new();
        let mut input_cursor = 0usize;

        for &idx in &topo {
            if self.nodes[idx].is_none() {
                // Input placeholder — assign from caller-provided inputs.
                let tr = inputs
                    .get(input_cursor)
                    .ok_or_else(|| anyhow!("too few inputs: needed >{input_cursor}"))?
                    .clone();
                ctx[idx] = Some(tr);
                input_cursor += 1;
                continue;
            }

            let loaded = self.nodes[idx].as_ref().unwrap();

            // Gather activation inputs from the context.
            let parent_refs: Vec<&TensorRef> = self.parents[idx]
                .iter()
                .map(|&p| {
                    ctx[p]
                        .as_ref()
                        .expect("parent must be computed before child")
                })
                .collect();
            let act_input_shapes: Vec<&[usize]> =
                parent_refs.iter().map(|tr| tr.shape.as_slice()).collect();
            let raw_output_shape = resolve_shape(&loaded.output_shape, batch_size);
            let output_shape = loaded
                .runtime_op
                .compute_concrete_output_shape(&act_input_shapes, &raw_output_shape);

            // Allocate tight output buffer.
            let n_elems: usize = output_shape.iter().product();
            let elem_bytes = dtype_bytes(loaded.output_dtype);
            let byte_size = n_elems * elem_bytes;
            let out_ptr = mem::alloc(byte_size)?;
            intermediate_ptrs.push(out_ptr);

            // Some ops (e.g. linear_forward) use TMA and require a row stride that
            // is a multiple of 16 bytes.  Allocate a padded output buffer when needed.
            let natural_stride = loaded
                .runtime_op
                .forward_output_row_elems(&output_shape);
            let required_stride = loaded.runtime_op.forward_output_row_stride(&output_shape);
            let n_rows = output_shape.iter().product::<usize>() / natural_stride.max(1);

            let (kernel_out_ptr, padded_out) = if required_stride > natural_stride {
                let padded_bytes = n_rows * required_stride * elem_bytes;
                let padded = mem::alloc(padded_bytes)?;
                unsafe {
                    cuda::cuMemsetD8_v2(padded, 0, padded_bytes);
                }
                (padded, Some(padded))
            } else {
                (out_ptr, None)
            };

            // Build arg inputs: (raw ptr, concrete shape slice).
            let act_inputs: Vec<(teeny_core::model::RawPtr, &[usize])> = parent_refs
                .iter()
                .map(|tr| (tr.ptr as *mut core::ffi::c_void, tr.shape.as_slice()))
                .collect();

            let param_ptrs: Vec<teeny_core::model::RawPtr> = loaded
                .param_bufs
                .iter()
                .map(|&p| p as *mut core::ffi::c_void)
                .collect();

            // Launch kernel(s). Most ops need one launch; multi-input scatter
            // ops (e.g. channel-cat) set n_launches > 1.
            let n_launches = loaded.runtime_op.n_launches();
            let input_shapes: Vec<&[usize]> = act_inputs.iter().map(|(_, s)| *s).collect();
            let block = [loaded.program.metadata.threads_per_block(), 1, 1];
            let cluster = [loaded.program.metadata.num_ctas, 1, 1];
            let out_raw = kernel_out_ptr as *mut core::ffi::c_void;

            let mut last_result = Ok(());
            for launch_idx in 0..n_launches {
                let mut packer = CudaArgPacker::new();
                if n_launches == 1 {
                    loaded.runtime_op.pack_args(
                        &act_inputs,
                        &param_ptrs,
                        out_raw,
                        &output_shape,
                        required_stride as i32,
                        &mut packer,
                    );
                } else {
                    loaded.runtime_op.pack_args_for_launch(
                        launch_idx,
                        &act_inputs,
                        &param_ptrs,
                        out_raw,
                        &output_shape,
                        required_stride as i32,
                        &mut packer,
                    );
                }
                let grid = if n_launches == 1 {
                    loaded.runtime_op.grid(&output_shape)
                } else {
                    loaded
                        .runtime_op
                        .grid_for_launch(launch_idx, &input_shapes, &output_shape)
                };
                last_result = device.launch_with_packer(
                    &loaded.program,
                    &CudaLaunchConfig {
                        grid,
                        block,
                        cluster,
                    },
                    &mut packer,
                );
                if last_result.is_err() {
                    break;
                }
            }

            // Copy valid rows from padded output back to tight buffer, then free padded.
            if let Some(padded) = padded_out {
                if last_result.is_ok() {
                    mem::copy_rows_d_to_d(
                        out_ptr,
                        natural_stride * elem_bytes,
                        padded,
                        required_stride * elem_bytes,
                        natural_stride * elem_bytes,
                        n_rows,
                    )?;
                }
                mem::free(padded).ok();
            }
            last_result?;

            ctx[idx] = Some(TensorRef::new(out_ptr, output_shape));
        }

        let last_idx = *topo.last().ok_or_else(|| anyhow!("empty model"))?;
        let result = ctx[last_idx]
            .clone()
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

    // ── Training-only methods ────────────────────────────────────────────────

    /// Run a forward pass and retain ALL intermediate activation buffers.
    ///
    /// Returns `(final_output, activation_cache)` where `activation_cache[i]`
    /// is the output tensor of node `i`. Call `drop(cache)` after `backward`
    /// to release the device buffers.
    #[cfg(feature = "training")]
    pub fn forward_train(
        &self,
        device: &CudaDevice<'_>,
        batch_size: usize,
        inputs: &[TensorRef],
    ) -> Result<(TensorRef, ActivationCache)> {
        let n = self.nodes.len();
        let topo = self.topo_sort();

        let mut ctx: Vec<Option<TensorRef>> = vec![None; n];
        let mut input_cursor = 0usize;

        for &idx in &topo {
            if self.nodes[idx].is_none() {
                let tr = inputs
                    .get(input_cursor)
                    .ok_or_else(|| anyhow!("too few inputs: needed >{input_cursor}"))?
                    .clone();
                ctx[idx] = Some(tr);
                input_cursor += 1;
                continue;
            }

            let loaded = self.nodes[idx].as_ref().unwrap();

            let parent_refs: Vec<&TensorRef> = self.parents[idx]
                .iter()
                .map(|&p| {
                    ctx[p]
                        .as_ref()
                        .expect("parent must be computed before child")
                })
                .collect();
            let act_inputs: Vec<(teeny_core::model::RawPtr, &[usize])> = parent_refs
                .iter()
                .map(|tr| (tr.ptr as *mut core::ffi::c_void, tr.shape.as_slice()))
                .collect();
            let param_ptrs: Vec<teeny_core::model::RawPtr> = loaded
                .param_bufs
                .iter()
                .map(|&p| p as *mut core::ffi::c_void)
                .collect();

            let input_shapes: Vec<&[usize]> = act_inputs.iter().map(|(_, s)| *s).collect();
            let raw_output_shape = resolve_shape(&loaded.output_shape, batch_size);
            let output_shape = loaded
                .runtime_op
                .compute_concrete_output_shape(&input_shapes, &raw_output_shape);

            let n_elems: usize = output_shape.iter().product();
            let elem_bytes = dtype_bytes(loaded.output_dtype);
            let byte_size = n_elems * elem_bytes;
            let out_ptr = mem::alloc(byte_size)?;

            // TMA alignment: allocate a padded output buffer when the op requires it.
            let natural_stride = loaded
                .runtime_op
                .forward_output_row_elems(&output_shape);
            let required_stride = loaded.runtime_op.forward_output_row_stride(&output_shape);
            let n_rows = output_shape.iter().product::<usize>() / natural_stride.max(1);

            let (kernel_out_ptr, padded_out) = if required_stride > natural_stride {
                let padded_bytes = n_rows * required_stride * elem_bytes;
                let padded = mem::alloc(padded_bytes)?;
                unsafe {
                    cuda::cuMemsetD8_v2(padded, 0, padded_bytes);
                }
                (padded, Some(padded))
            } else {
                (out_ptr, None)
            };

            let n_launches = loaded.runtime_op.n_launches();
            let block = [loaded.program.metadata.threads_per_block(), 1, 1];
            let cluster = [loaded.program.metadata.num_ctas, 1, 1];
            let out_raw = kernel_out_ptr as *mut core::ffi::c_void;

            let mut launch_result = Ok(());
            for launch_idx in 0..n_launches {
                let mut packer = CudaArgPacker::new();
                if n_launches == 1 {
                    loaded.runtime_op.pack_args(
                        &act_inputs,
                        &param_ptrs,
                        out_raw,
                        &output_shape,
                        required_stride as i32,
                        &mut packer,
                    );
                } else {
                    loaded.runtime_op.pack_args_for_launch(
                        launch_idx,
                        &act_inputs,
                        &param_ptrs,
                        out_raw,
                        &output_shape,
                        required_stride as i32,
                        &mut packer,
                    );
                }
                let grid = if n_launches == 1 {
                    loaded.runtime_op.grid(&output_shape)
                } else {
                    loaded
                        .runtime_op
                        .grid_for_launch(launch_idx, &input_shapes, &output_shape)
                };
                launch_result = device.launch_with_packer(
                    &loaded.program,
                    &CudaLaunchConfig {
                        grid,
                        block,
                        cluster,
                    },
                    &mut packer,
                );
                if launch_result.is_err() {
                    break;
                }
            }

            if let Some(padded) = padded_out {
                if launch_result.is_ok() {
                    mem::copy_rows_d_to_d(
                        out_ptr,
                        natural_stride * elem_bytes,
                        padded,
                        required_stride * elem_bytes,
                        natural_stride * elem_bytes,
                        n_rows,
                    )?;
                }
                mem::free(padded).ok();
            }
            launch_result?;

            ctx[idx] = Some(TensorRef::new(out_ptr, output_shape));
        }

        let last_idx = *topo.last().ok_or_else(|| anyhow!("empty model"))?;
        let output = ctx[last_idx]
            .clone()
            .ok_or_else(|| anyhow!("last node produced no output"))?;

        Ok((output, ActivationCache { tensors: ctx }))
    }

    /// Zero all parameter gradient buffers. Call before each backward pass.
    #[cfg(feature = "training")]
    pub fn zero_grad(&mut self) {
        for node in self.nodes.iter().flatten() {
            for (&gp, ps) in node.grad_param_bufs.iter().zip(node.param_shapes.iter()) {
                let byte_size = ps.iter().product::<usize>() * dtype_bytes(node.output_dtype);
                unsafe {
                    cuda::cuMemsetD8_v2(gp, 0, byte_size);
                }
            }
        }
    }

    /// Apply an AdamW update to all parameters using the accumulated gradient buffers.
    ///
    /// `kernel` — pre-compiled `adamw_step` PTX (compile with `AdamwStep::new(1024)` from
    ///            `teeny_kernels::nn::optim::adam`).
    // AdamW's hyperparameters (lr, beta1, beta2, eps, weight_decay) are each independently
    // meaningful at call sites; bundling them into a struct wouldn't be clearer.
    #[cfg(feature = "training")]
    #[allow(clippy::too_many_arguments)]
    pub fn adamw_step(
        &mut self,
        device: &CudaDevice<'_>,
        kernel: &AdamwKernel,
        lr: f32,
        beta1: f32,
        beta2: f32,
        eps: f32,
        weight_decay: f32,
    ) -> Result<()> {
        self.optim_step += 1;
        let bias_correction1 = 1.0_f32 - beta1.powi(self.optim_step as i32);
        let bias_correction2 = 1.0_f32 - beta2.powi(self.optim_step as i32);
        let step_size = lr / bias_correction1;
        let bias_corr2_sqrt = bias_correction2.sqrt();

        for node in self.nodes.iter().flatten() {
            if node.param_bufs.is_empty() {
                continue;
            }
            for i in 0..node.param_bufs.len() {
                let n_elems: usize = node.param_shapes[i].iter().product();
                let mut packer = CudaArgPacker::new();
                packer.visit_ptr(node.param_bufs[i] as *mut core::ffi::c_void); // params_ptr
                packer.visit_ptr(node.grad_param_bufs[i] as *mut core::ffi::c_void); // grad_ptr
                packer.visit_ptr(node.optim_m_bufs[i] as *mut core::ffi::c_void); // exp_avg_ptr
                packer.visit_ptr(node.optim_v_bufs[i] as *mut core::ffi::c_void); // exp_avg_sq_ptr
                packer.visit_i32(n_elems as i32); // n_elements
                packer.visit_f32(step_size); // step_size
                packer.visit_f32(bias_corr2_sqrt); // bias_corr2_sqrt
                packer.visit_f32(beta1); // beta1
                packer.visit_f32(beta2); // beta2
                packer.visit_f32(eps); // eps
                packer.visit_f32(weight_decay); // weight_decay
                packer.visit_f32(lr); // lr

                let threads = kernel.program.metadata.threads_per_block();
                let grid = [n_elems.div_ceil(threads as usize) as u32, 1, 1];
                let block = [threads, 1, 1];
                device.launch_with_packer(
                    &kernel.program,
                    &CudaLaunchConfig {
                        grid,
                        block,
                        cluster: [1, 1, 1],
                    },
                    &mut packer,
                )?;
            }
        }
        Ok(())
    }

    /// Indices of all DAG nodes that have no children (sinks / output nodes).
    ///
    /// For single-output models this returns one element.  YOLO26 returns two:
    /// the boxes node and the scores node.
    pub fn terminal_node_indices(&self) -> Vec<usize> {
        let n = self.nodes.len();
        let mut has_child = vec![false; n];
        for i in 0..n {
            for &p in &self.parents[i] {
                has_child[p] = true;
            }
        }
        (0..n).filter(|&i| !has_child[i]).collect()
    }

    /// Terminal node indices sorted by output tensor element count (ascending).
    ///
    /// For YOLO26 this reliably gives `[boxes_idx, scores_idx]`: boxes has
    /// `4·A` elements in the channel dim while scores has `nc·A`, and nc > 4
    /// for all practical detection models.
    /// Return the name for a node index, if one was recorded during compilation.
    pub fn node_name(&self, idx: usize) -> Option<&str> {
        self.names.get(&idx).map(|s| s.as_str())
    }

    /// Terminal (no-consumer) node indices, sorted by output element count (largest last).
    pub fn terminal_node_indices_sorted_by_size(&self) -> Vec<usize> {
        let mut terminals = self.terminal_node_indices();
        terminals.sort_by_key(|&i| {
            self.nodes[i]
                .as_ref()
                .map(|n| n.output_shape.iter().filter_map(|&d| d).product::<usize>())
                .unwrap_or(0)
        });
        terminals
    }

    /// Backward pass seeded from a single output node (common case for single-output models).
    ///
    /// `grad_output` — dL/d(model_output), provided by the loss backward.
    /// `cache`       — the activation cache returned by `forward_train`.
    #[cfg(feature = "training")]
    pub fn backward(
        &mut self,
        device: &CudaDevice<'_>,
        batch_size: usize,
        grad_output: TensorRef,
        cache: &ActivationCache,
    ) -> Result<()> {
        let topo = self.topo_sort();
        let last_idx = *topo.last().ok_or_else(|| anyhow!("empty model"))?;
        self.backward_multi(device, batch_size, &[(last_idx, grad_output)], cache)
    }

    /// Backward pass seeded from multiple output nodes (e.g. YOLO26 boxes + scores).
    ///
    /// `seed_grads` — list of `(node_idx, grad_tensor)` pairs, one per output node.
    /// `cache`      — the activation cache returned by `forward_train`.
    #[cfg(feature = "training")]
    pub fn backward_multi(
        &mut self,
        device: &CudaDevice<'_>,
        batch_size: usize,
        seed_grads: &[(usize, TensorRef)],
        cache: &ActivationCache,
    ) -> Result<()> {
        // Lazy-compile the gradient accumulation kernel on first use.
        if self.accum_program.is_none() {
            self.accum_program = Some(CudaProgram::<ErasedKernel>::try_from_ptx(
                GRAD_ACCUM_F32_PTX,
            )?);
        }

        let n = self.nodes.len();
        let topo = self.topo_sort();

        // grad_ctx[i]: gradient of the loss w.r.t. node i's output (device ptr).
        let mut grad_ctx: Vec<Option<DevicePtr>> = vec![None; n];
        // All intermediate gradient buffers we allocated (freed after backward).
        let mut owned_grad_ptrs: Vec<DevicePtr> = Vec::new();

        for (node_idx, grad) in seed_grads {
            grad_ctx[*node_idx] = Some(grad.ptr);
        }

        for &idx in topo.iter().rev() {
            let grad_in_ptr = match grad_ctx[idx] {
                Some(p) => p,
                None => continue,
            };

            // Clone parent indices early to avoid split borrows.
            let parent_indices: Vec<usize> = self.parents[idx].clone();

            {
                let loaded = match self.nodes[idx].as_ref() {
                    Some(n) => n,
                    None => continue,
                };
                let bwd_prog = match loaded.backward_program.as_ref() {
                    Some(p) => p,
                    None => continue,
                };

                let output_shape = resolve_shape(&loaded.output_shape, batch_size);
                let node_out_ptr = cache.tensors[idx]
                    .as_ref()
                    .ok_or_else(|| anyhow!("activation cache missing for node {idx}"))?
                    .ptr;

                // Gather parent activation refs from cache.
                let parent_trs: Vec<&TensorRef> = parent_indices
                    .iter()
                    .map(|&p| {
                        cache.tensors[p]
                            .as_ref()
                            .expect("activation cache must have parent activation")
                    })
                    .collect();

                let act_inputs: Vec<(teeny_core::model::RawPtr, &[usize])> = parent_trs
                    .iter()
                    .map(|tr| (tr.ptr as *mut core::ffi::c_void, tr.shape.as_slice()))
                    .collect();
                let param_ptrs: Vec<teeny_core::model::RawPtr> = loaded
                    .param_bufs
                    .iter()
                    .map(|&p| p as *mut core::ffi::c_void)
                    .collect();
                let grad_param_rawptrs: Vec<teeny_core::model::RawPtr> = loaded
                    .grad_param_bufs
                    .iter()
                    .map(|&p| p as *mut core::ffi::c_void)
                    .collect();

                // Allocate zero-initialised gradient buffers for each activation parent.
                let mut grad_input_ptrs: Vec<DevicePtr> = Vec::with_capacity(parent_trs.len());
                for tr in &parent_trs {
                    let n_elems: usize = tr.shape.iter().product();
                    let byte_size = n_elems * dtype_bytes(loaded.output_dtype);
                    let gptr = mem::alloc(byte_size)?;
                    unsafe {
                        cuda::cuMemsetD8_v2(gptr, 0, byte_size);
                    }
                    grad_input_ptrs.push(gptr);
                    owned_grad_ptrs.push(gptr);
                }

                let grad_input_rawptrs: Vec<teeny_core::model::RawPtr> = grad_input_ptrs
                    .iter()
                    .map(|&p| p as *mut core::ffi::c_void)
                    .collect();

                let input_shapes: Vec<&[usize]> =
                    parent_trs.iter().map(|tr| tr.shape.as_slice()).collect();

                // Some kernels (e.g. linear_backward) use TMA, which requires
                // 16-byte aligned row strides.  If the natural stride is too
                // small, allocate a zero-padded copy and use the padded stride.
                let natural_stride = output_shape.last().copied().unwrap_or(1);
                let required_stride = loaded
                    .runtime_op
                    .backward_grad_output_row_stride(&output_shape);
                let elem_bytes = dtype_bytes(loaded.output_dtype);
                let n_rows = output_shape.iter().product::<usize>() / natural_stride.max(1);

                let (dy_ptr, padded_dy) = if required_stride > natural_stride {
                    let padded_bytes = n_rows * required_stride * elem_bytes;
                    let padded = mem::alloc(padded_bytes)?;
                    unsafe {
                        cuda::cuMemsetD8_v2(padded, 0, padded_bytes);
                    }
                    mem::copy_rows_d_to_d(
                        padded,
                        required_stride * elem_bytes,
                        grad_in_ptr,
                        natural_stride * elem_bytes,
                        natural_stride * elem_bytes,
                        n_rows,
                    )?;
                    (padded, Some(padded))
                } else {
                    (grad_in_ptr, None)
                };

                let bwd_block = [bwd_prog.metadata.threads_per_block(), 1, 1];
                let bwd_cluster = [bwd_prog.metadata.num_ctas, 1, 1];
                let n_bwd_launches = loaded.runtime_op.n_backward_launches();
                let node_out_raw = node_out_ptr as teeny_core::model::RawPtr;
                let dy_raw = dy_ptr as teeny_core::model::RawPtr;

                let mut bwd_result = Ok(());
                for launch_idx in 0..n_bwd_launches {
                    let mut packer = CudaArgPacker::new();
                    if n_bwd_launches == 1 {
                        loaded.runtime_op.pack_backward_args(
                            &act_inputs,
                            &param_ptrs,
                            node_out_raw,
                            &output_shape,
                            dy_raw,
                            required_stride as i32,
                            &grad_input_rawptrs,
                            &grad_param_rawptrs,
                            &mut packer,
                        );
                    } else {
                        loaded.runtime_op.pack_backward_args_for_launch(
                            launch_idx,
                            &act_inputs,
                            &param_ptrs,
                            node_out_raw,
                            &output_shape,
                            dy_raw,
                            required_stride as i32,
                            &grad_input_rawptrs,
                            &grad_param_rawptrs,
                            &mut packer,
                        );
                    }
                    let grid = if n_bwd_launches == 1 {
                        loaded
                            .runtime_op
                            .backward_grid(&input_shapes, &output_shape)
                    } else {
                        loaded.runtime_op.backward_grid_for_launch(
                            launch_idx,
                            &input_shapes,
                            &output_shape,
                        )
                    };
                    bwd_result = device.launch_with_packer(
                        bwd_prog,
                        &CudaLaunchConfig {
                            grid,
                            block: bwd_block,
                            cluster: bwd_cluster,
                        },
                        &mut packer,
                    );
                    if bwd_result.is_err() {
                        break;
                    }
                }

                // Free the padded dy buffer after the (synchronous) launch completes.
                if let Some(padded) = padded_dy {
                    mem::free(padded).ok();
                }

                bwd_result?;

                // Propagate gradients to parent nodes. If a parent already has a
                // gradient (fan-out node), accumulate: existing += new_contrib.
                for (i, &pidx) in parent_indices.iter().enumerate() {
                    if let Some(existing) = grad_ctx[pidx] {
                        let n_elems: usize = parent_trs[i].shape.iter().product();
                        self.accum_grad_f32(device, existing, grad_input_ptrs[i], n_elems)?;
                    } else {
                        grad_ctx[pidx] = Some(grad_input_ptrs[i]);
                    }
                }
            }
        }

        // Free all intermediate gradient buffers.
        for ptr in owned_grad_ptrs {
            let _ = mem::free(ptr).map_err(|e| {
                eprintln!("LoadedModel::backward_multi: failed to free grad buffer: {e}");
            });
        }

        Ok(())
    }

    /// In-place GPU accumulation: `dst[i] += src[i]` for `n_elems` f32 values.
    #[cfg(feature = "training")]
    fn accum_grad_f32(
        &self,
        device: &CudaDevice<'_>,
        dst: DevicePtr,
        src: DevicePtr,
        n_elems: usize,
    ) -> Result<()> {
        let prog = self
            .accum_program
            .as_ref()
            .expect("accum_program must be initialised before calling accum_grad_f32");
        let threads: u32 = prog.metadata.threads_per_block();
        let grid = [n_elems.div_ceil(threads as usize) as u32, 1, 1];
        let block = [threads, 1, 1];
        let mut packer = CudaArgPacker::new();
        packer.visit_ptr(dst as *mut core::ffi::c_void);
        packer.visit_ptr(src as *mut core::ffi::c_void);
        packer.visit_i32(n_elems as i32);
        device.launch_with_packer(
            prog,
            &CudaLaunchConfig {
                grid,
                block,
                cluster: [1, 1, 1],
            },
            &mut packer,
        )
    }

    /// Capture a fixed-batch CUDA graph for low-overhead repeated inference.
    ///
    /// All device buffers (intermediate activations, scratch pads, TMA padding)
    /// are pre-allocated once. The kernel sequence is then recorded via
    /// `cuStreamBeginCapture_v2` / `cuStreamEndCapture` and instantiated as a
    /// `CUgraphExec`. Subsequent [`CudaGraphModel::run`] calls replay it with a
    /// single `cuGraphLaunch` + `cuStreamSynchronize`.
    ///
    /// # Parameters
    /// - `batch_size` — resolves dynamic (`None`) shape dimensions.
    /// - `input_shapes` — concrete shape per `Input` node, in topological order.
    /// - `output_node_indices` — which DAG node indices to read back as outputs.
    ///   Use [`LoadedModel::terminal_node_indices_sorted_by_size`] to obtain them.
    ///
    /// # Constraints
    /// - Inference-only (no backward pass).
    /// - Fixed topology: re-capture if `batch_size` or input shapes change.
    /// - f32 inputs and outputs assumed for the convenience [`CudaGraphModel::run`] API.
    pub fn capture_graph(
        &self,
        device: &CudaDevice<'_>,
        batch_size: usize,
        input_shapes: &[Vec<usize>],
        output_node_indices: &[usize],
    ) -> Result<CudaGraphModel> {
        let n = self.nodes.len();
        let topo = self.topo_sort();

        // ── Phase 1: concrete output shapes for every node ──────────────────
        let mut concrete_shapes: Vec<Vec<usize>> = vec![vec![]; n];
        let mut input_cursor = 0usize;
        for &idx in &topo {
            if self.nodes[idx].is_none() {
                let shape = input_shapes
                    .get(input_cursor)
                    .ok_or_else(|| anyhow!("capture_graph: input_shapes[{input_cursor}] missing"))?
                    .clone();
                concrete_shapes[idx] = shape;
                input_cursor += 1;
            } else {
                let loaded = self.nodes[idx].as_ref().unwrap();
                let parent_shapes: Vec<&[usize]> = self.parents[idx]
                    .iter()
                    .map(|&p| concrete_shapes[p].as_slice())
                    .collect();
                let raw = resolve_shape(&loaded.output_shape, batch_size);
                concrete_shapes[idx] = loaded
                    .runtime_op
                    .compute_concrete_output_shape(&parent_shapes, &raw);
            }
        }

        // ── Phase 2: pre-allocate all device buffers ─────────────────────────
        // All pointers accumulated here; CudaGraphModel::drop frees them.
        let mut owned: Vec<DevicePtr> = Vec::new();
        let mut node_out_bufs: Vec<DevicePtr> = vec![0; n];
        let mut node_scratch_bufs: Vec<DevicePtr> = vec![0; n];
        type PaddedEntry = Option<(DevicePtr, usize, usize, usize, usize)>;
        // (padded_ptr, natural_stride, required_stride, n_rows, elem_bytes)
        let mut node_padded: Vec<PaddedEntry> = vec![None; n];
        let mut input_bufs: Vec<(DevicePtr, usize)> = Vec::new();

        for &idx in &topo {
            if self.nodes[idx].is_none() {
                // f32 input buffer — filled by the caller before each run().
                let n_elems: usize = concrete_shapes[idx].iter().product();
                let ptr = mem::alloc(n_elems * 4)?;
                owned.push(ptr);
                node_out_bufs[idx] = ptr;
                input_bufs.push((ptr, n_elems));
                continue;
            }

            let loaded = self.nodes[idx].as_ref().unwrap();
            let output_shape = &concrete_shapes[idx];
            let n_elems: usize = output_shape.iter().product();
            let elem_bytes = dtype_bytes(loaded.output_dtype);

            // Tight output buffer for this node.
            let out_ptr = mem::alloc(n_elems * elem_bytes)?;
            owned.push(out_ptr);
            node_out_bufs[idx] = out_ptr;

            // TMA padded buffer when the op requires a wider row stride.
            let natural_stride = loaded.runtime_op.forward_output_row_elems(output_shape);
            let required_stride = loaded.runtime_op.forward_output_row_stride(output_shape);
            if required_stride > natural_stride {
                let n_rows = n_elems / natural_stride.max(1);
                let padded_bytes = n_rows * required_stride * elem_bytes;
                let padded_ptr = mem::alloc(padded_bytes)?;
                unsafe { cuda::cuMemsetD8_v2(padded_ptr, 0, padded_bytes) };
                owned.push(padded_ptr);
                node_padded[idx] = Some((
                    padded_ptr,
                    natural_stride,
                    required_stride,
                    n_rows,
                    elem_bytes,
                ));
            }

            // Global scratch pad for TMA descriptors.
            let grid = loaded.runtime_op.grid(output_shape);
            let num_ctas = (grid[0] * grid[1] * grid[2]) as u64;
            let scratch_total = loaded.program.metadata.global_scratch_size * num_ctas;
            if scratch_total > 0 {
                let scratch_ptr = mem::alloc(scratch_total as usize)?;
                unsafe { cuda::cuMemsetD8_v2(scratch_ptr, 0, scratch_total as usize) };
                owned.push(scratch_ptr);
                node_scratch_bufs[idx] = scratch_ptr;
            }
        }

        // ── Phase 3: create stream and begin graph capture ───────────────────
        let mut stream: cuda::CUstream = std::ptr::null_mut();
        {
            let s = unsafe { cuda::cuStreamCreate(&mut stream, 0) };
            if s != cuda::cudaError_enum_CUDA_SUCCESS {
                for &p in &owned {
                    let _ = mem::free(p);
                }
                return Err(Error::from_cuda_error(s).into());
            }
        }

        {
            let s = unsafe {
                cuda::cuStreamBeginCapture_v2(
                    stream,
                    cuda::CUstreamCaptureMode_enum_CU_STREAM_CAPTURE_MODE_GLOBAL,
                )
            };
            if s != cuda::cudaError_enum_CUDA_SUCCESS {
                unsafe { cuda::cuStreamDestroy_v2(stream) };
                for &p in &owned {
                    let _ = mem::free(p);
                }
                return Err(Error::from_cuda_error(s).into());
            }
        }

        // ── Phase 4: record kernels into the capture stream ──────────────────
        let mut capture_err: Option<anyhow::Error> = None;
        'capture: for &idx in &topo {
            let Some(loaded) = self.nodes[idx].as_ref() else {
                continue;
            };
            let output_shape = &concrete_shapes[idx];

            let act_inputs: Vec<(teeny_core::model::RawPtr, &[usize])> = self.parents[idx]
                .iter()
                .map(|&p| {
                    (
                        node_out_bufs[p] as *mut core::ffi::c_void,
                        concrete_shapes[p].as_slice(),
                    )
                })
                .collect();
            let param_ptrs: Vec<teeny_core::model::RawPtr> = loaded
                .param_bufs
                .iter()
                .map(|&p| p as *mut core::ffi::c_void)
                .collect();
            let input_shapes_ref: Vec<&[usize]> = act_inputs.iter().map(|(_, s)| *s).collect();

            let required_stride = loaded.runtime_op.forward_output_row_stride(output_shape);
            let scratch_ptr = node_scratch_bufs[idx];
            let (kernel_out_raw, tight_out_ptr) = if let Some((padded_ptr, ..)) = node_padded[idx] {
                (padded_ptr as *mut core::ffi::c_void, node_out_bufs[idx])
            } else {
                let op = node_out_bufs[idx];
                (op as *mut core::ffi::c_void, op)
            };

            let block = [loaded.program.metadata.threads_per_block(), 1, 1];
            let cluster = [loaded.program.metadata.num_ctas, 1, 1];
            let n_launches = loaded.runtime_op.n_launches();

            // Re-zero scratch each replay so TMA descriptors start clean.
            if scratch_ptr != 0 {
                let grid = loaded.runtime_op.grid(output_shape);
                let num_ctas = (grid[0] * grid[1] * grid[2]) as u64;
                let scratch_total = loaded.program.metadata.global_scratch_size * num_ctas;
                let s = unsafe {
                    cuda::cuMemsetD8Async(scratch_ptr, 0, scratch_total as usize, stream)
                };
                if s != cuda::cudaError_enum_CUDA_SUCCESS {
                    capture_err = Some(anyhow::anyhow!(
                        "capture_graph: scratch memset failed at node {idx} \
                         (shape={output_shape:?}, scratch_total={scratch_total}): {}",
                        Error::from_cuda_error(s),
                    ));
                    break 'capture;
                }
            }

            for launch_idx in 0..n_launches {
                let mut packer = CudaArgPacker::new();
                if n_launches == 1 {
                    loaded.runtime_op.pack_args(
                        &act_inputs,
                        &param_ptrs,
                        kernel_out_raw,
                        output_shape,
                        required_stride as i32,
                        &mut packer,
                    );
                } else {
                    loaded.runtime_op.pack_args_for_launch(
                        launch_idx,
                        &act_inputs,
                        &param_ptrs,
                        kernel_out_raw,
                        output_shape,
                        required_stride as i32,
                        &mut packer,
                    );
                }
                // Triton trailing args: global scratch pad + profile scratch pad.
                packer.visit_ptr(scratch_ptr as *mut core::ffi::c_void);
                packer.visit_ptr(std::ptr::null_mut());

                let grid = if n_launches == 1 {
                    loaded.runtime_op.grid(output_shape)
                } else {
                    loaded
                        .runtime_op
                        .grid_for_launch(launch_idx, &input_shapes_ref, output_shape)
                };
                let cfg = CudaLaunchConfig {
                    grid,
                    block,
                    cluster,
                };
                if let Err(e) = device.launch_on_stream(&loaded.program, &cfg, &mut packer, stream)
                {
                    capture_err = Some(anyhow::anyhow!(
                        "capture_graph: launch failed at node {idx} \
                         (grid={grid:?}, block={block:?}, cluster={cluster:?}, \
                         shape={output_shape:?}, row_stride={required_stride}, \
                         scratch={}): {e:#}",
                        loaded.program.metadata.global_scratch_size,
                    ));
                    break 'capture;
                }
            }

            // Depad: copy valid rows from padded → tight output buffer (async, captured).
            // Single cuMemcpy2DAsync replaces the previous per-row loop, giving the
            // driver one node in the graph instead of n_rows nodes.
            if let Some((padded_ptr, ns, rs, n_rows, eb)) = node_padded[idx] {
                let params = cuda::CUDA_MEMCPY2D {
                    srcMemoryType: cuda::CUmemorytype_enum_CU_MEMORYTYPE_DEVICE,
                    srcDevice: padded_ptr,
                    srcPitch: rs * eb,
                    dstMemoryType: cuda::CUmemorytype_enum_CU_MEMORYTYPE_DEVICE,
                    dstDevice: tight_out_ptr,
                    dstPitch: ns * eb,
                    WidthInBytes: ns * eb,
                    Height: n_rows,
                    ..Default::default()
                };
                let s = unsafe { cuda::cuMemcpy2DAsync_v2(&params, stream) };
                if s != cuda::cudaError_enum_CUDA_SUCCESS {
                    capture_err = Some(anyhow::anyhow!(
                        "capture_graph: cuMemcpy2DAsync depad failed at node {idx} \
                         (ns={ns}, rs={rs}, n_rows={n_rows}, eb={eb}): {}",
                        Error::from_cuda_error(s),
                    ));
                    break 'capture;
                }
            }
        }

        // ── Phase 5: end capture and instantiate ─────────────────────────────
        let mut graph: cuda::CUgraph = std::ptr::null_mut();
        let end_s = unsafe { cuda::cuStreamEndCapture(stream, &mut graph) };

        if let Some(err) = capture_err {
            // A kernel record step failed — discard the partial graph.
            unsafe {
                if !graph.is_null() {
                    cuda::cuGraphDestroy(graph);
                }
                cuda::cuStreamDestroy_v2(stream);
            }
            for &p in &owned {
                let _ = mem::free(p);
            }
            return Err(err);
        }
        if end_s != cuda::cudaError_enum_CUDA_SUCCESS {
            unsafe {
                if !graph.is_null() {
                    cuda::cuGraphDestroy(graph);
                }
                cuda::cuStreamDestroy_v2(stream);
            }
            for &p in &owned {
                let _ = mem::free(p);
            }
            return Err(anyhow::anyhow!(
                "capture_graph: cuStreamEndCapture failed: {}",
                Error::from_cuda_error(end_s)
            ));
        }

        let mut graph_exec: cuda::CUgraphExec = std::ptr::null_mut();
        let inst_s = unsafe { cuda::cuGraphInstantiateWithFlags(&mut graph_exec, graph, 0) };
        unsafe { cuda::cuGraphDestroy(graph) };
        if inst_s != cuda::cudaError_enum_CUDA_SUCCESS {
            unsafe { cuda::cuStreamDestroy_v2(stream) };
            for &p in &owned {
                let _ = mem::free(p);
            }
            return Err(anyhow::anyhow!(
                "capture_graph: cuGraphInstantiateWithFlags failed: {}",
                Error::from_cuda_error(inst_s)
            ));
        }

        if output_node_indices.is_empty() {
            unsafe {
                cuda::cuStreamDestroy_v2(stream);
            }
            for &p in &owned {
                let _ = mem::free(p);
            }
            return Err(anyhow!(
                "capture_graph: output_node_indices must not be empty"
            ));
        }
        let mut output_bufs: Vec<(DevicePtr, usize)> =
            Vec::with_capacity(output_node_indices.len());
        let mut output_shapes: Vec<Vec<usize>> = Vec::with_capacity(output_node_indices.len());
        for &oi in output_node_indices {
            if oi >= self.nodes.len() || self.nodes[oi].is_none() {
                unsafe {
                    cuda::cuStreamDestroy_v2(stream);
                }
                for &p in &owned {
                    let _ = mem::free(p);
                }
                return Err(anyhow!(
                    "capture_graph: output node {oi} is an Input or out of bounds"
                ));
            }
            let shape = concrete_shapes[oi].clone();
            let n_elems: usize = shape.iter().product();
            output_bufs.push((node_out_bufs[oi], n_elems));
            output_shapes.push(shape);
        }

        // ── Phase 5: allocate pinned (page-locked) host staging buffers ─────────
        // One buffer per input and per output.  cuMemcpyHtoD/DtoH over pinned
        // memory uses DMA directly, skipping the driver's internal pageable
        // staging bounce and achieving full PCIe bandwidth.
        let mut pinned_inputs: Vec<*mut f32> = Vec::with_capacity(input_bufs.len());
        for &(_, n_elems) in &input_bufs {
            match mem::alloc_host::<f32>(n_elems) {
                Ok(ptr) => pinned_inputs.push(ptr),
                Err(e) => {
                    unsafe {
                        cuda::cuStreamDestroy_v2(stream);
                    }
                    for &p in &owned {
                        let _ = mem::free(p);
                    }
                    for &p in &pinned_inputs {
                        unsafe {
                            let _ = mem::free_host(p);
                        }
                    }
                    return Err(e);
                }
            }
        }
        let mut pinned_outputs: Vec<*mut f32> = Vec::with_capacity(output_bufs.len());
        for &(_, n_elems) in &output_bufs {
            match mem::alloc_host::<f32>(n_elems) {
                Ok(ptr) => pinned_outputs.push(ptr),
                Err(e) => {
                    unsafe {
                        cuda::cuStreamDestroy_v2(stream);
                    }
                    for &p in &owned {
                        let _ = mem::free(p);
                    }
                    for &p in &pinned_inputs {
                        unsafe {
                            let _ = mem::free_host(p);
                        }
                    }
                    for &p in &pinned_outputs {
                        unsafe {
                            let _ = mem::free_host(p);
                        }
                    }
                    return Err(e);
                }
            }
        }

        Ok(CudaGraphModel {
            stream,
            graph_exec,
            input_bufs,
            output_bufs,
            output_shapes,
            pinned_inputs,
            pinned_outputs,
            _owned: owned,
        })
    }

    fn topo_sort(&self) -> Vec<usize> {
        let n = self.nodes.len();
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
                if in_deg[dep] == 0 {
                    stack.push(dep);
                }
            }
        }
        order
    }
}

/// Activation buffers retained from a `forward_train` call.
///
/// Implements `Drop` so device buffers are freed automatically.
#[cfg(feature = "training")]
pub struct ActivationCache {
    /// Per-node activation tensor, indexed by DAG node index. `None` for nodes with no cached
    /// activation (e.g. `Input` placeholders).
    pub tensors: Vec<Option<TensorRef>>,
}

#[cfg(feature = "training")]
impl Drop for ActivationCache {
    fn drop(&mut self) {
        for tr in self.tensors.iter().flatten() {
            if let Err(e) = mem::free(tr.ptr) {
                eprintln!("ActivationCache: failed to free buffer: {e}");
            }
        }
    }
}

// ---------------------------------------------------------------------------
// CudaGraphModel — captured inference graph for fixed-batch execution
// ---------------------------------------------------------------------------

/// A CUDA graph compiled from a [`LoadedModel`] for fixed-batch inference.
///
/// Created via [`LoadedModel::capture_graph`]. All device buffers are
/// pre-allocated; the kernel sequence is captured once and replayed on each
/// [`run`] call with a single `cuGraphLaunch` + `cuStreamSynchronize`.
pub struct CudaGraphModel {
    stream: cuda::CUstream,
    graph_exec: cuda::CUgraphExec,
    /// Pre-allocated input buffers (one per `Input` node, topological order).
    input_bufs: Vec<(DevicePtr, usize)>,
    /// Pre-allocated output buffers, in the order of `output_node_indices`.
    output_bufs: Vec<(DevicePtr, usize)>,
    output_shapes: Vec<Vec<usize>>,
    /// Page-locked (pinned) staging buffers for inputs — one per `input_bufs`
    /// entry. Pinned memory enables direct DMA, avoiding the driver's internal
    /// pageable staging bounce and achieving full PCIe bandwidth.
    pinned_inputs: Vec<*mut f32>,
    /// Page-locked (pinned) staging buffers for outputs — one per `output_bufs`.
    pinned_outputs: Vec<*mut f32>,
    /// All owned device allocations freed on drop.
    _owned: Vec<DevicePtr>,
}

// SAFETY: CudaGraphModel is used from a single thread. The raw CUDA handles
// and pinned pointers are not shared across threads.
unsafe impl Send for CudaGraphModel {}

impl CudaGraphModel {
    /// Concrete shapes of the output tensors, in the order of `output_node_indices`.
    pub fn output_shapes(&self) -> &[Vec<usize>] {
        &self.output_shapes
    }

    /// Like [`run`] but also returns GPU execution time in milliseconds.
    ///
    /// The GPU time is measured with CUDA events bracketing only `cuGraphLaunch`
    /// (pure kernel execution, excluding host↔device copies).
    /// The returned `f32` is milliseconds of GPU time for the whole batch.
    pub fn run_timed(&self, inputs: &[&[f32]]) -> Result<(Vec<Vec<f32>>, f32)> {
        if inputs.len() != self.input_bufs.len() {
            return Err(anyhow!(
                "CudaGraphModel::run_timed: expected {} inputs, got {}",
                self.input_bufs.len(),
                inputs.len()
            ));
        }
        for (i, (&(ptr, n_elems), &data)) in self.input_bufs.iter().zip(inputs.iter()).enumerate() {
            if data.len() != n_elems {
                return Err(anyhow!(
                    "CudaGraphModel::run_timed: input[{i}] has {} elements, expected {n_elems}",
                    data.len()
                ));
            }
            unsafe { mem::copy_h_to_d(ptr, data.as_ptr(), n_elems) }?;
        }

        // Create CUDA events to measure GPU kernel time.
        let mut ev_start = cuda::CUevent::default();
        let mut ev_end = cuda::CUevent::default();
        let cu_event_default = 0u32;
        unsafe {
            cuda::cuEventCreate(&mut ev_start, cu_event_default);
            cuda::cuEventCreate(&mut ev_end, cu_event_default);
            cuda::cuEventRecord(ev_start, self.stream);
        }

        let launch_s = unsafe { cuda::cuGraphLaunch(self.graph_exec, self.stream) };
        if launch_s != cuda::cudaError_enum_CUDA_SUCCESS {
            unsafe {
                cuda::cuEventDestroy_v2(ev_start);
                cuda::cuEventDestroy_v2(ev_end);
            }
            return Err(Error::from_cuda_error(launch_s).into());
        }

        unsafe {
            cuda::cuEventRecord(ev_end, self.stream);
        }

        let sync_s = unsafe { cuda::cuEventSynchronize(ev_end) };
        if sync_s != cuda::cudaError_enum_CUDA_SUCCESS {
            unsafe {
                cuda::cuEventDestroy_v2(ev_start);
                cuda::cuEventDestroy_v2(ev_end);
            }
            return Err(Error::from_cuda_error(sync_s).into());
        }

        let mut gpu_ms = 0.0f32;
        unsafe {
            cuda::cudaEventElapsedTime(&mut gpu_ms, ev_start, ev_end);
            cuda::cuEventDestroy_v2(ev_start);
            cuda::cuEventDestroy_v2(ev_end);
        }

        let mut result = Vec::with_capacity(self.output_bufs.len());
        for &(ptr, n_elems) in &self.output_bufs {
            let mut out = vec![0.0_f32; n_elems];
            unsafe { mem::copy_d_to_h(out.as_mut_ptr(), ptr, n_elems) }?;
            result.push(out);
        }
        Ok((result, gpu_ms))
    }

    /// Copy f32 `inputs` to device, replay the CUDA graph, copy f32 outputs to host.
    ///
    /// Returns one `Vec<f32>` per requested output node (same order as
    /// `output_node_indices` passed to [`LoadedModel::capture_graph`]).
    pub fn run(&self, inputs: &[&[f32]]) -> Result<Vec<Vec<f32>>> {
        if inputs.len() != self.input_bufs.len() {
            return Err(anyhow!(
                "CudaGraphModel::run: expected {} inputs, got {}",
                self.input_bufs.len(),
                inputs.len()
            ));
        }
        for (i, (&(ptr, n_elems), &data)) in self.input_bufs.iter().zip(inputs.iter()).enumerate() {
            if data.len() != n_elems {
                return Err(anyhow!(
                    "CudaGraphModel::run: input[{i}] has {} elements, expected {n_elems}",
                    data.len()
                ));
            }
            unsafe { mem::copy_h_to_d(ptr, data.as_ptr(), n_elems) }?;
        }

        let launch_s = unsafe { cuda::cuGraphLaunch(self.graph_exec, self.stream) };
        if launch_s != cuda::cudaError_enum_CUDA_SUCCESS {
            return Err(Error::from_cuda_error(launch_s).into());
        }
        let sync_s = unsafe { cuda::cuStreamSynchronize(self.stream) };
        if sync_s != cuda::cudaError_enum_CUDA_SUCCESS {
            return Err(Error::from_cuda_error(sync_s).into());
        }

        let mut result = Vec::with_capacity(self.output_bufs.len());
        for &(ptr, n_elems) in &self.output_bufs {
            let mut out = vec![0.0_f32; n_elems];
            unsafe { mem::copy_d_to_h(out.as_mut_ptr(), ptr, n_elems) }?;
            result.push(out);
        }
        Ok(result)
    }

    /// Mutable slice into the `i`-th pinned (page-locked) input staging buffer.
    ///
    /// Write your input data here before calling [`run_inplace`] /
    /// [`run_timed_inplace`] to avoid the intermediate CPU copy that
    /// [`run`] / [`run_timed`] perform when given a pageable `&[f32]`.
    ///
    /// # Safety
    /// The slice is valid until this `CudaGraphModel` is dropped.
    pub fn input_slice_mut(&mut self, i: usize) -> &mut [f32] {
        let (_, n_elems) = self.input_bufs[i];
        unsafe { std::slice::from_raw_parts_mut(self.pinned_inputs[i], n_elems) }
    }

    /// Immutable slice into the `i`-th pinned (page-locked) output staging
    /// buffer. Valid after [`run_inplace`] / [`run_timed_inplace`] returns.
    pub fn output_slice(&self, i: usize) -> &[f32] {
        let (_, n_elems) = self.output_bufs[i];
        unsafe { std::slice::from_raw_parts(self.pinned_outputs[i], n_elems) }
    }

    /// Copy pinned inputs → device, launch graph, copy device → pinned outputs.
    ///
    /// Callers must fill [`input_slice_mut`] before calling and read
    /// [`output_slice`] afterwards.  No heap allocations are performed.
    pub fn run_inplace(&self) -> Result<()> {
        for (i, &(dev_ptr, n_elems)) in self.input_bufs.iter().enumerate() {
            unsafe { mem::copy_h_to_d(dev_ptr, self.pinned_inputs[i], n_elems) }?;
        }
        let launch_s = unsafe { cuda::cuGraphLaunch(self.graph_exec, self.stream) };
        if launch_s != cuda::cudaError_enum_CUDA_SUCCESS {
            return Err(Error::from_cuda_error(launch_s).into());
        }
        let sync_s = unsafe { cuda::cuStreamSynchronize(self.stream) };
        if sync_s != cuda::cudaError_enum_CUDA_SUCCESS {
            return Err(Error::from_cuda_error(sync_s).into());
        }
        for (i, &(dev_ptr, n_elems)) in self.output_bufs.iter().enumerate() {
            unsafe { mem::copy_d_to_h(self.pinned_outputs[i], dev_ptr, n_elems) }?;
        }
        Ok(())
    }

    /// Like [`run_inplace`] but also returns GPU execution time in milliseconds.
    pub fn run_timed_inplace(&self) -> Result<f32> {
        for (i, &(dev_ptr, n_elems)) in self.input_bufs.iter().enumerate() {
            unsafe { mem::copy_h_to_d(dev_ptr, self.pinned_inputs[i], n_elems) }?;
        }

        let mut ev_start = cuda::CUevent::default();
        let mut ev_end = cuda::CUevent::default();
        unsafe {
            cuda::cuEventCreate(&mut ev_start, 0);
            cuda::cuEventCreate(&mut ev_end, 0);
            cuda::cuEventRecord(ev_start, self.stream);
        }

        let launch_s = unsafe { cuda::cuGraphLaunch(self.graph_exec, self.stream) };
        if launch_s != cuda::cudaError_enum_CUDA_SUCCESS {
            unsafe {
                cuda::cuEventDestroy_v2(ev_start);
                cuda::cuEventDestroy_v2(ev_end);
            }
            return Err(Error::from_cuda_error(launch_s).into());
        }

        unsafe {
            cuda::cuEventRecord(ev_end, self.stream);
        }

        let sync_s = unsafe { cuda::cuEventSynchronize(ev_end) };
        if sync_s != cuda::cudaError_enum_CUDA_SUCCESS {
            unsafe {
                cuda::cuEventDestroy_v2(ev_start);
                cuda::cuEventDestroy_v2(ev_end);
            }
            return Err(Error::from_cuda_error(sync_s).into());
        }

        let mut gpu_ms = 0.0f32;
        unsafe {
            cuda::cudaEventElapsedTime(&mut gpu_ms, ev_start, ev_end);
            cuda::cuEventDestroy_v2(ev_start);
            cuda::cuEventDestroy_v2(ev_end);
        }

        for (i, &(dev_ptr, n_elems)) in self.output_bufs.iter().enumerate() {
            unsafe { mem::copy_d_to_h(self.pinned_outputs[i], dev_ptr, n_elems) }?;
        }
        Ok(gpu_ms)
    }
}

impl Drop for CudaGraphModel {
    fn drop(&mut self) {
        unsafe {
            cuda::cuGraphExecDestroy(self.graph_exec);
            cuda::cuStreamDestroy_v2(self.stream);
        }
        for &ptr in &self._owned {
            if ptr != 0 && mem::free(ptr).is_err() {
                eprintln!("CudaGraphModel::drop: failed to free buffer at {ptr:#x}");
            }
        }
        for &ptr in &self.pinned_inputs {
            if !ptr.is_null()
                && let Err(e) = unsafe { mem::free_host(ptr) }
            {
                eprintln!("CudaGraphModel::drop: failed to free pinned input: {e}");
            }
        }
        for &ptr in &self.pinned_outputs {
            if !ptr.is_null()
                && let Err(e) = unsafe { mem::free_host(ptr) }
            {
                eprintln!("CudaGraphModel::drop: failed to free pinned output: {e}");
            }
        }
    }
}

/// A pre-compiled `adamw_step` kernel ready for use in `LoadedModel::adamw_step`.
///
/// Create via:
/// ```ignore
/// let ptx = std::fs::read(compile_kernel(&AdamwStep::new(1024), &target, true)?)?;
/// let kernel = AdamwKernel::from_ptx(&ptx)?;
/// ```
#[cfg(feature = "training")]
pub struct AdamwKernel {
    pub(crate) program: CudaProgram<'static, ErasedKernel>,
}

#[cfg(feature = "training")]
impl AdamwKernel {
    /// Loads a pre-compiled `adamw_step` kernel from raw PTX bytes.
    pub fn from_ptx(ptx: &[u8]) -> Result<Self> {
        let program = CudaProgram::<ErasedKernel>::try_from_ptx(ptx)?;
        Ok(Self { program })
    }
}
