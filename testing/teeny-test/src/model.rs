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

//! A device-generic graph executor: runs a lowered `teeny-core` [`Graph`] on whichever backend
//! [`teeny_runtime`] was built for (`cuda`, or `riscv` under `qemu-riscv64`), so the same model
//! can be checked for numerical agreement across backends.
//!
//! Unlike `teeny_cuda::model::CudaModel` this is deliberately simple -- every node's output
//! buffer is allocated once at load time, kernels are launched one at a time through
//! [`teeny_core::device::Device::launch`], and there is no graph capture or optimisation -- since
//! its purpose is cross-backend correctness rather than speed.

use std::collections::HashMap;
use std::rc::Rc;
use std::sync::Arc;
use std::time::Instant;

use anyhow::{Context, Result, anyhow, ensure};
use teeny_core::device::Device as _;
use teeny_core::device::buffer::Buffer as _;
use teeny_core::device::program::{ArgVisitor, Kernel, KernelArgs};
use teeny_core::graph::{DtypeRepr, Graph, Shape};
use teeny_core::model::{Lowering, LoweringMode, RawPtr, RuntimeOp};
use teeny_runtime::{Buffer, Device, Program, Target};

use crate::ExecKernel;

/// The [`teeny_runtime`] backend this module runs on, re-exported so callers can open a
/// device and resolve a target without depending on teeny-runtime directly.
pub use teeny_runtime as runtime;

/// One recorded kernel argument -- see [`PackedArgs`].
#[derive(Debug, Clone, Copy)]
enum Arg {
    Ptr(RawPtr),
    Bool(bool),
    I8(i8),
    I16(i16),
    I32(i32),
    I64(i64),
    U8(u8),
    U16(u16),
    U32(u32),
    U64(u64),
    F32(f32),
    F64(f64),
}

/// A kernel argument list recorded from [`RuntimeOp::pack_args`] (which emits arguments through
/// a `dyn ArgVisitor`) and replayed into a backend's [`teeny_core::device::Device::launch`]
/// (which takes them as [`KernelArgs`]).
#[derive(Debug, Clone, Default)]
pub struct PackedArgs(Vec<Arg>);

impl ArgVisitor for PackedArgs {
    fn visit_ptr(&mut self, ptr: *mut core::ffi::c_void) {
        self.0.push(Arg::Ptr(ptr));
    }
    fn visit_bool(&mut self, val: bool) {
        self.0.push(Arg::Bool(val));
    }
    fn visit_i8(&mut self, val: i8) {
        self.0.push(Arg::I8(val));
    }
    fn visit_i16(&mut self, val: i16) {
        self.0.push(Arg::I16(val));
    }
    fn visit_i32(&mut self, val: i32) {
        self.0.push(Arg::I32(val));
    }
    fn visit_i64(&mut self, val: i64) {
        self.0.push(Arg::I64(val));
    }
    fn visit_u8(&mut self, val: u8) {
        self.0.push(Arg::U8(val));
    }
    fn visit_u16(&mut self, val: u16) {
        self.0.push(Arg::U16(val));
    }
    fn visit_u32(&mut self, val: u32) {
        self.0.push(Arg::U32(val));
    }
    fn visit_u64(&mut self, val: u64) {
        self.0.push(Arg::U64(val));
    }
    fn visit_f32(&mut self, val: f32) {
        self.0.push(Arg::F32(val));
    }
    fn visit_f64(&mut self, val: f64) {
        self.0.push(Arg::F64(val));
    }
}

impl KernelArgs for PackedArgs {
    fn visit_args<V: ArgVisitor>(&self, visitor: &mut V) {
        for arg in &self.0 {
            match *arg {
                Arg::Ptr(v) => visitor.visit_ptr(v),
                Arg::Bool(v) => visitor.visit_bool(v),
                Arg::I8(v) => visitor.visit_i8(v),
                Arg::I16(v) => visitor.visit_i16(v),
                Arg::I32(v) => visitor.visit_i32(v),
                Arg::I64(v) => visitor.visit_i64(v),
                Arg::U8(v) => visitor.visit_u8(v),
                Arg::U16(v) => visitor.visit_u16(v),
                Arg::U32(v) => visitor.visit_u32(v),
                Arg::U64(v) => visitor.visit_u64(v),
                Arg::F32(v) => visitor.visit_f32(v),
                Arg::F64(v) => visitor.visit_f64(v),
            }
        }
    }
}

/// The type-erased kernel every [`RuntimeModel`] program is loaded as: kernels are compiled via
/// [`ExecKernel`], and only launched through this type, whose arguments are [`PackedArgs`].
pub struct PackedKernel;

impl Kernel for PackedKernel {
    type Args<'a> = PackedArgs;

    fn name(&self) -> &str {
        "packed"
    }

    fn source(&self) -> &str {
        ""
    }

    fn kernel_source(&self) -> &str {
        ""
    }

    fn entry_point_source(&self) -> &str {
        ""
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

/// Allocates a zero-filled buffer of `n_elems` elements of `dtype`. Storage is always `f32`
/// words (rounded up), since kernels only ever see the raw pointer.
fn zeroed_buffer<'a>(
    device: &Device<'a>,
    n_elems: usize,
    dtype: DtypeRepr,
) -> Result<Buffer<'a, f32>> {
    let words = (n_elems * dtype_bytes(dtype)).div_ceil(4).max(1);
    let mut buf = device.buffer::<f32>(words)?;
    buf.to_device(&vec![0.0f32; words])?;
    Ok(buf)
}

fn ptr(buf: &Buffer<'_, f32>) -> RawPtr {
    buf.as_device_ptr() as RawPtr
}

/// A padded output buffer, for ops whose kernel writes rows with a stride wider than the
/// tight output layout (see [`RuntimeOp::forward_output_row_stride`]).
struct Padded<'a> {
    buf: Buffer<'a, f32>,
    natural_stride: usize,
    required_stride: usize,
    n_rows: usize,
}

struct Node<'a> {
    /// `None` for `Input` placeholder nodes.
    op: Option<NodeOp<'a>>,
    output_shape: Vec<usize>,
    output: Buffer<'a, f32>,
}

struct NodeOp<'a> {
    program: Rc<Program<'static, PackedKernel>>,
    runtime_op: Arc<dyn RuntimeOp>,
    params: Vec<Buffer<'a, f32>>,
    param_shapes: Vec<Vec<usize>>,
    padded: Option<Padded<'a>>,
}

/// A lowered graph compiled for, and loaded onto, the [`teeny_runtime`] backend, with every
/// buffer pre-allocated for a fixed `batch_size`.
pub struct RuntimeModel<'a> {
    nodes: Vec<Node<'a>>,
    parents: Vec<Vec<usize>>,
    topo: Vec<usize>,
    names: HashMap<usize, String>,
    trace: bool,
}

impl<'a> RuntimeModel<'a> {
    /// Lowers `graph`, compiles every op's forward kernel for `target`, and allocates all
    /// parameter and activation buffers on `device` for `batch_size`.
    pub fn compile<'l, L: Lowering<'l>>(
        device: &Device<'a>,
        graph: &Graph,
        lowering: &L,
        mode: LoweringMode,
        target: &Target,
        batch_size: usize,
        force: bool,
    ) -> Result<Self> {
        let (op_dag, graph_to_dag) = lowering.lower_with_mapping(graph, mode)?;
        let n = op_dag.len();
        let topo = op_dag.topological_sort();
        let parents: Vec<Vec<usize>> = (0..n).map(|i| op_dag.node(i).parents.clone()).collect();

        let mut names: HashMap<usize, String> = graph
            .names
            .iter()
            .filter_map(|(&graph_idx, name)| Some((*graph_to_dag.get(graph_idx)?, name.clone())))
            .collect();
        for (dag_idx, name) in lowering.extra_dag_names(graph, &graph_to_dag) {
            names.entry(dag_idx).or_insert(name);
        }

        // Compiled artifact path -> loaded program, so identical kernels load once.
        let mut programs: HashMap<String, Rc<Program<'static, PackedKernel>>> = HashMap::new();
        let mut shapes: Vec<Vec<usize>> = vec![Vec::new(); n];
        let mut nodes: Vec<Option<Node<'a>>> = (0..n).map(|_| None).collect();

        for &idx in &topo {
            let op = op_dag.node(idx).value.as_ref();
            let raw_output_shape = resolve_shape(op.output_shape(), batch_size);
            let dtype = op.output_dtype();

            let Some(runtime_op) = op.runtime_op().filter(|_| !op.is_input()) else {
                let output = zeroed_buffer(device, raw_output_shape.iter().product(), dtype)?;
                shapes[idx] = raw_output_shape.clone();
                nodes[idx] = Some(Node {
                    op: None,
                    output_shape: raw_output_shape,
                    output,
                });
                continue;
            };

            let parent_shapes: Vec<&[usize]> =
                parents[idx].iter().map(|&p| shapes[p].as_slice()).collect();
            let output_shape =
                runtime_op.compute_concrete_output_shape(&parent_shapes, &raw_output_shape);

            ensure!(
                !op.forward_kernel_source().is_empty(),
                "no forward kernel source for op {}",
                op.name()
            );
            let path = teeny_runtime::compile_kernel(&ExecKernel(op), target, force, false)
                .with_context(|| format!("compiling node {idx} ({})", op.name()))?;
            let program = match programs.get(&path) {
                Some(p) => Rc::clone(p),
                None => {
                    let p = Rc::new(
                        teeny_runtime::load_program::<PackedKernel>(&path)
                            .with_context(|| format!("loading {path}"))?,
                    );
                    programs.insert(path, Rc::clone(&p));
                    p
                }
            };

            let param_shapes = runtime_op.param_shapes(&parent_shapes, &output_shape);
            let mut params = Vec::with_capacity(param_shapes.len());
            for (pi, ps) in param_shapes.iter().enumerate() {
                let mut buf = zeroed_buffer(device, ps.iter().product(), dtype)?;
                if let Some(bytes) = runtime_op.param_init_data(pi) {
                    let words: Vec<f32> = bytes
                        .chunks(4)
                        .map(|c| {
                            let mut w = [0u8; 4];
                            w[..c.len()].copy_from_slice(c);
                            f32::from_ne_bytes(w)
                        })
                        .collect();
                    buf.to_device(&words)?;
                }
                params.push(buf);
            }

            let n_elems: usize = output_shape.iter().product();
            let output = zeroed_buffer(device, n_elems, dtype)?;

            let natural_stride = runtime_op.forward_output_row_elems(&output_shape);
            let required_stride = runtime_op.forward_output_row_stride(&output_shape);
            let padded = if required_stride > natural_stride {
                let n_rows = n_elems / natural_stride.max(1);
                Some(Padded {
                    buf: zeroed_buffer(device, n_rows * required_stride, dtype)?,
                    natural_stride,
                    required_stride,
                    n_rows,
                })
            } else {
                None
            };

            shapes[idx] = output_shape.clone();
            nodes[idx] = Some(Node {
                op: Some(NodeOp {
                    program,
                    runtime_op,
                    params,
                    param_shapes,
                    padded,
                }),
                output_shape,
                output,
            });
        }

        Ok(Self {
            nodes: nodes
                .into_iter()
                .map(|n| n.expect("every node visited"))
                .collect(),
            parents,
            topo,
            names,
            trace: false,
        })
    }

    /// Prints each node's launch time to stderr as it runs -- useful under `qemu-riscv64`,
    /// where a single forward pass can take minutes.
    pub fn set_trace(&mut self, trace: bool) {
        self.trace = trace;
    }

    /// Every named parameter slot as `(safetensors key, node_idx, param_idx)` -- same naming as
    /// `teeny_cuda::model::LoadedModel::param_info_named`.
    pub fn param_info_named(&self) -> impl Iterator<Item = (String, usize, usize)> + '_ {
        self.nodes
            .iter()
            .enumerate()
            .flat_map(move |(node_idx, node)| {
                let entries: Vec<(String, usize, usize)> =
                    match (&node.op, self.names.get(&node_idx)) {
                        (Some(op), Some(node_name)) => {
                            let slots = op.runtime_op.param_names();
                            (0..op.param_shapes.len())
                                .filter_map(|pi| {
                                    let slot = slots.get(pi).copied().unwrap_or("");
                                    (!slot.is_empty())
                                        .then(|| (format!("{node_name}.{slot}"), node_idx, pi))
                                })
                                .collect()
                        }
                        _ => Vec::new(),
                    };
                entries
            })
    }

    /// Uploads `data` into parameter slot `param_idx` of node `node_idx`.
    pub fn load_param_f32(
        &mut self,
        node_idx: usize,
        param_idx: usize,
        data: &[f32],
    ) -> Result<()> {
        let op = self.nodes[node_idx]
            .op
            .as_mut()
            .ok_or_else(|| anyhow!("node {node_idx} is an Input placeholder"))?;
        let n_elems: usize = op
            .param_shapes
            .get(param_idx)
            .ok_or_else(|| anyhow!("node {node_idx} has no param at index {param_idx}"))?
            .iter()
            .product();
        ensure!(
            data.len() == n_elems,
            "node {node_idx} param {param_idx}: expected {n_elems} values, got {}",
            data.len()
        );
        op.params[param_idx].to_device(data)?;
        Ok(())
    }

    /// The recorded dotted name of node `idx`, if any.
    pub fn node_name(&self, idx: usize) -> Option<&str> {
        self.names.get(&idx).map(String::as_str)
    }

    /// Number of DAG nodes.
    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    /// Whether the model has no nodes.
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    /// Node `idx`'s concrete output shape.
    pub fn output_shape(&self, idx: usize) -> &[usize] {
        &self.nodes[idx].output_shape
    }

    /// Terminal (no-consumer) node indices, sorted by output element count (largest last).
    pub fn terminal_node_indices_sorted_by_size(&self) -> Vec<usize> {
        let mut has_child = vec![false; self.nodes.len()];
        for ps in &self.parents {
            for &p in ps {
                has_child[p] = true;
            }
        }
        let mut terminals: Vec<usize> = (0..self.nodes.len()).filter(|&i| !has_child[i]).collect();
        terminals.sort_by_key(|&i| self.nodes[i].output_shape.iter().product::<usize>());
        terminals
    }

    /// Copies node `idx`'s most recent output back to the host (assumes an `f32` output).
    pub fn read_output_f32(&self, idx: usize) -> Result<Vec<f32>> {
        let node = &self.nodes[idx];
        let mut out = vec![0.0f32; node.output_shape.iter().product()];
        node.output.to_host(&mut out)?;
        Ok(out)
    }

    /// Runs one forward pass. `inputs` are matched to `Input` nodes in topological order;
    /// returns the outputs of the nodes in `outputs`, in that order.
    pub fn run(
        &mut self,
        device: &Device<'a>,
        inputs: &[&[f32]],
        outputs: &[usize],
    ) -> Result<Vec<Vec<f32>>> {
        let mut input_cursor = 0usize;
        for &idx in &self.topo {
            if self.nodes[idx].op.is_none() {
                let data = inputs
                    .get(input_cursor)
                    .ok_or_else(|| anyhow!("too few inputs: needed >{input_cursor}"))?;
                self.nodes[idx].output.to_device(data)?;
                input_cursor += 1;
            }
        }

        let total = self.topo.len();
        for step in 0..total {
            let idx = self.topo[step];
            let started = Instant::now();
            let Some(tight) = self.launch_node(device, idx)? else {
                continue;
            };
            if let Some(tight) = tight {
                self.nodes[idx].output.to_device(&tight)?;
            }
            if self.trace {
                eprintln!(
                    "[{}/{total}] node {idx} {} {:?}: {:.2?}",
                    step + 1,
                    self.node_name(idx).unwrap_or("unnamed"),
                    self.nodes[idx].output_shape,
                    started.elapsed()
                );
            }
        }

        outputs
            .iter()
            .map(|&idx| self.read_output_f32(idx))
            .collect()
    }

    /// Launches node `idx`'s kernel(s). Returns `None` for an `Input` node, and otherwise the
    /// de-padded output rows when the kernel wrote to a padded buffer (for the caller to copy
    /// into the node's output buffer).
    fn launch_node(&self, device: &Device<'a>, idx: usize) -> Result<Option<Option<Vec<f32>>>> {
        let node = &self.nodes[idx];
        let Some(op) = node.op.as_ref() else {
            return Ok(None);
        };

        let act_inputs: Vec<(RawPtr, &[usize])> = self.parents[idx]
            .iter()
            .map(|&p| {
                (
                    ptr(&self.nodes[p].output),
                    self.nodes[p].output_shape.as_slice(),
                )
            })
            .collect();
        let input_shapes: Vec<&[usize]> = act_inputs.iter().map(|(_, s)| *s).collect();
        let param_ptrs: Vec<RawPtr> = op.params.iter().map(ptr).collect();
        let (out_ptr, row_stride) = match &op.padded {
            Some(p) => (ptr(&p.buf), p.required_stride),
            None => (
                ptr(&node.output),
                op.runtime_op.forward_output_row_stride(&node.output_shape),
            ),
        };

        let n_launches = op.runtime_op.n_launches();
        for launch_idx in 0..n_launches {
            let mut args = PackedArgs::default();
            let grid = if n_launches == 1 {
                op.runtime_op.pack_args(
                    &act_inputs,
                    &param_ptrs,
                    out_ptr,
                    &node.output_shape,
                    row_stride as i32,
                    &mut args,
                );
                op.runtime_op.grid(&node.output_shape)
            } else {
                op.runtime_op.pack_args_for_launch(
                    launch_idx,
                    &act_inputs,
                    &param_ptrs,
                    out_ptr,
                    &node.output_shape,
                    row_stride as i32,
                    &mut args,
                );
                op.runtime_op
                    .grid_for_launch(launch_idx, &input_shapes, &node.output_shape)
            };
            let cfg = teeny_runtime::launch_config_grid(grid, &op.program);
            device.launch(&op.program, &cfg, args).with_context(|| {
                format!(
                    "launch {launch_idx}/{n_launches} failed at node {idx} ({}, grid={grid:?}, \
                         shape={:?})",
                    self.node_name(idx).unwrap_or("unnamed"),
                    node.output_shape
                )
            })?;
        }

        let Some(p) = &op.padded else {
            return Ok(Some(None));
        };
        let mut padded = vec![0.0f32; p.n_rows * p.required_stride];
        p.buf.to_host(&mut padded)?;
        Ok(Some(Some(
            padded
                .chunks(p.required_stride)
                .flat_map(|row| &row[..p.natural_stride])
                .copied()
                .collect(),
        )))
    }
}
