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

//! Test fixtures: graphs of ops with hand-written tile specs, and the
//! device + shared memory hierarchy.

use teeny_core::device::hardware::MemoryLevelKind;
use teeny_core::graph::{DtypeRepr, Shape};
use teeny_core::model::{
    ExecutableOp, GridSpec, KernelTileSpec, TensorTileSpec, TileAxisBinding, TileWindow,
};

use super::{Hierarchy, LevelInfo, Owner};
use crate::graph::optimizer::anduin::tile_graph::{
    Edge, EdgeId, Node, NodeId, NodeKind, OpMeta, TileGraph,
};
use crate::graph::tile_resolve::{ConstLookup, NoConsts};

/// Device memory and shared memory only: Triton, not the kernel writer,
/// decides what lives in registers, so shared memory is the compute level H.
/// Capacities are Orin Nano's (`teeny_test::hardware_profile::orin_nano`).
pub(super) fn device_and_shared() -> Hierarchy {
    Hierarchy {
        levels: vec![
            LevelInfo {
                kind: MemoryLevelKind::DeviceMemory,
                owner: Owner::Device,
                capacity_per_owner: 8 * 1024 * 1024 * 1024,
                capacity_per_cu: u64::MAX,
                bandwidth: 68.3e9,
                read_txn_bytes: 128,
                write_txn_bytes: 32,
                max_fanout: u32::MAX,
                fixed_fanout: false,
            },
            LevelInfo {
                kind: MemoryLevelKind::SharedMemory,
                owner: Owner::Block,
                capacity_per_owner: 96 * 1024,
                capacity_per_cu: 96 * 1024,
                bandwidth: f64::INFINITY,
                read_txn_bytes: 1,
                write_txn_bytes: 1,
                max_fanout: 1024,
                fixed_fanout: true,
            },
        ],
        compute_units: 8,
        max_units_per_cu: 4,
        simt_width: 32,
        flops_per_s: 1.0e12,
        launch_overhead_s: 5.0e-6,
    }
}

/// A binding of `dims` to block const `block` and extent `extent`.
pub(super) fn bind(dims: &[usize], block: &'static str, extent: &'static str) -> TileAxisBinding {
    TileAxisBinding {
        dims: Vec::leak(dims.to_vec()),
        block_const: block,
        extent_param: extent,
        window: None,
        divide_by: None,
    }
}

/// [`bind`] read through a window of the output axis `output`, with stride,
/// pad and kernel size consts named as given.
pub(super) fn windowed(
    dim: usize,
    extent: &'static str,
    output: &'static str,
    stride: &'static str,
    pad: &'static str,
    kernel: &'static str,
) -> TileAxisBinding {
    TileAxisBinding {
        window: Some(TileWindow {
            output_extent_param: output,
            stride_const: stride,
            pad_const: Some(pad),
            kernel_size_const: kernel,
        }),
        ..bind(&[dim], "1", extent)
    }
}

pub(super) fn tensor(rank: usize, axes: &[TileAxisBinding]) -> TensorTileSpec {
    TensorTileSpec {
        param: "t",
        rank,
        axes: Vec::leak(axes.to_vec()),
        reduction_axis: None,
        untiled_dims: &[],
    }
}

pub(super) fn kernel(inputs: &[TensorTileSpec], output: TensorTileSpec) -> KernelTileSpec {
    KernelTileSpec {
        inputs: Vec::leak(inputs.to_vec()),
        outputs: Vec::leak(vec![output]),
        loop_spec: None,
    }
}

/// An elementwise kernel over every dim of a rank-`rank` tensor, with `n`
/// operands: the `tile_spec(rank)` the macro emits for a flat kernel.
pub(super) fn flat(rank: usize, n: usize) -> KernelTileSpec {
    let dims: Vec<usize> = (0..rank).collect();
    let t = tensor(rank, &[bind(&dims, "BLOCK", "N")]);
    kernel(&vec![t; n], t)
}

/// An op with an optional tile spec (no spec = opaque), marked a rider or not.
struct SpecOp {
    name: String,
    shape: Shape,
    spec: Option<KernelTileSpec>,
    rider: bool,
}

impl ExecutableOp for SpecOp {
    fn name(&self) -> &str {
        &self.name
    }
    fn forward_kernel_source(&self) -> &str {
        ""
    }
    fn forward_kernel_entry_point(&self) -> &str {
        ""
    }
    fn output_shape(&self) -> &Shape {
        &self.shape
    }
    fn output_dtype(&self) -> DtypeRepr {
        DtypeRepr::F32
    }
    fn tile_spec(&self) -> Option<KernelTileSpec> {
        self.spec
    }
    fn grid_spec(&self) -> Option<GridSpec> {
        self.spec.map(|_| GridSpec { axes: &[] })
    }
    fn pointwise_fuse_block_size(&self) -> Option<i32> {
        self.rider.then_some(1024)
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

/// Builds a [`TileGraph`] node by node, in topological order (which becomes
/// its priority order). Every op with no consumer feeds an `Output` node, as
/// `TileGraph::from_dag` does, once [`Builder::build`] runs.
pub(super) struct Builder {
    g: TileGraph,
}

impl Default for Builder {
    fn default() -> Self {
        Builder {
            g: TileGraph {
                nodes: vec![],
                edges: vec![],
                order: vec![],
                pos: vec![],
            },
        }
    }
}

impl Builder {
    fn push(&mut self, kind: NodeKind, name: &str, shape: &[u64], meta: Option<OpMeta>) -> NodeId {
        let id = NodeId(self.g.nodes.len());
        self.g.nodes.push(Node {
            id,
            kind,
            name: name.to_string(),
            in_edges: vec![],
            out_edges: vec![],
            shape: shape.to_vec(),
            dtype: DtypeRepr::F32,
            meta,
        });
        self.g.order.push(id);
        self.g.pos.push(id.0);
        id
    }

    fn connect(&mut self, src: NodeId, dst: NodeId, port: usize) {
        let id = EdgeId(self.g.edges.len());
        self.g.edges.push(Edge {
            id,
            src,
            dst,
            dst_port: port,
        });
        self.g.nodes[src.0].out_edges.push(id);
        self.g.nodes[dst.0].in_edges.push(id);
    }

    pub fn input(&mut self, name: &str, shape: &[u64]) -> NodeId {
        self.push(NodeKind::Placeholder, name, shape, None)
    }

    /// An op reading `inputs` (operand `p` from `inputs[p]`), with `spec`
    /// (None = opaque) and the window consts `consts`.
    pub fn op_with(
        &mut self,
        name: &str,
        shape: &[u64],
        spec: Option<KernelTileSpec>,
        rider: bool,
        consts: Box<dyn ConstLookup>,
        inputs: &[NodeId],
    ) -> NodeId {
        let op = SpecOp {
            name: name.to_string(),
            shape: shape.iter().map(|&d| Some(d as usize)).collect(),
            spec,
            rider,
        };
        let meta = OpMeta::new(Box::new(op), consts, 1.0);
        let id = self.push(NodeKind::Op, name, shape, meta);
        for (p, &src) in inputs.iter().enumerate() {
            self.connect(src, id, p);
        }
        id
    }

    pub fn op(
        &mut self,
        name: &str,
        shape: &[u64],
        spec: KernelTileSpec,
        inputs: &[NodeId],
    ) -> NodeId {
        self.op_with(name, shape, Some(spec), false, Box::new(NoConsts), inputs)
    }

    pub fn rider(
        &mut self,
        name: &str,
        shape: &[u64],
        spec: KernelTileSpec,
        inputs: &[NodeId],
    ) -> NodeId {
        self.op_with(name, shape, Some(spec), true, Box::new(NoConsts), inputs)
    }

    /// Marks `n` a model output, as well as whatever else consumes it.
    pub fn output(&mut self, n: NodeId) {
        let out = self.push(NodeKind::Output, "out", &[], None);
        self.connect(n, out, 0);
    }

    pub fn build(mut self) -> TileGraph {
        let sinks: Vec<NodeId> = self
            .g
            .nodes
            .iter()
            .filter(|n| n.kind == NodeKind::Op && n.out_edges.is_empty())
            .map(|n| n.id)
            .collect();
        for n in sinks {
            self.output(n);
        }
        self.g
    }
}

/// ANALYSIS.md §12.7 D1 with the specs `TritonLowering` gives the softmax
/// kernels: `x -> max -> sub(x, .) -> exp -> sum -> div(exp, .)` over
/// `[rows, cols]`.
/// - `sub` and `div` are the row-broadcast kernels: one row per unit (block
///   literal 1), columns by `BLOCK_C`, and their `[rows, 1]` second operand
///   read over the row axis alone.
/// - `max` and `sum` produce one row per unit and read the whole row: `C`
///   names no output dim, so it stays `Full`.
/// - `exp` is a flat rider.
pub(super) struct Softmax {
    pub g: TileGraph,
    pub x: NodeId,
    pub max: NodeId,
    pub sub: NodeId,
    pub exp: NodeId,
    pub sum: NodeId,
    pub div: NodeId,
}

pub(super) fn row_tile() -> TensorTileSpec {
    tensor(2, &[bind(&[0], "1", "R"), bind(&[1], "BLOCK_C", "C")])
}

/// `[rows, 1]` broadcast over the columns: only the row axis is bound.
pub(super) fn row_vector() -> TensorTileSpec {
    tensor(2, &[bind(&[0], "1", "R")])
}

/// A keepdims reduction of the last axis, one row per unit.
pub(super) fn row_reduce() -> KernelTileSpec {
    kernel(
        &[tensor(
            2,
            &[bind(&[0], "1", "R"), bind(&[1], "BLOCK_C", "C")],
        )],
        tensor(2, &[bind(&[0], "1", "R")]),
    )
}

/// `a op b` with `b` a `[rows, 1]` row vector.
pub(super) fn row_broadcast() -> KernelTileSpec {
    kernel(&[row_tile(), row_vector()], row_tile())
}

pub(super) fn softmax(rows: u64, cols: u64) -> Softmax {
    let mut b = Builder::default();
    let (full, reduced) = ([rows, cols], [rows, 1]);
    let x = b.input("x", &full);
    let max = b.op("max", &reduced, row_reduce(), &[x]);
    let sub = b.op("sub", &full, row_broadcast(), &[x, max]);
    let exp = b.rider("exp", &full, flat(2, 1), &[sub]);
    let sum = b.op("sum", &reduced, row_reduce(), &[exp]);
    let div = b.op("div", &full, row_broadcast(), &[exp, sum]);
    Softmax {
        g: b.build(),
        x,
        max,
        sub,
        exp,
        sum,
        div,
    }
}
