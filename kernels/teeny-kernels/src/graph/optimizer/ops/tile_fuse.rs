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

//! [`TileFuse`] — a unary pointwise chain fused with a fan-in binary tail.
//!
//! This is the tile-DSL-scheduler slice of teenygrad-3w0 (phase 2 /
//! teenygrad-3w0.3): the first fusion shape `PointwiseFuse` structurally
//! cannot reach, because `Anduin::fuse_pointwise_chain_pass` only ever
//! considers nodes with `inputs.len() == 1`. A binary op like `Op::Add` with
//! two distinct graph-node inputs — e.g. the residual pattern `relu(x) + z`
//! — never even reaches the eligibility checks today. `TileFuse` covers
//! exactly that: one input runs through a `PointwiseFuse`-style unary chain,
//! the other is a second, unchained tensor, and a binary tail op combines them
//! into a single generated kernel — teenygrad-1bf's fusion case 2 (multi-input
//! fan-in), which needed a real DAG region rather than `PointwiseFuse`'s
//! linear `Vec<Op>` chain.

use std::any::Any;
use std::sync::Arc;

use teeny_core::device::program::ArgVisitor;
use teeny_core::graph::{CustomOp, DtypeRepr, Op, Shape};
use teeny_core::model::{RawPtr, RuntimeOp};
use teeny_triton::PointwiseFuseProbe;

use crate::graph::optimizer::ops::pointwise_fuse::{dtype_name, member_kernel};
use crate::nn::tensor::elemwise_add::ElemwiseAddForward;

/// Binary ops `TileFuse` can use as its fan-in tail. Only `Add` for this
/// first cut — it's the canonical residual-connection shape and its kernel
/// (`elemwise_add_forward`) already has the plain `(a_ptr, b_ptr, out_ptr,
/// n)` ABI this composition needs; other binary ops can extend this list
/// once they're confirmed to share that shape.
pub fn is_tile_fuse_tail(op: &Op) -> bool {
    matches!(op, Op::Add)
}

/// A unary pointwise chain (`branch`) fused with a second, unchained input
/// via a binary elementwise `tail` op — e.g. `y = relu(x) + z`.
///
/// Produced by Anduin; lowered by concatenating the branch's member bodies
/// (scratch-chained, same scheme as [`super::PointwiseFuse`]) followed by the
/// tail kernel's body, and synthesizing an entry point that threads
/// `x_ptr` through the branch into the tail alongside the untouched `z_ptr`.
#[derive(Debug, Clone)]
pub struct TileFuse {
    /// Unary chain applied to the primary input (len >= 1).
    pub branch: Vec<Op>,
    /// Binary op combining the branch's result with the second input.
    /// Must satisfy [`is_tile_fuse_tail`].
    pub tail: Op,
    /// Element dtype shared by every member and the tail (must be float/num,
    /// whatever `PointwiseFuse` and the tail kernel both support).
    pub dtype: DtypeRepr,
    /// Shared CTA probe for the branch (all members compatible).
    pub probe: PointwiseFuseProbe,
}

impl TileFuse {
    /// Builds a fuse of a non-empty `branch` combined via `tail` (must pass
    /// [`is_tile_fuse_tail`]).
    pub fn new(branch: Vec<Op>, tail: Op, dtype: DtypeRepr, probe: PointwiseFuseProbe) -> Self {
        debug_assert!(!branch.is_empty());
        debug_assert!(is_tile_fuse_tail(&tail));
        Self {
            branch,
            tail,
            dtype,
            probe,
        }
    }
}

impl CustomOp for TileFuse {
    fn name(&self) -> &str {
        "tile_fuse"
    }

    fn infer_output_shape(&self, input_shapes: &[&Shape]) -> Shape {
        // Same "approximate as first-input shape" convention the rest of the
        // graph uses for binary/variadic elementwise ops today (see
        // `infer_output_shape`'s `Op::Mul | Op::Sub | ...` arm) — a real
        // broadcast-aware shape isn't computed anywhere in this codebase yet.
        input_shapes[0].clone()
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn lower(&self) -> Option<(String, String, String, Arc<dyn RuntimeOp>)> {
        match lower_tile_fuse(&self.branch, &self.tail, self.dtype, self.probe) {
            Ok(v) => Some(v),
            Err(e) => panic!("TileFuse::lower failed: {e}"),
        }
    }

    fn pointwise_fuse_block_size(&self) -> Option<i32> {
        // TileFuse's output is bool-safe/float-safe like PointwiseFuse's, so
        // it can itself feed a further chain — not exercised by the current
        // Anduin pass, but keeps the probe contract consistent.
        Some(self.probe.block_size)
    }
}

/// Runtime ABI: `x_ptr, z_ptr, y_ptr, scratch0 [, scratch1, …], n_elements`.
struct TileFuseRuntimeOp {
    block_size: i32,
    n_scratch: usize,
}

impl RuntimeOp for TileFuseRuntimeOp {
    fn n_activation_inputs(&self) -> usize {
        2
    }

    fn param_shapes(&self, _input_shapes: &[&[usize]], output_shape: &[usize]) -> Vec<Vec<usize>> {
        let n: usize = output_shape.iter().product();
        vec![vec![n]; self.n_scratch]
    }

    fn param_names(&self) -> &'static [&'static str] {
        match self.n_scratch {
            1 => &["scratch0"],
            2 => &["scratch0", "scratch1"],
            _ => &["scratch0", "scratch1", "scratch2"],
        }
    }

    fn pack_args(
        &self,
        inputs: &[(RawPtr, &[usize])],
        params: &[RawPtr],
        output: RawPtr,
        output_shape: &[usize],
        _output_row_stride: i32,
        visitor: &mut dyn ArgVisitor,
    ) {
        let n: usize = output_shape.iter().product();
        visitor.visit_ptr(inputs[0].0); // x (branch input)
        visitor.visit_ptr(inputs[1].0); // z (unchained second input)
        visitor.visit_ptr(output); // y
        for &scratch in params.iter().take(self.n_scratch) {
            visitor.visit_ptr(scratch);
        }
        visitor.visit_i32(n as i32);
    }

    fn grid(&self, output_shape: &[usize]) -> [u32; 3] {
        let n: usize = output_shape.iter().product();
        PointwiseFuseProbe {
            block_size: self.block_size,
        }
        .grid(n)
    }
}

/// `(fn_name, kernel_body)` for `op` at `dtype`, instantiated at `block_size`
/// (the branch's, so both stages share one CTA/grid shape). Only `Op::Add`
/// for this first cut — extend alongside [`is_tile_fuse_tail`].
fn tail_kernel(op: &Op, dtype: DtypeRepr, block_size: i32) -> Result<(String, String), String> {
    match (op, dtype) {
        (Op::Add, DtypeRepr::F32) => {
            let k = ElemwiseAddForward::<f32>::new(block_size);
            Ok((k.name.to_string(), k.kernel_source.clone()))
        }
        (Op::Add, DtypeRepr::F64) => {
            let k = ElemwiseAddForward::<f64>::new(block_size);
            Ok((k.name.to_string(), k.kernel_source.clone()))
        }
        (other, dt) => Err(format!(
            "no TileFuse tail kernel for op={other:?} dtype={dt:?}"
        )),
    }
}

fn lower_tile_fuse(
    branch: &[Op],
    tail: &Op,
    dtype: DtypeRepr,
    probe: PointwiseFuseProbe,
) -> Result<(String, String, String, Arc<dyn RuntimeOp>), String> {
    if branch.is_empty() {
        return Err("TileFuse requires a non-empty branch".to_string());
    }
    if !is_tile_fuse_tail(tail) {
        return Err(format!("TileFuse tail op {tail:?} is not supported"));
    }

    let dtype_str = dtype_name(dtype)?;

    let branch_pieces = branch
        .iter()
        .map(|m| member_kernel(m, dtype))
        .collect::<Result<Vec<_>, _>>()?;
    for (i, p) in branch_pieces.iter().enumerate() {
        if !probe.compatible(p.probe) {
            return Err(format!(
                "TileFuse probe mismatch: branch has {probe:?}, member {i} has {:?}",
                p.probe
            ));
        }
    }

    // Instantiate the tail kernel directly at the *branch's* BLOCK_SIZE,
    // rather than resolving it through the generic per-op lowering table
    // (`graph/mod.rs`'s `Op::Add => make_num_kernel!(ElemwiseAddForward(128),
    // ...)`, fixed at 128 there): the two stages must share one CTA/grid
    // shape to compose into a single kernel, and `elemwise_add_forward`'s two
    // `In` pointers also fail `KernelIo::is_unary_elementwise`'s narrower
    // "exactly one In, one Out" shape, so it never gets a
    // `pointwise_fuse_block_size` from the probe machinery `PointwiseFuse`'s
    // members use — irrelevant here since `is_tile_fuse_tail` already
    // establishes the tail's identity directly, with no probe needed.
    let (tail_name, tail_body) = tail_kernel(tail, dtype, probe.block_size)?;
    if tail_body.is_empty() {
        return Err(format!("TileFuse tail kernel `{tail_name}` has empty body"));
    }

    let n_scratch = branch_pieces.len();
    let tag = branch_pieces
        .iter()
        .map(|p| p.fn_name.trim_end_matches("_forward"))
        .chain(std::iter::once(tail_name.trim_end_matches("_forward")))
        .collect::<Vec<_>>()
        .join("_");
    let fused_name = format!("tile_fuse_{tag}");
    let entry_point = format!("{fused_name}_entry_point");

    let mut bodies = String::new();
    for p in &branch_pieces {
        if !bodies.is_empty() {
            bodies.push_str("\n\n");
        }
        bodies.push_str(&p.kernel_source);
    }
    bodies.push_str("\n\n");
    bodies.push_str(&tail_body);

    let branch_fn_names: Vec<&str> = branch_pieces.iter().map(|p| p.fn_name.as_str()).collect();
    let entry = synthesize_tile_fuse_entry(
        &entry_point,
        dtype_str,
        probe.block_size,
        &branch_fn_names,
        &tail_name,
    );
    let kernel_source = format!("{bodies}\n\n{entry}");

    let runtime_op = Arc::new(TileFuseRuntimeOp {
        block_size: probe.block_size,
        n_scratch,
    });
    Ok((fused_name, kernel_source, entry_point, runtime_op))
}

/// Entry ABI: `x_ptr, z_ptr, y_ptr, scratch0.., n_elements`. The branch's
/// members chain `x_ptr → scratch0 → … → scratch{n-1}`; the tail reads
/// `scratch{n-1}` and `z_ptr`, writing `y_ptr`.
fn synthesize_tile_fuse_entry(
    entry_name: &str,
    dtype: &str,
    block_size: i32,
    branch_fn_names: &[&str],
    tail_fn_name: &str,
) -> String {
    let n_scratch = branch_fn_names.len();

    let mut params = format!("x_ptr: *mut {dtype}, z_ptr: *mut {dtype}, y_ptr: *mut {dtype}");
    for i in 0..n_scratch {
        params.push_str(&format!(", scratch{i}: *mut {dtype}"));
    }
    params.push_str(", n_elements: i32");

    let mut body = String::new();
    body.push_str("    let x_ptr = LlvmPointer(x_ptr as *mut _);\n");
    body.push_str("    let z_ptr = LlvmPointer(z_ptr as *mut _);\n");
    body.push_str("    let y_ptr = LlvmPointer(y_ptr as *mut _);\n");
    for i in 0..n_scratch {
        body.push_str(&format!(
            "    let scratch{i} = LlvmPointer(scratch{i} as *mut _);\n"
        ));
    }

    for (i, fname) in branch_fn_names.iter().enumerate() {
        let in_buf = if i == 0 {
            "x_ptr".to_string()
        } else {
            format!("scratch{}", i - 1)
        };
        let out_buf = format!("scratch{i}");
        body.push_str(&format!(
            "    {fname}::<LlvmTriton, {dtype}, {block_size}>({in_buf}, {out_buf}, n_elements);\n"
        ));
    }

    let branch_result = format!("scratch{}", n_scratch - 1);
    body.push_str(&format!(
        "    {tail_fn_name}::<LlvmTriton, {dtype}, {block_size}>({branch_result}, z_ptr, y_ptr, n_elements);\n"
    ));

    format!(
        concat!(
            "use triton::llvm::triton::num::*;\n",
            "use triton::llvm::triton::pointer::LlvmPointer;\n",
            "type LlvmTriton = triton::llvm::triton::LlvmTriton;\n",
            "\n",
            "#[no_mangle]\n",
            "pub extern \"C\" fn {entry}({params}) {{\n",
            "{body}",
            "}}"
        ),
        entry = entry_name,
        params = params,
        body = body,
    )
}
