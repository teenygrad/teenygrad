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

//! [`PointwiseFuse`] — compose a linear chain of unary elementwise activations.

use std::any::Any;
use std::sync::Arc;

use teeny_core::device::program::ArgVisitor;
use teeny_core::graph::{CustomOp, DtypeRepr, Op, Shape};
use teeny_core::model::{RawPtr, RuntimeOp};
use teeny_triton::PointwiseFuseProbe;

use crate::graph::TritonLowering;

/// Probe whether `op` is pointwise-fusable at `dtype`.
///
/// Instantiates the op through [`TritonLowering::lower_unary_op`] and keeps it
/// only when the lowered kernel's pointwise-fuse probe succeeds.
pub fn probe_pointwise_op(op: &Op, dtype: DtypeRepr) -> Option<PointwiseFuseProbe> {
    match op {
        Op::Custom { data } => {
            if let Some(pf) = data.downcast_ref::<PointwiseFuse>() {
                return Some(pf.probe);
            }
            data.0
                .pointwise_fuse_block_size()
                .map(|block_size| PointwiseFuseProbe { block_size })
        }
        _ => member_kernel(op, dtype).ok().map(|m| m.probe),
    }
}

/// Fused linear chain of unary pointwise activations.
///
/// Produced by Anduin; lowered by concatenating each member's `#[kernel]` body
/// and synthesizing a scratch ping-pong entry (`in → m0 → scratch → m1 → out`).
#[derive(Debug, Clone)]
pub struct PointwiseFuse {
    /// Member ops in execution order (length ≥ 2).
    pub members: Vec<Op>,
    /// Element dtype of the chain (must be float for current activation mix).
    pub dtype: DtypeRepr,
    /// Shared CTA probe for the chain (all members compatible).
    pub probe: PointwiseFuseProbe,
}

impl PointwiseFuse {
    /// Builds a fuse of `members` that already share `probe` (len ≥ 2).
    pub fn new(members: Vec<Op>, dtype: DtypeRepr, probe: PointwiseFuseProbe) -> Self {
        debug_assert!(members.len() >= 2);
        Self {
            members,
            dtype,
            probe,
        }
    }
}

impl CustomOp for PointwiseFuse {
    fn name(&self) -> &str {
        "pointwise_fuse"
    }

    fn infer_output_shape(&self, input_shapes: &[&Shape]) -> Shape {
        input_shapes[0].clone()
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn lower(&self) -> Option<(String, String, String, Arc<dyn RuntimeOp>)> {
        match lower_pointwise_fuse(&self.members, self.dtype, self.probe) {
            Ok(v) => Some(v),
            Err(e) => panic!("PointwiseFuse::lower failed: {e}"),
        }
    }

    fn lower_backward_source(&self) -> String {
        // Inference-first: no fused backward yet.
        String::new()
    }

    fn pointwise_fuse_block_size(&self) -> Option<i32> {
        Some(self.probe.block_size)
    }
}

struct MemberKernel {
    fn_name: String,
    kernel_source: String,
    runtime_op: Arc<dyn RuntimeOp>,
    probe: PointwiseFuseProbe,
}

/// Runtime ABI: `x_ptr, y_ptr, scratch0 [, scratch1], n_elements`.
struct PointwiseFuseRuntimeOp {
    block_size: i32,
    n_scratch: usize,
}

impl PointwiseFuseRuntimeOp {
    fn new(members: &[Arc<dyn RuntimeOp>], block_size: i32) -> Result<Self, String> {
        if members.len() < 2 {
            return Err("PointwiseFuse needs at least 2 members".into());
        }
        let sample = [16usize];
        let g0 = members[0].grid(&sample);
        for (i, m) in members.iter().enumerate().skip(1) {
            let g = m.grid(&sample);
            if g != g0 {
                return Err(format!(
                    "PointwiseFuse grid mismatch: member 0 grid={g0:?}, member {i} grid={g:?}"
                ));
            }
        }
        let n_scratch = if members.len() == 2 { 1 } else { 2 };
        Ok(Self {
            block_size,
            n_scratch,
        })
    }
}

impl RuntimeOp for PointwiseFuseRuntimeOp {
    fn n_activation_inputs(&self) -> usize {
        1
    }

    fn param_shapes(&self, _input_shapes: &[&[usize]], output_shape: &[usize]) -> Vec<Vec<usize>> {
        let n: usize = output_shape.iter().product();
        vec![vec![n]; self.n_scratch]
    }

    fn param_names(&self) -> &'static [&'static str] {
        match self.n_scratch {
            1 => &["scratch0"],
            _ => &["scratch0", "scratch1"],
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
        visitor.visit_ptr(inputs[0].0);
        visitor.visit_ptr(output);
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

    #[cfg(feature = "training")]
    fn has_backward(&self) -> bool {
        false
    }
}

fn lower_pointwise_fuse(
    members: &[Op],
    dtype: DtypeRepr,
    probe: PointwiseFuseProbe,
) -> Result<(String, String, String, Arc<dyn RuntimeOp>), String> {
    if members.len() < 2 {
        return Err(format!(
            "PointwiseFuse requires at least 2 members, got {}",
            members.len()
        ));
    }

    let dtype_name = match dtype {
        DtypeRepr::F32 => "f32",
        DtypeRepr::F64 => "f64",
        other => {
            return Err(format!(
                "PointwiseFuse currently requires f32/f64, got {other:?}"
            ));
        }
    };

    let pieces: Vec<MemberKernel> = members
        .iter()
        .map(|m| member_kernel(m, dtype))
        .collect::<Result<Vec<_>, _>>()?;

    for (i, p) in pieces.iter().enumerate() {
        if !probe.compatible(p.probe) {
            return Err(format!(
                "PointwiseFuse probe mismatch: chain has {probe:?}, member {i} has {:?}",
                p.probe
            ));
        }
    }

    let runtime_ops: Vec<Arc<dyn RuntimeOp>> =
        pieces.iter().map(|p| Arc::clone(&p.runtime_op)).collect();
    let fused_rop = PointwiseFuseRuntimeOp::new(&runtime_ops, probe.block_size)?;

    let tag = pieces
        .iter()
        .map(|p| p.fn_name.trim_end_matches("_forward"))
        .collect::<Vec<_>>()
        .join("_");
    let fused_name = format!("pointwise_fuse_{tag}");
    let entry_point = format!("{fused_name}_entry_point");

    let mut bodies = String::new();
    for (i, p) in pieces.iter().enumerate() {
        if i > 0 {
            bodies.push_str("\n\n");
        }
        bodies.push_str(&p.kernel_source);
    }

    let entry = synthesize_entry(
        &entry_point,
        dtype_name,
        probe.block_size,
        &pieces
            .iter()
            .map(|p| p.fn_name.as_str())
            .collect::<Vec<_>>(),
    );
    let kernel_source = format!("{bodies}\n\n{entry}");

    Ok((fused_name, kernel_source, entry_point, Arc::new(fused_rop)))
}

/// Resolve `op` via [`TritonLowering::lower_unary_op`], then keep it only if
/// the pointwise-fuse probe succeeds.
fn member_kernel(op: &Op, dtype: DtypeRepr) -> Result<MemberKernel, String> {
    let exec = TritonLowering::new()
        .lower_unary_op(op, dtype)
        .map_err(|e| e.to_string())?;
    let block_size = exec.pointwise_fuse_block_size.ok_or_else(|| {
        format!(
            "kernel `{}` is not pointwise-fusable (failed pointwise fuse probe)",
            exec.name
        )
    })?;
    if exec.kernel_body.is_empty() {
        return Err(format!(
            "kernel `{}` has empty kernel_body; cannot compose into PointwiseFuse",
            exec.name
        ));
    }
    Ok(MemberKernel {
        fn_name: exec.name,
        kernel_source: exec.kernel_body,
        runtime_op: exec.runtime_op,
        probe: PointwiseFuseProbe { block_size },
    })
}

fn synthesize_entry(entry_name: &str, dtype: &str, block_size: i32, fn_names: &[&str]) -> String {
    let n = fn_names.len();
    let n_scratch = if n == 2 { 1 } else { 2 };

    let mut params = String::from("x_ptr: *mut ");
    params.push_str(dtype);
    params.push_str(", y_ptr: *mut ");
    params.push_str(dtype);
    for i in 0..n_scratch {
        params.push_str(&format!(", scratch{i}: *mut {dtype}"));
    }
    params.push_str(", n_elements: i32");

    let mut body = String::new();
    body.push_str("    let x_ptr = LlvmPointer(x_ptr as *mut _);\n");
    body.push_str("    let y_ptr = LlvmPointer(y_ptr as *mut _);\n");
    for i in 0..n_scratch {
        body.push_str(&format!(
            "    let scratch{i} = LlvmPointer(scratch{i} as *mut _);\n"
        ));
    }

    for (i, fname) in fn_names.iter().enumerate() {
        let in_buf = if i == 0 {
            "x_ptr".to_string()
        } else if n == 2 {
            "scratch0".to_string()
        } else {
            format!("scratch{}", (i - 1) % 2)
        };
        let out_buf = if i + 1 == n {
            "y_ptr".to_string()
        } else if n == 2 {
            "scratch0".to_string()
        } else {
            format!("scratch{}", i % 2)
        };
        body.push_str(&format!(
            "    {fname}::<LlvmTriton, {dtype}, {block_size}>({in_buf}, {out_buf}, n_elements);\n"
        ));
    }

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
