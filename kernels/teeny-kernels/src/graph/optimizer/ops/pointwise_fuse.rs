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

//! [`PointwiseFuse`] — compose a linear chain of unary elementwise activations.

use std::any::Any;
use std::sync::Arc;

use teeny_core::device::program::ArgVisitor;
use teeny_core::graph::{CustomOp, DtypeRepr, Op, Shape};
use teeny_core::model::{RawPtr, RuntimeOp};
use teeny_triton::PointwiseFuseProbe;

use crate::graph::TritonLowering;
#[cfg(feature = "training")]
use crate::nn::activation::relu::ReluBackward;
#[cfg(feature = "training")]
use crate::nn::activation::sigmoid::SigmoidBackward;
#[cfg(feature = "training")]
use crate::nn::activation::tanh::TanhBackward;

/// CTA block size used when lowering unary activations into PointwiseFuse members.
#[cfg(feature = "training")]
const MEMBER_BLOCK_SIZE: i32 = 1024;

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

/// True when `dtype` is one [`PointwiseFuse::lower`] can actually emit (see
/// [`dtype_name`]). Callers building a chain must check this before fusing --
/// [`dtype_name`] itself only runs at lower time, which is too late to refuse
/// gracefully.
pub fn is_pointwise_fuse_dtype(dtype: DtypeRepr) -> bool {
    dtype_name(dtype).is_ok()
}

/// True when `op`'s output dtype (`bool`) differs from the float dtype
/// threaded through a chain's scratch buffers, so it may only be the last
/// member of a [`PointwiseFuse`] chain -- never a value a later member reads.
pub fn is_bool_terminal_only(op: &Op) -> bool {
    matches!(op, Op::IsNaN | Op::IsInf { .. })
}

/// Fused linear chain of unary pointwise activations.
///
/// Produced by Anduin; lowered by concatenating each member's `#[kernel]` body
/// and synthesizing a scratch entry (`in → m0 → scratch0 → … → out`).
///
/// Backward (training) is supported when every member uses the y-style ABI
/// `(dy, y, dx, n)` — currently [`Op::Relu`], [`Op::Sigmoid`], [`Op::Tanh`].
/// Forward scratch buffers retain each intermediate activation so reverse-mode
/// can walk the chain in-place on `dx`.
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

    /// True when every member has a y-style `(dy, y, dx, n)` backward.
    pub fn supports_fused_backward(&self) -> bool {
        self.members.iter().all(is_y_style_pointwise_bwd)
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
        #[cfg(feature = "training")]
        {
            if !self.supports_fused_backward() {
                return String::new();
            }
            match lower_pointwise_fuse_backward(&self.members, self.dtype, self.probe) {
                Ok(src) => src,
                Err(e) => panic!("PointwiseFuse::lower_backward_source failed: {e}"),
            }
        }
        #[cfg(not(feature = "training"))]
        {
            String::new()
        }
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

/// Runtime ABI: `x_ptr, y_ptr, scratch0 [, scratch1, …], n_elements`.
///
/// Backward ABI (when enabled): `dy_ptr, y_ptr, dx_ptr, scratch0 [, …], n_elements`.
struct PointwiseFuseRuntimeOp {
    block_size: i32,
    n_scratch: usize,
    #[cfg(feature = "training")]
    has_bwd: bool,
}

impl PointwiseFuseRuntimeOp {
    fn new(
        members: &[Arc<dyn RuntimeOp>],
        block_size: i32,
        #[cfg(feature = "training")] has_bwd: bool,
    ) -> Result<Self, String> {
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
        // One scratch slot per intermediate so longer chains (and backward)
        // keep every activation, not just a 2-buffer ping-pong.
        let n_scratch = members.len() - 1;
        Ok(Self {
            block_size,
            n_scratch,
            #[cfg(feature = "training")]
            has_bwd,
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
        // Static tables cover the chains we emit today (2–4 members → 1–3 scratch).
        match self.n_scratch {
            1 => &["scratch0"],
            2 => &["scratch0", "scratch1"],
            3 => &["scratch0", "scratch1", "scratch2"],
            _ => &["scratch0", "scratch1", "scratch2", "scratch3"],
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
        self.has_bwd
    }

    #[cfg(feature = "training")]
    fn pack_backward_args(
        &self,
        _inputs: &[(RawPtr, &[usize])],
        params: &[RawPtr],
        output: RawPtr,
        output_shape: &[usize],
        grad_output: RawPtr,
        _grad_output_row_stride: i32,
        grad_inputs: &[RawPtr],
        _grad_params: &[RawPtr],
        visitor: &mut dyn ArgVisitor,
    ) {
        let n: usize = output_shape.iter().product();
        visitor.visit_ptr(grad_output);
        visitor.visit_ptr(output);
        visitor.visit_ptr(grad_inputs[0]);
        for &scratch in params.iter().take(self.n_scratch) {
            visitor.visit_ptr(scratch);
        }
        visitor.visit_i32(n as i32);
    }

    #[cfg(feature = "training")]
    fn backward_grid(&self, _input_shapes: &[&[usize]], output_shape: &[usize]) -> [u32; 3] {
        self.grid(output_shape)
    }
}

fn is_y_style_pointwise_bwd(op: &Op) -> bool {
    matches!(op, Op::Relu | Op::Sigmoid | Op::Tanh)
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

    let dtype_name = dtype_name(dtype)?;

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
    #[cfg(feature = "training")]
    let has_bwd = members.iter().all(is_y_style_pointwise_bwd);
    let fused_rop = PointwiseFuseRuntimeOp::new(
        &runtime_ops,
        probe.block_size,
        #[cfg(feature = "training")]
        has_bwd,
    )?;

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

#[cfg(feature = "training")]
fn lower_pointwise_fuse_backward(
    members: &[Op],
    dtype: DtypeRepr,
    probe: PointwiseFuseProbe,
) -> Result<String, String> {
    if members.len() < 2 {
        return Err(format!(
            "PointwiseFuse backward requires at least 2 members, got {}",
            members.len()
        ));
    }
    if !members.iter().all(is_y_style_pointwise_bwd) {
        return Err(
            "PointwiseFuse fused backward only supports y-style members (Relu, Sigmoid, Tanh)"
                .into(),
        );
    }

    let dtype_name = dtype_name(dtype)?;
    let bwd_pieces: Vec<(String, String)> = members
        .iter()
        .map(|m| member_backward(m, dtype))
        .collect::<Result<Vec<_>, _>>()?;

    let fwd_tag = members
        .iter()
        .map(|m| match m {
            Op::Relu => "relu",
            Op::Sigmoid => "sigmoid",
            Op::Tanh => "tanh",
            _ => "op",
        })
        .collect::<Vec<_>>()
        .join("_");
    let fused_name = format!("pointwise_fuse_{fwd_tag}");
    let entry_point = format!("{fused_name}_backward_entry_point");

    let mut bodies = String::new();
    for (i, (name, body)) in bwd_pieces.iter().enumerate() {
        let _ = name;
        if i > 0 {
            bodies.push_str("\n\n");
        }
        bodies.push_str(body);
    }

    let bwd_names: Vec<&str> = bwd_pieces.iter().map(|(n, _)| n.as_str()).collect();
    let entry = synthesize_backward_entry(&entry_point, dtype_name, probe.block_size, &bwd_names);
    Ok(format!("{bodies}\n\n{entry}"))
}

fn dtype_name(dtype: DtypeRepr) -> Result<&'static str, String> {
    match dtype {
        DtypeRepr::F32 => Ok("f32"),
        DtypeRepr::F64 => Ok("f64"),
        other => Err(format!(
            "PointwiseFuse currently requires f32/f64, got {other:?}"
        )),
    }
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

/// `(backward_fn_name, kernel_body)` for a y-style member.
#[cfg(feature = "training")]
fn member_backward(op: &Op, dtype: DtypeRepr) -> Result<(String, String), String> {
    match (op, dtype) {
        (Op::Relu, DtypeRepr::F32) => {
            let k = ReluBackward::<f32>::new(MEMBER_BLOCK_SIZE);
            Ok((k.name.to_string(), k.kernel_source.clone()))
        }
        (Op::Relu, DtypeRepr::F64) => {
            let k = ReluBackward::<f64>::new(MEMBER_BLOCK_SIZE);
            Ok((k.name.to_string(), k.kernel_source.clone()))
        }
        (Op::Sigmoid, DtypeRepr::F32) => {
            let k = SigmoidBackward::<f32>::new(MEMBER_BLOCK_SIZE);
            Ok((k.name.to_string(), k.kernel_source.clone()))
        }
        (Op::Sigmoid, DtypeRepr::F64) => {
            let k = SigmoidBackward::<f64>::new(MEMBER_BLOCK_SIZE);
            Ok((k.name.to_string(), k.kernel_source.clone()))
        }
        (Op::Tanh, DtypeRepr::F32) => {
            let k = TanhBackward::<f32>::new(MEMBER_BLOCK_SIZE);
            Ok((k.name.to_string(), k.kernel_source.clone()))
        }
        (Op::Tanh, DtypeRepr::F64) => {
            let k = TanhBackward::<f64>::new(MEMBER_BLOCK_SIZE);
            Ok((k.name.to_string(), k.kernel_source.clone()))
        }
        (other, dt) => Err(format!(
            "no y-style PointwiseFuse backward for op={other:?} dtype={dt:?}"
        )),
    }
}

fn synthesize_entry(entry_name: &str, dtype: &str, block_size: i32, fn_names: &[&str]) -> String {
    let n = fn_names.len();
    let n_scratch = n - 1;

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
        } else {
            format!("scratch{}", i - 1)
        };
        let out_buf = if i + 1 == n {
            "y_ptr".to_string()
        } else {
            format!("scratch{i}")
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

/// Reverse-mode entry: walk members last→first, using forward scratch as `y` and
/// threading the gradient through `dx_ptr` in-place.
#[cfg(feature = "training")]
fn synthesize_backward_entry(
    entry_name: &str,
    dtype: &str,
    block_size: i32,
    bwd_fn_names: &[&str],
) -> String {
    let n = bwd_fn_names.len();
    let n_scratch = n - 1;

    let mut params = String::from("dy_ptr: *mut ");
    params.push_str(dtype);
    params.push_str(", y_ptr: *mut ");
    params.push_str(dtype);
    params.push_str(", dx_ptr: *mut ");
    params.push_str(dtype);
    for i in 0..n_scratch {
        params.push_str(&format!(", scratch{i}: *mut {dtype}"));
    }
    params.push_str(", n_elements: i32");

    let mut body = String::new();
    body.push_str("    let dy_ptr = LlvmPointer(dy_ptr as *mut _);\n");
    body.push_str("    let y_ptr = LlvmPointer(y_ptr as *mut _);\n");
    body.push_str("    let dx_ptr = LlvmPointer(dx_ptr as *mut _);\n");
    for i in 0..n_scratch {
        body.push_str(&format!(
            "    let scratch{i} = LlvmPointer(scratch{i} as *mut _);\n"
        ));
    }

    // Member i's forward output lives in y (last) or scratch{i} (intermediate).
    for i in (0..n).rev() {
        let fname = bwd_fn_names[i];
        let y_buf = if i + 1 == n {
            "y_ptr".to_string()
        } else {
            format!("scratch{i}")
        };
        let dy_buf = if i + 1 == n {
            "dy_ptr".to_string()
        } else {
            "dx_ptr".to_string()
        };
        body.push_str(&format!(
            "    {fname}::<LlvmTriton, {dtype}, {block_size}>({dy_buf}, {y_buf}, dx_ptr, n_elements);\n"
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pointwise_fuse_dtype_accepts_only_float() {
        assert!(is_pointwise_fuse_dtype(DtypeRepr::F32));
        assert!(is_pointwise_fuse_dtype(DtypeRepr::F64));
        assert!(!is_pointwise_fuse_dtype(DtypeRepr::I32));
        assert!(!is_pointwise_fuse_dtype(DtypeRepr::Bool));
    }

    #[test]
    fn bool_terminal_only_flags_isnan_and_isinf() {
        assert!(is_bool_terminal_only(&Op::IsNaN));
        assert!(is_bool_terminal_only(&Op::IsInf {
            detect_negative: true,
            detect_positive: true,
        }));
        assert!(!is_bool_terminal_only(&Op::Relu));
        assert!(!is_bool_terminal_only(&Op::Sign));
    }
}
