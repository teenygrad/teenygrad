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

//! [`ReduceFuse`] — a unary pointwise chain fused directly into a row-reduction.
//!
//! teenygrad-3w0.9: fusion taxonomy case 4 (`y = reduce_sum(relu(x))`).
//! Structurally different from [`super::PointwiseFuse`]/[`super::TileFuse`],
//! both of which fuse by concatenating **whole, unmodified** kernel function
//! bodies that all share one flat `BLOCK_SIZE`/`pid*BLOCK_SIZE+arange`/
//! `n_elements` grid. Row-reduction kernels (`nn::tensor::reduction`) use a
//! structurally different grid — single CTA per output row, `BLOCK_INNER`/
//! `n_inner`/`n_outer` — so reaching this fusion shape means re-deriving the
//! chain members' index arithmetic under the reduction's own row-major grid,
//! not concatenating whole bodies. This module builds one freshly synthesized
//! kernel: the reduction's own load/reduce/store shape, with each chain
//! member's [`teeny_triton::FusionCore`] spliced in between the load and the
//! reduction call.
//!
//! **v1 scope (hard boundary, not full generality):** reduce op is
//! `Op::ReduceSum` or `Op::ReduceMean` only — every other `Op::Reduce*`
//! variant has either a different load-fill (`ReduceMax`/`ReduceMin`'s
//! +/-inf fill) or a materially different epilogue (`ReduceL2`'s sqrt,
//! `ReduceProd`'s product, `ReduceLogSum(Exp)`'s log, ...) not yet
//! hand-verified and templated here — extending `reduce_epilogue` to cover
//! them is a follow-up, not attempted blind. Chain members are `Op::Relu`,
//! `Op::Sigmoid`, `Op::Tanh`, `Op::Exp` only — the zero-extra-param
//! `fusion_core()`-capable ops; `Op::Elu`/`Op::Selu` are `fusion_core()`-ready
//! too but need their extra scalar params (e.g. `alpha`) threaded through
//! this module's synthesized signature/ABI, deferred as a follow-up.

use std::any::Any;
use std::sync::Arc;

use teeny_core::device::program::ArgVisitor;
use teeny_core::graph::{CustomOp, DtypeRepr, Op, Shape};
use teeny_core::model::{RawPtr, RuntimeOp};
use teeny_triton::{CostModel, FusionCore, KernelTileSpec, TensorTileSpec, TileAxisBinding};

use crate::graph::optimizer::ops::pointwise_fuse::dtype_name;
use crate::nn::activation::relu::ReluForward;
use crate::nn::activation::sigmoid::SigmoidForward;
use crate::nn::activation::tanh::TanhForward;
use crate::nn::tensor::elemwise_unary::ElemwiseExpForward;

/// Default/fallback CTA size for the fused kernel's row dimension, matching
/// every existing `Op::Reduce* => make_*_kernel!(...Forward(1024), node)`
/// call site (`graph/mod.rs`). Used only by tests that don't care about the
/// teenygrad-3w0.11 search below; real fusion decisions go through
/// [`choose_reduce_fuse_block_inner`] instead.
#[cfg(test)]
const DEFAULT_REDUCE_FUSE_BLOCK_INNER: i32 = 1024;

/// Universal real-hardware ceiling for CUDA block size across current
/// NVIDIA architectures (`CudaDeviceInfo::max_threads_per_block`, always
/// 1024 on every architecture this project has run on so far). Anduin runs
/// as a static graph pass with no device handle, so this is a documented
/// constant rather than a runtime query — same convention `REDUCE_FUSE_COST_MODEL`
/// below already uses for its hardware facts.
const MAX_THREADS_PER_BLOCK: i32 = 1024;

/// teenygrad-3w0.4's calibrated constants (RTX 5070, 48 SMs), reused as-is —
/// see `test_mem_traffic_calibration.rs`'s
/// `cost_model_ranks_real_launches_like_measured_latency` for where these
/// conservative, rounded-down values come from. Re-derive if retargeting to
/// different hardware, same framing `CostModel`'s own doc comment uses.
const REDUCE_FUSE_COST_MODEL: CostModel = CostModel {
    sm_count: 48,
    saturation_ctas_per_sm: 8,
    under_parallel_penalty: 1.5,
    window_penalty: 1.2,
    // ReduceFuse never stages through shared memory, so
    // `penalized_traffic_with_shared_mem` is never called on this model —
    // these two fields are unused, present only because `CostModel` is
    // non-`Default`. Same real-hardware value as `shared_transpose_fuse.rs`
    // (RTX 5070) for consistency; the penalty is a neutral no-op.
    shared_mem_budget_bytes: 49_152,
    shared_mem_occupancy_penalty: 1.0,
};

/// Static tile-shape declaration for `ReduceFuse`'s synthesized kernel,
/// modeled on `matmul_forward`'s real declared spec. `x_ptr`'s one tiled
/// axis is what a candidate `BLOCK_INNER` actually varies — `TensorTileSpec`/
/// `TileAxisBinding` only record parameter *names*, so the concrete value
/// is supplied per candidate via the `resolve` closure passed to
/// `mem_traffic`/`penalized_traffic`, not baked into this spec. `y_ptr` has
/// no tiled axis at all (its real per-row output is a single scalar); both
/// tensors carry `untiled_dims: &["n_outer"]` for the real, grid-driven row
/// dimension (conv2d_forward's own precedent for this field, teenygrad-3w0.8).
const REDUCE_FUSE_TILE_SPEC: KernelTileSpec = KernelTileSpec {
    inputs: &[TensorTileSpec {
        param: "x_ptr",
        rank: 1,
        axes: &[TileAxisBinding {
            dim: 0,
            block_const: "BLOCK_INNER",
            extent_param: "n_inner",
            window: None,
        }],
        reduction_axis: None,
        untiled_dims: &["n_outer"],
    }],
    outputs: &[TensorTileSpec {
        param: "y_ptr",
        rank: 0,
        axes: &[],
        reduction_axis: None,
        untiled_dims: &["n_outer"],
    }],
    loop_spec: None,
};

/// Enumerates the power-of-two CTA ladder `{32, 64, ..., 1024}` (clamped to
/// `max_threads_per_block`), costs each candidate `BLOCK_INNER` via the
/// calibrated `REDUCE_FUSE_COST_MODEL`, and returns the minimum-cost one.
///
/// `BLOCK_INNER` is a correctness constraint here, not just a perf knob:
/// `ReduceFuse`'s kernel does a single-shot load (`REDUCE_FUSE_LOAD_PRELUDE`,
/// no chunking loop over `n_inner`), so any candidate `< n_inner` would
/// silently drop elements past `BLOCK_INNER` rather than reduce them —
/// candidates below `n_inner` are excluded, not merely deprioritized.
/// Returns `None` when no candidate covers `n_inner` (i.e.
/// `n_inner > max_threads_per_block`) — the caller must decline to fuse in
/// that case, not guess.
///
/// Under `ReduceFuse`'s fixed `[n_outer, 1, 1]` grid (independent of
/// `BLOCK_INNER`) and single-shot load (`n_tiles` is always exactly `1` for
/// any valid candidate), `CostModel`'s `under_parallel_penalty` term is
/// identical across every candidate — it never discriminates for this
/// search. The `n_ctas` value passed to `penalized_traffic` below is
/// therefore an arbitrary fixed stand-in (chosen large enough that the
/// penalty is consistently *not* applied, matching the common case), not a
/// per-candidate value; only `mem_traffic`'s raw byte estimate (which
/// scales with `BLOCK_INNER`'s padding waste past `n_inner`) actually
/// varies the ranking.
pub(crate) fn choose_reduce_fuse_block_inner(n_inner: i64) -> Option<i32> {
    let n_ctas =
        (REDUCE_FUSE_COST_MODEL.sm_count * REDUCE_FUSE_COST_MODEL.saturation_ctas_per_sm) as u64;
    [32, 64, 128, 256, 512, 1024]
        .into_iter()
        .filter(|&b| b <= MAX_THREADS_PER_BLOCK && i64::from(b) >= n_inner)
        .map(|b| {
            let cost = REDUCE_FUSE_COST_MODEL.penalized_traffic(
                &REDUCE_FUSE_TILE_SPEC,
                4, // f32; a dtype-size mismatch doesn't change the argmin ranking
                |name| match name {
                    "BLOCK_INNER" => i64::from(b),
                    "n_inner" => n_inner,
                    "n_outer" => 1,
                    other => panic!("unexpected REDUCE_FUSE_TILE_SPEC param {other}"),
                },
                n_ctas,
            );
            (b, cost)
        })
        .min_by(|(_, a), (_, b)| a.total_cmp(b))
        .map(|(b, _)| b)
}

/// True when `op` is a chain member `ReduceFuse` v1 can splice (zero extra
/// scalar params — see module doc).
pub fn is_reduce_fuse_member(op: &Op) -> bool {
    matches!(op, Op::Relu | Op::Sigmoid | Op::Tanh | Op::Exp)
}

/// True when `op` is a reduce op `ReduceFuse` v1 can terminate a chain with.
pub fn is_reduce_fuse_reducible(op: &Op) -> bool {
    matches!(op, Op::ReduceSum { .. } | Op::ReduceMean { .. })
}

fn reduce_keepdims(op: &Op) -> bool {
    match op {
        Op::ReduceSum { keepdims, .. } | Op::ReduceMean { keepdims, .. } => *keepdims,
        _ => false,
    }
}

/// A unary pointwise chain fused directly into a row-reduction's own kernel
/// — e.g. `y = reduce_sum(relu(x))`. See the module doc for v1 scope.
#[derive(Debug, Clone)]
pub struct ReduceFuse {
    /// Unary chain applied to the input before reduction (may be empty —
    /// a bare `reduce_sum(x)` with no chain is still a valid, if trivial,
    /// `ReduceFuse`).
    pub chain: Vec<Op>,
    /// Terminal reduction op. Must satisfy [`is_reduce_fuse_reducible`].
    pub reduce_op: Op,
    /// Element dtype shared by every chain member and the reduction.
    pub dtype: DtypeRepr,
    /// CTA size for the fused kernel's row dimension (`BLOCK_INNER`) —
    /// teenygrad-3w0.11's cost-driven choice, from
    /// [`choose_reduce_fuse_block_inner`], not a fixed constant.
    pub block_inner: i32,
}

impl ReduceFuse {
    /// Builds a fuse of `chain` (every member must satisfy
    /// [`is_reduce_fuse_member`]) terminated by `reduce_op` (must satisfy
    /// [`is_reduce_fuse_reducible`]), at the given `block_inner` (must be
    /// `>=` the reduction axis's real extent — see
    /// [`choose_reduce_fuse_block_inner`]'s doc comment for why).
    pub fn new(chain: Vec<Op>, reduce_op: Op, dtype: DtypeRepr, block_inner: i32) -> Self {
        debug_assert!(chain.iter().all(is_reduce_fuse_member));
        debug_assert!(is_reduce_fuse_reducible(&reduce_op));
        Self {
            chain,
            reduce_op,
            dtype,
            block_inner,
        }
    }
}

impl CustomOp for ReduceFuse {
    fn name(&self) -> &str {
        "reduce_fuse"
    }

    fn infer_output_shape(&self, input_shapes: &[&Shape]) -> Shape {
        // Same "no axis info at static-inference time" convention as the
        // free `infer_output_shape`'s `Op::Reduce*` arm (`core/teeny-core`):
        // keepdims=true -> same rank, keepdims=false -> reduce all -> scalar.
        if reduce_keepdims(&self.reduce_op) {
            input_shapes[0].clone()
        } else {
            vec![Some(1)]
        }
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn lower(&self) -> Option<(String, String, String, Arc<dyn RuntimeOp>)> {
        match lower_reduce_fuse(&self.chain, &self.reduce_op, self.dtype, self.block_inner) {
            Ok(v) => Some(v),
            Err(e) => panic!("ReduceFuse::lower failed: {e}"),
        }
    }
}

/// Runtime ABI: `x_ptr, y_ptr, n_inner, n_outer` — no scratch buffers, unlike
/// [`super::PointwiseFuse`]/[`super::TileFuse`]: this is one synthesized
/// kernel, not several member kernels chained through memory.
struct ReduceFuseRuntimeOp;

impl RuntimeOp for ReduceFuseRuntimeOp {
    fn n_activation_inputs(&self) -> usize {
        1
    }

    fn param_shapes(&self, _input_shapes: &[&[usize]], _output_shape: &[usize]) -> Vec<Vec<usize>> {
        vec![]
    }

    fn pack_args(
        &self,
        inputs: &[(RawPtr, &[usize])],
        _params: &[RawPtr],
        output: RawPtr,
        output_shape: &[usize],
        _output_row_stride: i32,
        visitor: &mut dyn ArgVisitor,
    ) {
        // Mirrors `reduction.rs`'s `impl_reduce_num_runtime_op!`/
        // `impl_reduce_float_runtime_op!` exactly.
        let n_outer: usize = output_shape.iter().product::<usize>().max(1);
        let n_total: usize = inputs[0].1.iter().product();
        let n_inner: usize = n_total / n_outer;
        visitor.visit_ptr(inputs[0].0);
        visitor.visit_ptr(output);
        visitor.visit_i32(n_inner as i32);
        visitor.visit_i32(n_outer as i32);
    }

    fn grid(&self, output_shape: &[usize]) -> [u32; 3] {
        let n_outer: usize = output_shape.iter().product::<usize>().max(1);
        [n_outer as u32, 1, 1]
    }
}

/// `fusion_core()` for `op` at `dtype` — explicit match arms only (mirrors
/// `tile_fuse.rs`'s `tail_kernel` bypass-the-generic-table precedent). Only
/// the zero-extra-param ops in [`is_reduce_fuse_member`]'s set.
pub(super) fn member_fusion_core(op: &Op, dtype: DtypeRepr) -> Option<FusionCore> {
    match (op, dtype) {
        (Op::Relu, DtypeRepr::F32) => ReluForward::<f32>::fusion_core(),
        (Op::Relu, DtypeRepr::F64) => ReluForward::<f64>::fusion_core(),
        (Op::Sigmoid, DtypeRepr::F32) => SigmoidForward::<f32>::fusion_core(),
        (Op::Sigmoid, DtypeRepr::F64) => SigmoidForward::<f64>::fusion_core(),
        (Op::Tanh, DtypeRepr::F32) => TanhForward::<f32>::fusion_core(),
        (Op::Tanh, DtypeRepr::F64) => TanhForward::<f64>::fusion_core(),
        (Op::Exp, DtypeRepr::F32) => ElemwiseExpForward::<f32>::fusion_core(),
        (Op::Exp, DtypeRepr::F64) => ElemwiseExpForward::<f64>::fusion_core(),
        _ => None,
    }
}

/// `(tag, epilogue_source)` for a v1-supported reduce op, transcribed
/// verbatim from that op's real kernel body in `reduction.rs` (the part
/// after the shared load prelude). See the module doc for why only these
/// two are covered.
fn reduce_epilogue(reduce_op: &Op) -> Result<(&'static str, &'static str), String> {
    match reduce_op {
        Op::ReduceSum { .. } => Ok((
            "sum",
            "    let sum = T::sum(x, Some(0), true);\n\
             \x20   let row_offsets = T::arange(0, 1) + row;\n\
             \x20   T::store(y_ptr.add_offsets(row_offsets), sum, None, &[], None, None);\n",
        )),
        Op::ReduceMean { .. } => Ok((
            "mean",
            "    let sum = T::sum(x, Some(0), true);\n\
             \x20   let n_f = T::cast::<i32, D>(T::full::<i32>(&[1], n_inner), None, false);\n\
             \x20   let mean = sum / n_f;\n\
             \x20   let row_offsets = T::arange(0, 1) + row;\n\
             \x20   T::store(y_ptr.add_offsets(row_offsets), mean, None, &[], None, None);\n",
        )),
        other => Err(format!(
            "ReduceFuse v1 supports Op::ReduceSum/Op::ReduceMean only, got {other:?}"
        )),
    }
}

/// The reduction's own load prelude, transcribed verbatim from
/// `reduce_sum_forward`/`reduce_mean_forward` (identical in both).
const REDUCE_FUSE_LOAD_PRELUDE: &str = "    let row = T::program_id(Axis::X);\n\
    if row >= n_outer {\n\
    \x20   return;\n\
    }\n\
    let col_offsets = T::arange(0, BLOCK_INNER);\n\
    let offsets = col_offsets + row * n_inner;\n\
    let mask = col_offsets.lt(n_inner);\n\
    let x = T::load(\n\
    \x20   x_ptr.add_offsets(offsets),\n\
    \x20   Some(mask),\n\
    \x20   Some(T::zeros::<D>(&[BLOCK_INNER])),\n\
    \x20   &[],\n\
    \x20   None,\n\
    \x20   None,\n\
    \x20   None,\n\
    \x20   false,\n\
    );\n";

/// Splice `member_cores` (in chain order) between the load prelude and the
/// reduction epilogue, threading each member's output into the next's
/// input, and finally back into `x` (the epilogue's expected local name).
pub(super) fn splice_chain(member_cores: &[FusionCore]) -> String {
    let mut spliced = String::new();
    let mut prev_ident = "x".to_string();
    for core in member_cores {
        if core.input_ident != prev_ident {
            spliced.push_str(&format!("    let {} = {};\n", core.input_ident, prev_ident));
        }
        spliced.push_str("    ");
        spliced.push_str(core.body_source);
        spliced.push('\n');
        prev_ident = core.output_ident.to_string();
    }
    if prev_ident != "x" {
        spliced.push_str(&format!("    let x = {prev_ident};\n"));
    }
    spliced
}

fn reduce_fuse_kernel_source(fn_name: &str, member_cores: &[FusionCore], epilogue: &str) -> String {
    format!(
        "pub fn {fn_name}<T: Triton, D: Float, const BLOCK_INNER: i32>(\n\
         \x20   x_ptr: T::Pointer<D>,\n\
         \x20   y_ptr: T::Pointer<D>,\n\
         \x20   n_inner: i32,\n\
         \x20   n_outer: i32,\n\
         ) where\n\
         \x20   T::I32Tensor: types::Tensor<i32, 1>,\n\
         \x20   T::I32Tensor: Comparison<i32, BoolTensor = T::BoolTensor>,\n\
         \x20   T::Pointer<D>: AddOffsets<i32, 1, T::I32Tensor, Output = T::Tensor<T::Pointer<D>>>,\n\
         {{\n\
         {REDUCE_FUSE_LOAD_PRELUDE}\
         {spliced}\
         {epilogue}\
         }}\n",
        spliced = splice_chain(member_cores),
    )
}

fn synthesize_reduce_fuse_entry(
    entry_name: &str,
    dtype: &str,
    fn_name: &str,
    block_inner: i32,
) -> String {
    format!(
        concat!(
            "use triton::llvm::triton::num::*;\n",
            "use triton::llvm::triton::pointer::LlvmPointer;\n",
            "type LlvmTriton = triton::llvm::triton::LlvmTriton;\n",
            "\n",
            "#[no_mangle]\n",
            "pub extern \"C\" fn {entry}(x_ptr: *mut {dtype}, y_ptr: *mut {dtype}, n_inner: i32, n_outer: i32) {{\n",
            "    let x_ptr = LlvmPointer(x_ptr as *mut _);\n",
            "    let y_ptr = LlvmPointer(y_ptr as *mut _);\n",
            "    {fn_name}::<LlvmTriton, {dtype}, {block_inner}>(x_ptr, y_ptr, n_inner, n_outer);\n",
            "}}"
        ),
        entry = entry_name,
        dtype = dtype,
        fn_name = fn_name,
        block_inner = block_inner,
    )
}

fn lower_reduce_fuse(
    chain: &[Op],
    reduce_op: &Op,
    dtype: DtypeRepr,
    block_inner: i32,
) -> Result<(String, String, String, Arc<dyn RuntimeOp>), String> {
    if !is_reduce_fuse_reducible(reduce_op) {
        return Err(format!(
            "ReduceFuse reduce op {reduce_op:?} is not supported"
        ));
    }
    if !chain.iter().all(is_reduce_fuse_member) {
        return Err(format!(
            "ReduceFuse chain has an unsupported member: {chain:?}"
        ));
    }

    let dtype_str = dtype_name(dtype)?;
    let (reduce_tag, epilogue) = reduce_epilogue(reduce_op)?;

    let member_cores: Vec<FusionCore> = chain
        .iter()
        .map(|op| {
            member_fusion_core(op, dtype).ok_or_else(|| {
                format!("no ReduceFuse fusion_core for member op={op:?} dtype={dtype:?}")
            })
        })
        .collect::<Result<_, _>>()?;

    let tag = chain
        .iter()
        .map(|op| format!("{op:?}").to_lowercase())
        .chain(std::iter::once(reduce_tag.to_string()))
        .collect::<Vec<_>>()
        .join("_");
    let fused_name = format!("reduce_fuse_{tag}");
    let entry_point = format!("{fused_name}_entry_point");

    let kernel_source = reduce_fuse_kernel_source(&fused_name, &member_cores, epilogue);
    let entry = synthesize_reduce_fuse_entry(&entry_point, dtype_str, &fused_name, block_inner);
    let full_source = format!("{kernel_source}\n\n{entry}");

    Ok((
        fused_name,
        full_source,
        entry_point,
        Arc::new(ReduceFuseRuntimeOp) as Arc<dyn RuntimeOp>,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reduce_fuse_member_and_reducible_gates() {
        assert!(is_reduce_fuse_member(&Op::Relu));
        assert!(is_reduce_fuse_member(&Op::Sigmoid));
        assert!(is_reduce_fuse_member(&Op::Tanh));
        assert!(is_reduce_fuse_member(&Op::Exp));
        assert!(!is_reduce_fuse_member(&Op::Elu { alpha: 1.0 }));
        assert!(!is_reduce_fuse_member(&Op::Selu));

        assert!(is_reduce_fuse_reducible(&Op::ReduceSum {
            keepdims: true,
            noop_with_empty_axes: false,
        }));
        assert!(is_reduce_fuse_reducible(&Op::ReduceMean {
            keepdims: true,
            noop_with_empty_axes: false,
        }));
        assert!(!is_reduce_fuse_reducible(&Op::ReduceMax {
            keepdims: true,
            noop_with_empty_axes: false,
        }));
        assert!(!is_reduce_fuse_reducible(&Op::ReduceLogSumExp {
            keepdims: true,
            noop_with_empty_axes: false,
        }));
    }

    #[test]
    fn lower_reduce_fuse_relu_sum_splices_relu_core_before_reduction() {
        let (name, source, entry_point, _rop) = lower_reduce_fuse(
            &[Op::Relu],
            &Op::ReduceSum {
                keepdims: false,
                noop_with_empty_axes: false,
            },
            DtypeRepr::F32,
            DEFAULT_REDUCE_FUSE_BLOCK_INNER,
        )
        .expect("relu -> reduce_sum is ReduceFuse v1's canonical shape");

        assert_eq!(name, "reduce_fuse_relu_sum");
        assert_eq!(entry_point, "reduce_fuse_relu_sum_entry_point");
        assert!(source.contains("T :: maximum"), "relu's core spliced in");
        assert!(
            source.contains("T::sum(x"),
            "reduction reads the spliced value back as `x`"
        );
        assert!(source.contains(&entry_point));
    }

    #[test]
    fn lower_reduce_fuse_rejects_unsupported_reduce_op() {
        let result = lower_reduce_fuse(
            &[Op::Relu],
            &Op::ReduceMax {
                keepdims: true,
                noop_with_empty_axes: false,
            },
            DtypeRepr::F32,
            DEFAULT_REDUCE_FUSE_BLOCK_INNER,
        );
        let err = match result {
            Err(e) => e,
            Ok(_) => panic!("expected an error for an unsupported reduce op"),
        };
        assert!(err.contains("ReduceMax"));
    }

    #[test]
    fn lower_reduce_fuse_rejects_unsupported_chain_member() {
        let result = lower_reduce_fuse(
            &[Op::Elu { alpha: 1.0 }],
            &Op::ReduceSum {
                keepdims: false,
                noop_with_empty_axes: false,
            },
            DtypeRepr::F32,
            DEFAULT_REDUCE_FUSE_BLOCK_INNER,
        );
        let err = match result {
            Err(e) => e,
            Ok(_) => panic!("expected an error for an unsupported chain member"),
        };
        assert!(err.contains("unsupported member"));
    }

    #[test]
    fn lower_reduce_fuse_empty_chain_is_a_bare_reduction() {
        let (name, source, _entry_point, _rop) = lower_reduce_fuse(
            &[],
            &Op::ReduceSum {
                keepdims: false,
                noop_with_empty_axes: false,
            },
            DtypeRepr::F32,
            DEFAULT_REDUCE_FUSE_BLOCK_INNER,
        )
        .expect("empty chain is a valid, if trivial, ReduceFuse");
        assert_eq!(name, "reduce_fuse_sum");
        assert!(source.contains("let x = T :: load") || source.contains("T::load"));
    }

    // teenygrad-3w0.11: cost-driven BLOCK_INNER search.

    #[test]
    fn choose_reduce_fuse_block_inner_picks_next_power_of_two() {
        assert_eq!(choose_reduce_fuse_block_inner(1), Some(32));
        assert_eq!(choose_reduce_fuse_block_inner(32), Some(32));
        assert_eq!(choose_reduce_fuse_block_inner(33), Some(64));
        assert_eq!(choose_reduce_fuse_block_inner(100), Some(128));
        assert_eq!(choose_reduce_fuse_block_inner(129), Some(256));
        assert_eq!(choose_reduce_fuse_block_inner(1024), Some(1024));
    }

    #[test]
    fn choose_reduce_fuse_block_inner_declines_when_n_inner_exceeds_max_threads() {
        // ReduceFuse's single-shot load can't reach past BLOCK_INNER; a
        // reduction axis wider than the largest valid CTA size must decline
        // to fuse rather than silently drop elements.
        assert_eq!(choose_reduce_fuse_block_inner(1025), None);
        assert_eq!(choose_reduce_fuse_block_inner(100_000), None);
    }
}
