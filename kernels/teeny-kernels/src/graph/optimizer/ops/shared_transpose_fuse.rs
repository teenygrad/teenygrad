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

//! [`SharedTransposeFuse`] — a unary pointwise chain fused directly into a
//! rank-2 transpose.
//!
//! teenygrad-3w0.10's `SetConnect` demonstration case: `T::trans` (used by
//! `nn::tensor::transpose::transpose_2d_forward`) already lowers through a
//! real shared-memory stage-and-transpose internally (compiler-transparent,
//! the same way `tt.reduce` does — see `nn::tensor::transpose`'s module
//! doc), so splicing a unary chain directly in front of it — computed once
//! per tile in registers, never materialized to global memory — is exactly
//! Welder's shared-memory tier: neither a register-level splice (transpose
//! genuinely needs cross-thread data movement; the producer and consumer
//! touch the intermediate in different thread-to-element patterns) nor a
//! full global round-trip (the unfused baseline: a separate kernel for the
//! chain, then `transpose_2d_forward` reading its output back from global
//! memory).
//!
//! Structurally this mirrors [`super::ReduceFuse`] exactly: one freshly
//! synthesized kernel, `transpose_2d_forward`'s own load/trans/store shape
//! with each chain member's [`teeny_triton::FusionCore`] spliced in between
//! the load and the transpose. `member_fusion_core`/`splice_chain` are
//! reused directly from `reduce_fuse.rs` (`pub(super)`) rather than
//! duplicated.
//!
//! **v1 scope:** chain members are `Op::Relu`/`Op::Sigmoid`/`Op::Tanh`/
//! `Op::Exp` only (same set as `ReduceFuse`, and for the same reason: the
//! only zero-extra-param `fusion_core()`-capable ops). `BLOCK_M`/`BLOCK_N`
//! must both exactly divide `M`/`N` — `transpose_2d_forward`'s own
//! documented non-goal (a real `T::trans` + non-block-aligned
//! tensor-descriptor-store bug, found by direct hardware experimentation —
//! see that module's doc), inherited here unchanged.

use std::any::Any;
use std::sync::Arc;

use teeny_core::device::program::ArgVisitor;
use teeny_core::graph::{CustomOp, DtypeRepr, Op, Shape};
use teeny_core::model::{RawPtr, RuntimeOp};
use teeny_triton::{CostModel, FusionCore, KernelTileSpec, TensorTileSpec, TileAxisBinding};

use crate::graph::optimizer::ops::pointwise_fuse::dtype_name;

use super::reduce_fuse::{is_reduce_fuse_member, member_fusion_core, splice_chain};

/// teenygrad-3w0.4/.10's calibrated constants (RTX 5070, 48 SMs; shared-
/// memory budget from `CudaDeviceInfo::shared_mem_per_block`, queried
/// directly — see `tile.rs`'s `CostModel::shared_mem_budget_bytes` doc).
/// `shared_mem_occupancy_penalty` is a documented placeholder (same honesty
/// convention `CostModel::window_penalty` already uses) — not yet
/// independently calibrated against a real measured occupancy cliff, only
/// against the general "under-parallel launches measure slower" effect
/// `under_parallel_penalty` already captures.
const SHARED_TRANSPOSE_FUSE_COST_MODEL: CostModel = CostModel {
    sm_count: 48,
    saturation_ctas_per_sm: 8,
    under_parallel_penalty: 1.5,
    window_penalty: 1.2,
    shared_mem_budget_bytes: 49_152,
    shared_mem_occupancy_penalty: 2.0,
};

/// Static tile-shape declaration for `SharedTransposeFuse`'s synthesized
/// kernel, modeled on `transpose_2d_forward`'s real shape: `x_ptr` tiled
/// `[BLOCK_M, BLOCK_N]` over `[M, N]`, `y_ptr` tiled `[BLOCK_N, BLOCK_M]`
/// over `[N, M]` (the transposed output), no reduction axis.
const SHARED_TRANSPOSE_FUSE_TILE_SPEC: KernelTileSpec = KernelTileSpec {
    inputs: &[TensorTileSpec {
        param: "x_ptr",
        rank: 2,
        axes: &[
            TileAxisBinding {
                dim: 0,
                block_const: "BLOCK_M",
                extent_param: "M",
                window: None,
            },
            TileAxisBinding {
                dim: 1,
                block_const: "BLOCK_N",
                extent_param: "N",
                window: None,
            },
        ],
        reduction_axis: None,
        untiled_dims: &[],
    }],
    outputs: &[TensorTileSpec {
        param: "y_ptr",
        rank: 2,
        axes: &[
            TileAxisBinding {
                dim: 0,
                block_const: "BLOCK_N",
                extent_param: "N",
                window: None,
            },
            TileAxisBinding {
                dim: 1,
                block_const: "BLOCK_M",
                extent_param: "M",
                window: None,
            },
        ],
        reduction_axis: None,
        untiled_dims: &[],
    }],
    loop_spec: None,
};

/// Element size for `SharedTransposeFuse`'s supported dtypes (same set
/// `member_fusion_core`/`dtype_name` already restrict to).
pub(crate) fn elem_bytes(dtype: DtypeRepr) -> Option<usize> {
    match dtype {
        DtypeRepr::F32 => Some(4),
        DtypeRepr::F64 => Some(8),
        _ => None,
    }
}

/// Searches the candidate ladder `{16, 32, 64, 128}` x `{16, 32, 64, 128}`
/// for `(BLOCK_M, BLOCK_N)`, keeping only candidates that exactly divide
/// `m`/`n` (`transpose_2d_forward`'s alignment requirement — see the module
/// doc), and returns the minimum-cost one under
/// [`CostModel::penalized_traffic_with_shared_mem`]. Unlike
/// [`super::choose_reduce_fuse_block_inner`]'s fixed `[n_outer, 1, 1]` grid,
/// a transpose's grid genuinely varies with `(BLOCK_M, BLOCK_N)` (`n_ctas =
/// (m/BLOCK_M) * (n/BLOCK_N)`), so both `under_parallel_penalty` (fewer,
/// bigger tiles → fewer CTAs) and the new shared-memory term (bigger tiles →
/// more bytes/CTA → lower occupancy) trade off against each other for real
/// — this is genuinely Welder's `SubGraphTiling` search, not a
/// single-dimension special case.
///
/// Returns `None` when no candidate pair evenly divides both `m` and `n`.
pub(crate) fn choose_shared_transpose_fuse_block_size(
    m: i64,
    n: i64,
    elem_bytes: usize,
) -> Option<(i32, i32)> {
    const CANDIDATES: [i32; 4] = [16, 32, 64, 128];
    CANDIDATES
        .into_iter()
        .flat_map(|bm| CANDIDATES.into_iter().map(move |bn| (bm, bn)))
        .filter(|&(bm, bn)| m % i64::from(bm) == 0 && n % i64::from(bn) == 0)
        .map(|(bm, bn)| {
            let n_ctas = ((m / i64::from(bm)) * (n / i64::from(bn))).max(0) as u64;
            let shared_mem_bytes = bm as u64 * bn as u64 * elem_bytes as u64;
            let cost = SHARED_TRANSPOSE_FUSE_COST_MODEL.penalized_traffic_with_shared_mem(
                &SHARED_TRANSPOSE_FUSE_TILE_SPEC,
                elem_bytes,
                |name| match name {
                    "BLOCK_M" => i64::from(bm),
                    "BLOCK_N" => i64::from(bn),
                    "M" => m,
                    "N" => n,
                    other => panic!("unexpected SHARED_TRANSPOSE_FUSE_TILE_SPEC param {other}"),
                },
                n_ctas,
                shared_mem_bytes,
            );
            ((bm, bn), cost)
        })
        .min_by(|(_, a), (_, b)| a.total_cmp(b))
        .map(|(bm_bn, _)| bm_bn)
}

/// A unary pointwise chain fused directly into `transpose_2d_forward`'s own
/// kernel. See the module doc for v1 scope.
#[derive(Debug, Clone)]
pub struct SharedTransposeFuse {
    /// Unary chain applied to the input before the transpose (may be empty
    /// — a bare `transpose(x)` with no chain is still a valid, if trivial,
    /// `SharedTransposeFuse`).
    pub chain: Vec<Op>,
    /// Element dtype shared by every chain member and the transpose.
    pub dtype: DtypeRepr,
    /// CTA tile shape for the fused kernel — teenygrad-3w0.10's cost-driven
    /// choice, from [`choose_shared_transpose_fuse_block_size`].
    pub block_m: i32,
    pub block_n: i32,
}

impl SharedTransposeFuse {
    /// Builds a fuse of `chain` (every member must satisfy
    /// [`is_reduce_fuse_member`]) at the given `(block_m, block_n)` (must
    /// both exactly divide the real `M`/`N` — see
    /// [`choose_shared_transpose_fuse_block_size`]'s doc comment for why).
    pub fn new(chain: Vec<Op>, dtype: DtypeRepr, block_m: i32, block_n: i32) -> Self {
        debug_assert!(chain.iter().all(is_reduce_fuse_member));
        Self {
            chain,
            dtype,
            block_m,
            block_n,
        }
    }
}

impl CustomOp for SharedTransposeFuse {
    fn name(&self) -> &str {
        "shared_transpose_fuse"
    }

    fn infer_output_shape(&self, input_shapes: &[&Shape]) -> Shape {
        // Same convention as the free `infer_output_shape`'s `Op::Transpose`
        // arm: rank-2 reverse.
        input_shapes[0].iter().rev().cloned().collect()
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn lower(&self) -> Option<(String, String, String, Arc<dyn RuntimeOp>)> {
        match lower_shared_transpose_fuse(&self.chain, self.dtype, self.block_m, self.block_n) {
            Ok(v) => Some(v),
            Err(e) => panic!("SharedTransposeFuse::lower failed: {e}"),
        }
    }
}

/// Runtime ABI: `x_ptr, y_ptr, M, N` — no scratch buffers, one synthesized
/// kernel (same shape as [`super::reduce_fuse`]'s `ReduceFuseRuntimeOp`).
/// Mirrors `nn::tensor::transpose::TransposeRuntimeOp`'s `pack_args`/`grid`
/// exactly, parameterized by this fuse's own chosen `(block_m, block_n)`.
struct SharedTransposeFuseRuntimeOp {
    block_m: i32,
    block_n: i32,
}

impl RuntimeOp for SharedTransposeFuseRuntimeOp {
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
        _output_shape: &[usize],
        _output_row_stride: i32,
        visitor: &mut dyn ArgVisitor,
    ) {
        // x: [M, N], y (output): [N, M].
        let m = inputs[0].1.first().copied().unwrap_or(1) as i32;
        let n = inputs[0].1.get(1).copied().unwrap_or(1) as i32;
        visitor.visit_ptr(inputs[0].0);
        visitor.visit_ptr(output);
        visitor.visit_i32(m);
        visitor.visit_i32(n);
    }

    fn grid(&self, output_shape: &[usize]) -> [u32; 3] {
        // output_shape is [N, M]; the input's [M, N] is what BLOCK_M/BLOCK_N tile.
        let n = output_shape.first().copied().unwrap_or(1) as u32;
        let m = output_shape.get(1).copied().unwrap_or(1) as u32;
        let pm = m.div_ceil(self.block_m as u32);
        let pn = n.div_ceil(self.block_n as u32);
        [pm * pn, 1, 1]
    }
}

/// `transpose_2d_forward`'s own load prelude, transcribed verbatim
/// (`nn::tensor::transpose`), up to and including the input tile load bound
/// to `x` (the identifier `splice_chain`'s spliced-in members expect).
const SHARED_TRANSPOSE_FUSE_LOAD_PRELUDE: &str = "    let pid = T::program_id(Axis::X);\n\
    let num_pid_n = T::cdiv(N, BLOCK_N);\n\
    let pid_m = pid / num_pid_n;\n\
    let pid_n = pid % num_pid_n;\n\
    let x_desc = T::make_tensor_descriptor(\n\
    \x20   x_ptr,\n\
    \x20   &[M, N],\n\
    \x20   &[N, 1],\n\
    \x20   &[BLOCK_M, BLOCK_N],\n\
    \x20   Some(PaddingOption::Zero),\n\
    );\n\
    let x = T::load_tensor_descriptor(x_desc, &[pid_m * BLOCK_M, pid_n * BLOCK_N]);\n";

/// `transpose_2d_forward`'s own trans + store, transcribed verbatim, reading
/// the (possibly chain-spliced) `x`.
const SHARED_TRANSPOSE_FUSE_EPILOGUE: &str = "    let tile_t = T::trans(x, &[1, 0]);\n\
    let y_desc = T::make_tensor_descriptor(\n\
    \x20   y_ptr,\n\
    \x20   &[N, M],\n\
    \x20   &[M, 1],\n\
    \x20   &[BLOCK_N, BLOCK_M],\n\
    \x20   Some(PaddingOption::Zero),\n\
    );\n\
    T::store_tensor_descriptor(y_desc, &[pid_n * BLOCK_N, pid_m * BLOCK_M], tile_t);\n";

fn shared_transpose_fuse_kernel_source(fn_name: &str, member_cores: &[FusionCore]) -> String {
    format!(
        "pub fn {fn_name}<T: Triton, D: Float, const BLOCK_M: i32, const BLOCK_N: i32>(\n\
         \x20   x_ptr: T::Pointer<D>,\n\
         \x20   y_ptr: T::Pointer<D>,\n\
         \x20   M: i32,\n\
         \x20   N: i32,\n\
         ) {{\n\
         {SHARED_TRANSPOSE_FUSE_LOAD_PRELUDE}\
         {spliced}\
         {SHARED_TRANSPOSE_FUSE_EPILOGUE}\
         }}\n",
        spliced = splice_chain(member_cores),
    )
}

fn synthesize_shared_transpose_fuse_entry(
    entry_name: &str,
    dtype: &str,
    fn_name: &str,
    block_m: i32,
    block_n: i32,
) -> String {
    format!(
        concat!(
            "use triton::llvm::triton::num::*;\n",
            "use triton::llvm::triton::pointer::LlvmPointer;\n",
            "type LlvmTriton = triton::llvm::triton::LlvmTriton;\n",
            "\n",
            "#[no_mangle]\n",
            "pub extern \"C\" fn {entry}(x_ptr: *mut {dtype}, y_ptr: *mut {dtype}, M: i32, N: i32) {{\n",
            "    let x_ptr = LlvmPointer(x_ptr as *mut _);\n",
            "    let y_ptr = LlvmPointer(y_ptr as *mut _);\n",
            "    {fn_name}::<LlvmTriton, {dtype}, {block_m}, {block_n}>(x_ptr, y_ptr, M, N);\n",
            "}}"
        ),
        entry = entry_name,
        dtype = dtype,
        fn_name = fn_name,
        block_m = block_m,
        block_n = block_n,
    )
}

fn lower_shared_transpose_fuse(
    chain: &[Op],
    dtype: DtypeRepr,
    block_m: i32,
    block_n: i32,
) -> Result<(String, String, String, Arc<dyn RuntimeOp>), String> {
    if !chain.iter().all(is_reduce_fuse_member) {
        return Err(format!(
            "SharedTransposeFuse chain has an unsupported member: {chain:?}"
        ));
    }

    let dtype_str = dtype_name(dtype)?;

    let member_cores: Vec<FusionCore> = chain
        .iter()
        .map(|op| {
            member_fusion_core(op, dtype).ok_or_else(|| {
                format!("no SharedTransposeFuse fusion_core for member op={op:?} dtype={dtype:?}")
            })
        })
        .collect::<Result<_, _>>()?;

    let tag = chain
        .iter()
        .map(|op| format!("{op:?}").to_lowercase())
        .chain(std::iter::once("transpose".to_string()))
        .collect::<Vec<_>>()
        .join("_");
    let fused_name = format!("shared_transpose_fuse_{tag}");
    let entry_point = format!("{fused_name}_entry_point");

    let kernel_source = shared_transpose_fuse_kernel_source(&fused_name, &member_cores);
    let entry = synthesize_shared_transpose_fuse_entry(
        &entry_point,
        dtype_str,
        &fused_name,
        block_m,
        block_n,
    );
    let full_source = format!("{kernel_source}\n\n{entry}");

    Ok((
        fused_name,
        full_source,
        entry_point,
        Arc::new(SharedTransposeFuseRuntimeOp { block_m, block_n }) as Arc<dyn RuntimeOp>,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lower_shared_transpose_fuse_relu_splices_relu_core_before_trans() {
        let (name, source, entry_point, _rop) =
            lower_shared_transpose_fuse(&[Op::Relu], DtypeRepr::F32, 32, 32)
                .expect("relu -> transpose is SharedTransposeFuse's canonical shape");

        assert_eq!(name, "shared_transpose_fuse_relu_transpose");
        assert_eq!(
            entry_point,
            "shared_transpose_fuse_relu_transpose_entry_point"
        );
        assert!(source.contains("T :: maximum"), "relu's core spliced in");
        assert!(
            source.contains("T::trans(x") || source.contains("T :: trans(x"),
            "trans reads the spliced value back as `x`"
        );
        assert!(source.contains(&entry_point));
    }

    #[test]
    fn lower_shared_transpose_fuse_rejects_unsupported_chain_member() {
        let result = lower_shared_transpose_fuse(&[Op::Elu { alpha: 1.0 }], DtypeRepr::F32, 32, 32);
        let err = match result {
            Err(e) => e,
            Ok(_) => panic!("expected an error for an unsupported chain member"),
        };
        assert!(err.contains("unsupported member"));
    }

    #[test]
    fn lower_shared_transpose_fuse_empty_chain_is_a_bare_transpose() {
        let (name, source, _entry_point, _rop) =
            lower_shared_transpose_fuse(&[], DtypeRepr::F32, 32, 32)
                .expect("empty chain is a valid, if trivial, SharedTransposeFuse");
        assert_eq!(name, "shared_transpose_fuse_transpose");
        assert!(
            source.contains("T::load_tensor_descriptor")
                || source.contains("T :: load_tensor_descriptor")
        );
    }

    // teenygrad-3w0.10: cost-driven (BLOCK_M, BLOCK_N) search.

    #[test]
    fn choose_shared_transpose_fuse_block_size_only_considers_exact_divisors() {
        // 100 isn't a multiple of any {16,32,64,128} candidate, so no
        // (BLOCK_M, *) pair can be chosen for m=100 -- must decline, not
        // silently pick a misaligned tile (transpose_2d_forward's
        // documented correctness requirement).
        assert_eq!(choose_shared_transpose_fuse_block_size(100, 128, 4), None);
        assert_eq!(choose_shared_transpose_fuse_block_size(128, 100, 4), None);
    }

    #[test]
    fn choose_shared_transpose_fuse_block_size_finds_an_aligned_candidate() {
        let (bm, bn) = choose_shared_transpose_fuse_block_size(128, 256, 4)
            .expect("128 and 256 are both covered by the {16,32,64,128} ladder");
        assert_eq!(128 % bm, 0);
        assert_eq!(256 % bn, 0);
    }

    #[test]
    fn choose_shared_transpose_fuse_block_size_avoids_shared_mem_over_budget() {
        // 128*128*4 = 65536 bytes > 49152-byte budget for ONE CTA already --
        // the search should still return *something* (the penalty biases
        // the ranking, it doesn't reject candidates outright), but a smaller
        // tile should be preferred when it's also perfectly aligned.
        let (bm, bn) = choose_shared_transpose_fuse_block_size(128, 128, 4)
            .expect("128 is covered by the {16,32,64,128} ladder");
        assert!(
            (bm as u64) * (bn as u64) * 4 <= 49_152,
            "expected the search to prefer a tile that fits the shared-memory budget, got {bm}x{bn}"
        );
    }
}
