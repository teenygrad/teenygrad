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

//! Static tile-shape layout of a `#[tiled_kernel]` function (host-side only).
//!
//! Derived by `teeny_macros::tiled_kernel` from `#[tile(...)]`-annotated pointer
//! params. Lives outside `triton/` so it is not embedded into the no_core DSL
//! string, mirroring [`crate::kernel_io`].
//!
//! Phase 1a only: a single tile axis per tensor (`rank == 1`), bound to a
//! `BLOCK_*` compile-time const and an `i32` extent parameter. This is the
//! queryable metadata a future Welder-style (OSDI'23) tile-graph scheduler
//! would use for `Propagate`/`MemTraffic` — see teenygrad-3w0.

use std::collections::BTreeMap;

/// Strided/padded window relating an axis's *output* tile to the actual
/// *input* positions it reads — e.g. `conv2d_forward`'s `x_ptr`, whose
/// per-output-tile input region is `(block-1)*stride + kernel_size`
/// elements wide, not simply `block` (Welder's "conservative upper bound"
/// treatment for Gather/Conv-shaped irregular access — see teenygrad-3w0's
/// notes on `Propagate`). `block_const`/`extent_param` on the owning
/// [`TileAxisBinding`] still name this axis's *own* block/extent (e.g.
/// `x_ptr`'s `W`); `block_const` is shared with the driving output axis
/// (e.g. `y_ptr`'s `BLOCK_OW`) since the input side has no independent tile
/// size of its own.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TileWindow {
    /// Name of the `const {NAME}: i32` generic giving this axis's stride.
    pub stride_const: &'static str,
    /// Name of the `const {NAME}: i32` generic giving this axis's symmetric
    /// padding (matches every conv kernel in this codebase's `PAD_*`
    /// convention — applied equally on both sides).
    pub pad_const: &'static str,
    /// Name of the `const {NAME}: i32` generic giving this axis's kernel
    /// (receptive-field) size.
    pub kernel_size_const: &'static str,
}

/// One tensor axis's tile binding: which compile-time block size and runtime
/// extent parameter it's sliced by.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TileAxisBinding {
    /// Axis index within the tensor's total shape.
    pub dim: usize,
    /// Name of the `const {NAME}: i32` generic providing this axis's tile size.
    pub block_const: &'static str,
    /// Name of the `{NAME}: i32` kernel parameter providing this axis's total extent.
    pub extent_param: &'static str,
    /// `Some` when this axis is read through a strided/padded sliding
    /// window rather than a plain contiguous `arange(block)+pid*block`
    /// slice (teenygrad-3w0.5).
    pub window: Option<TileWindow>,
}

/// Tile-shape metadata for one pointer parameter.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TensorTileSpec {
    /// The parameter's name, e.g. `"x_ptr"`.
    pub param: &'static str,
    /// The tensor's rank. Phase 1a requires `rank == axes.len() == 1`.
    pub rank: usize,
    /// Per-axis tile bindings, in tensor-axis order.
    pub axes: &'static [TileAxisBinding],
    /// Axis index this tensor is reduced/accumulated over, if any.
    ///
    /// Always `None` in phase 1a (no reduction loops yet); wired up in
    /// phase 1b (teenygrad-3w0.2) for GEMM-style kernels.
    pub reduction_axis: Option<usize>,
    /// Names of `{NAME}: i32` params giving the sizes of this tensor's
    /// *other* real dimensions — present in memory, but not individually
    /// tiled by a `#[tile(...)]` axis (teenygrad-3w0.8). E.g.
    /// `conv2d_forward`'s `y_ptr` is tagged on its `OW` axis alone;
    /// `B`/`C_OUT`/`OH` are real but grid-driven, so they belong here, not
    /// in `axes`. Fixes the gap teenygrad-3w0.4 found: [`mem_traffic`]
    /// previously omitted these entirely from the byte count.
    pub untiled_dims: &'static [&'static str],
}

/// One loop-carried accumulator variable threaded through a kernel's
/// reduction/online loop (e.g. flash-attn's online-softmax state:
/// `acc`/`m_i`/`l_i`, each re-bound every iteration and read again after the
/// loop). Not representable as a [`TensorTileSpec`]: these aren't tiles of a
/// pointer parameter sliced by `(block, extent)`, they're kernel-body-local
/// tensors whose *shape* is fixed per const generics (not grid-varying) and
/// whose *value* the loop updates in place.
///
/// This records shape and identity only — not the per-iteration combine
/// expression (flash-attn's `m_new = max(m_i, qk)` etc. stays exactly as
/// written in the kernel body). Representing the combine semantics
/// themselves as data would need a small expression AST, out of scope here
/// (teenygrad-3w0.7); this primitive exists so a future consumer (e.g.
/// `Propagate`, teenygrad-3w0.8) can at least see that these variables
/// exist, what shape they are, and that they're loop-carried, rather than
/// the kernel being entirely opaque to tile-graph analysis.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TileCarryBinding {
    /// The variable's name in the kernel body, e.g. `"acc"`.
    pub name: &'static str,
    /// Names of the `const {NAME}: i32` generics giving the carried
    /// tensor's shape, in dimension order (e.g. `["HEAD_DIM"]`).
    pub shape_consts: &'static [&'static str],
}

/// Loop-carry metadata for a kernel's online/reduction loop (teenygrad-3w0.7).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TileLoopSpec {
    /// Loop-carried accumulators, in the order they're declared.
    pub carries: &'static [TileCarryBinding],
    /// Name of the `{NAME}: i32` kernel parameter giving the loop's trip
    /// count (e.g. `"n_ctx_k"`).
    pub trip_count_param: &'static str,
}

/// Tile-shape metadata for a kernel's full pointer-parameter set.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KernelTileSpec {
    /// Tile specs for `#[tile(...)]`-annotated input ([`crate::triton::In`]) params.
    pub inputs: &'static [TensorTileSpec],
    /// Tile specs for `#[tile(...)]`-annotated output ([`crate::triton::Out`]) params.
    pub outputs: &'static [TensorTileSpec],
    /// Loop-carry metadata from a `#[tile_loop(...)]` attribute on the
    /// kernel fn, if present (teenygrad-3w0.7).
    pub loop_spec: Option<TileLoopSpec>,
}

/// Tile-shape metadata from a `#[tiled_kernel]` signature's `#[tile(...)]` attributes.
///
/// `#[tiled_kernel]` implements this for every generated kernel struct that has at
/// least one `#[tile(...)]`-annotated pointer parameter.
pub trait TileSpecLayout {
    /// Declared tile shape of this kernel's pointer parameters.
    fn tile_spec() -> KernelTileSpec;
}

/// Analytic memory-traffic estimate for a kernel's declared tile shape:
/// total bytes moved to cover every `#[tile(...)]`-tagged tensor, summed as
/// `tile_bytes * num_tiles` per tensor (Welder's cost model, OSDI'23 §3.1).
/// Penalty factors for coalescing/under-parallelism are applied separately
/// by [`CostModel::penalized_traffic`], not baked into this function.
///
/// `resolve` maps a named `block_const`/`extent_param` to its concrete
/// runtime value: `KernelTileSpec` only records parameter *names* (the
/// actual sizes depend on which graph node / launch config a kernel is
/// instantiated with, not on its static declaration), so the caller supplies
/// the lookup — e.g. from a `RuntimeOp`'s resolved output shape and the
/// kernel struct's own const-generic fields.
///
/// An `InOut` tensor appears in both `spec.inputs` and `spec.outputs`
/// (see `#[tiled_kernel]`'s codegen), so its traffic is correctly counted twice —
/// once for the read, once for the write.
///
/// Multiplies in `tensor.untiled_dims` too (teenygrad-3w0.8) — real
/// dimensions a tensor has that aren't individually tiled (e.g.
/// `conv2d_forward`'s `y_ptr` only tags its `OW` axis; `B`/`C_OUT`/`OH` are
/// grid-driven but still real). Before this field existed, `mem_traffic`
/// silently omitted such dimensions from the byte count entirely — this
/// was discovered calibrating [`CostModel`] against real hardware in
/// teenygrad-3w0.4, where it made `conv2d_forward`'s estimate several
/// orders of magnitude too small (see that commit / the calibration test's
/// history for the concrete numbers).
pub fn mem_traffic(
    spec: &KernelTileSpec,
    elem_bytes: usize,
    mut resolve: impl FnMut(&str) -> i64,
) -> u64 {
    let mut total: u64 = 0;
    for tensor in spec.inputs.iter().chain(spec.outputs.iter()) {
        let mut tile_elems: u64 = 1;
        let mut n_tiles: u64 = 1;
        for axis in tensor.axes {
            let block = resolve(axis.block_const).max(1) as u64;
            let extent = resolve(axis.extent_param).max(0) as u64;
            // A windowed axis's *input* footprint per output tile is the
            // conservative upper bound `(block-1)*stride + kernel_size`, not
            // `block` — e.g. conv's x_ptr reads more input elements than the
            // output tile is wide once STRIDE_W/KW/PAD_W are non-trivial.
            let footprint = match axis.window {
                Some(w) => {
                    let stride = resolve(w.stride_const).max(1) as u64;
                    let kernel_size = resolve(w.kernel_size_const).max(1) as u64;
                    (block.saturating_sub(1)) * stride + kernel_size
                }
                None => block,
            };
            tile_elems *= footprint;
            n_tiles *= extent.div_ceil(block);
        }
        // Untiled-but-real dimensions (teenygrad-3w0.8): each CTA re-reads/
        // writes its tile once per combination of these — e.g. conv2d's
        // per-(b, c_out, oh) CTAs each touch their own full-width tile.
        for dim in tensor.untiled_dims {
            n_tiles *= resolve(dim).max(1) as u64;
        }
        total += tile_elems * n_tiles * elem_bytes as u64;
    }
    total
}

/// Welder's `Propagate` (OSDI'23 §3.1), scoped to one kernel's own declared
/// tile shape: resolve as many `extent_param` names as possible from a
/// chosen output tile shape, by matching names shared across `spec`'s own
/// inputs and outputs.
///
/// This needs no new attribute syntax to connect axes across different
/// tensors — `matmul_forward`'s existing `#[tile(...)]` tags already declare
/// `a_ptr`'s axes as `extent = [M, K]` and `c_ptr`'s as `extent = [M, N]`,
/// sharing the literal string `"M"` because both are written against the
/// same kernel's own `M: i32` parameter. Propagation is exactly: seed the
/// output's axis values, then any axis anywhere in `spec` whose
/// `extent_param` matches an already-known name is now known too.
///
/// `output_param` selects which entry of `spec.outputs` `chosen_output`
/// describes (by [`TensorTileSpec::param`]); `chosen_output` gives one
/// concrete size per axis, in that tensor's declared axis order.
///
/// Returns every resolved `extent_param` name -> concrete size, including
/// the seed. An axis whose `extent_param` never appears in the resolved set
/// (e.g. GEMM's reduction axis `"K"`, absent from `c_ptr`) is simply absent
/// from the result — not derivable from the output alone, matching Welder's
/// model where reduction-axis tile size is chosen independently by the
/// tiling search (`teenygrad-3w0.9`/`.11`), not propagated top-down.
/// Callers needing a concrete value for an unresolved axis must supply one
/// themselves (e.g. the kernel's own declared `BLOCK_*` default).
///
/// Panics if `output_param` doesn't name one of `spec.outputs`, or if
/// `chosen_output.len()` doesn't match that tensor's axis count — both
/// caller bugs, not data-dependent failures.
pub fn propagate_within_kernel(
    spec: &KernelTileSpec,
    output_param: &str,
    chosen_output: &[i64],
) -> BTreeMap<&'static str, i64> {
    let output_tensor = spec
        .outputs
        .iter()
        .find(|t| t.param == output_param)
        .unwrap_or_else(|| {
            panic!("`{output_param}` is not a declared output of this KernelTileSpec")
        });
    assert_eq!(
        output_tensor.axes.len(),
        chosen_output.len(),
        "chosen_output has {} entries but `{output_param}` has {} axes",
        chosen_output.len(),
        output_tensor.axes.len()
    );

    // Seed from the chosen output's own axes. That's the entire algorithm:
    // any other tensor's axis becomes "resolved" simply by sharing one of
    // these `extent_param` names — there's no further value to derive per
    // axis (an axis's name isn't itself a value to propagate transitively),
    // so a single pass is complete. Callers check `resolved.get(name)` for
    // any axis, input or output; a miss means that axis's extent isn't
    // determined by this output shape (e.g. a reduction axis).
    output_tensor
        .axes
        .iter()
        .zip(chosen_output)
        .map(|(axis, &size)| (axis.extent_param, size))
        .collect()
}

/// Calibrated penalty factors for [`mem_traffic`]'s analytic estimate,
/// derived from real measurements on this project's target hardware
/// (teenygrad-3w0.4) rather than assumed from Welder's paper constants
/// (which were measured on V100/MI50/IPU, not whatever GPU this project
/// actually targets).
///
/// Two effects are modeled, each a flat multiplier on the raw
/// [`mem_traffic`] estimate, at very different confidence levels:
/// - **Under-parallelism** (`under_parallel_penalty`): a launch with too few
///   CTAs to keep every SM busy can't hide memory latency behind other
///   warps' work, so achieved bandwidth is lower than the analytic estimate
///   assumes. Cleanly isolated and confirmed on real hardware — the same
///   kernel (`elu_forward`) at the same `BLOCK_SIZE`, varying only CTA
///   count, showed >10x higher achieved bandwidth once comfortably
///   oversubscribed vs. ~1 CTA/SM.
/// - **Windowed access** (`window_penalty`): intended to model a
///   [`TileWindow`]-bound axis (e.g. conv's strided, padded,
///   overlapping-receptive-field reads) achieving lower bandwidth than a
///   plain contiguous access. **Still not cleanly calibrated**, even after
///   teenygrad-3w0.8 fixed `mem_traffic`'s multi-dim gap (`untiled_dims`):
///   the only real-hardware comparison available (`conv2d_forward` vs
///   `elu_forward`) still can't isolate coalescing cost specifically from
///   `conv2d_forward`'s much higher compute-per-element, so this field
///   remains a conservative placeholder, not a derived constant. A real
///   isolation would need a purpose-built strided-vs-contiguous kernel pair
///   differing in nothing else — future work if this needs tightening.
///
/// See `kernels/teeny-kernels/tests/test_mem_traffic_calibration.rs` for the
/// wall-clock-timed (not CUDA-event; `device.launch` already synchronizes
/// internally, see that file's module doc) experiments this was calibrated
/// from.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CostModel {
    /// The target GPU's streaming-multiprocessor count (from
    /// `CudaDeviceInfo::multi_processor_count` — real hardware data, not a
    /// guess).
    pub sm_count: u32,
    /// CTAs-per-SM below which a launch is considered under-parallel. `1`
    /// means "fewer CTAs than SMs is the only penalized case"; higher values
    /// require more concurrent CTAs per SM to count as fully parallel.
    pub saturation_ctas_per_sm: u32,
    /// Multiplier applied when `n_ctas < sm_count * saturation_ctas_per_sm`.
    pub under_parallel_penalty: f64,
    /// Multiplier applied when any tagged tensor has a [`TileWindow`] axis.
    pub window_penalty: f64,
    /// Per-SM *static* shared-memory budget, in bytes (real hardware fact —
    /// `CudaDeviceInfo::shared_mem_per_block`; teenygrad-3w0.10's RTX 5070
    /// calibration: `49152` = 48 KiB). A kernel that stages a tile through
    /// shared memory (e.g. a shared-memory-fused transpose) competes with
    /// every other co-resident CTA on the same SM for this budget — a
    /// larger tile means fewer CTAs/SM can run concurrently, independent of
    /// `under_parallel_penalty` (which prices grid size, not per-SM
    /// occupancy from a resource constraint).
    pub shared_mem_budget_bytes: u64,
    /// Multiplier applied when a candidate tile's shared-memory footprint
    /// would drop co-resident occupancy below `saturation_ctas_per_sm`
    /// CTAs/SM (see [`CostModel::penalized_traffic_with_shared_mem`]).
    pub shared_mem_occupancy_penalty: f64,
}

impl CostModel {
    /// [`mem_traffic`]'s raw byte estimate, scaled by whichever calibrated
    /// penalties apply to this launch. `n_ctas` is the actual grid size (CTA
    /// count) the kernel will be launched with — the caller's `RuntimeOp::grid()`
    /// or equivalent, since `KernelTileSpec` alone doesn't determine it.
    pub fn penalized_traffic(
        &self,
        spec: &KernelTileSpec,
        elem_bytes: usize,
        resolve: impl FnMut(&str) -> i64,
        n_ctas: u64,
    ) -> f64 {
        let raw = mem_traffic(spec, elem_bytes, resolve) as f64;
        let mut factor = 1.0;
        if n_ctas < self.sm_count as u64 * self.saturation_ctas_per_sm as u64 {
            factor *= self.under_parallel_penalty;
        }
        let has_window = spec
            .inputs
            .iter()
            .chain(spec.outputs.iter())
            .any(|t| t.axes.iter().any(|a| a.window.is_some()));
        if has_window {
            factor *= self.window_penalty;
        }
        raw * factor
    }

    /// Like [`CostModel::penalized_traffic`], but also prices in a
    /// candidate's shared-memory footprint (teenygrad-3w0.10) —
    /// `shared_mem_bytes` is the actual bytes-per-CTA a candidate tile
    /// stages through shared memory (e.g. `BLOCK_M * BLOCK_N * elem_bytes`
    /// for a transpose staging buffer). `shared_mem_budget_bytes /
    /// shared_mem_bytes` gives the number of CTAs that can be co-resident on
    /// one SM purely from the shared-memory constraint; if that's below
    /// `saturation_ctas_per_sm`, [`Self::shared_mem_occupancy_penalty`]
    /// applies, stacking multiplicatively with whatever
    /// [`Self::penalized_traffic`] already applied (a tile can be both
    /// under-parallel *and* shared-memory-constrained at once — these are
    /// different causes of the same underlying effect, low occupancy, and
    /// there's no calibration yet distinguishing how much each contributes
    /// when both apply simultaneously, so they're treated as independent
    /// multipliers rather than deduplicated).
    pub fn penalized_traffic_with_shared_mem(
        &self,
        spec: &KernelTileSpec,
        elem_bytes: usize,
        resolve: impl FnMut(&str) -> i64,
        n_ctas: u64,
        shared_mem_bytes: u64,
    ) -> f64 {
        let base = self.penalized_traffic(spec, elem_bytes, resolve, n_ctas);
        let ctas_per_sm_by_smem = self.shared_mem_budget_bytes / shared_mem_bytes.max(1);
        if ctas_per_sm_by_smem < self.saturation_ctas_per_sm as u64 {
            base * self.shared_mem_occupancy_penalty
        } else {
            base
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const X: TensorTileSpec = TensorTileSpec {
        param: "x_ptr",
        rank: 1,
        axes: &[TileAxisBinding {
            dim: 0,
            block_const: "BLOCK_SIZE",
            extent_param: "n_elements",
            window: None,
        }],
        reduction_axis: None,
        untiled_dims: &[],
    };
    const Y: TensorTileSpec = TensorTileSpec {
        param: "y_ptr",
        ..X
    };

    #[test]
    fn mem_traffic_flat_elementwise_covers_input_and_output() {
        // n_elements=1000, BLOCK_SIZE=256 -> ceil(1000/256)=4 tiles of 256
        // elements each, for both x (input) and y (output), at 4 bytes/elem.
        let spec = KernelTileSpec {
            inputs: &[X],
            outputs: &[Y],
            loop_spec: ::core::option::Option::None,
        };
        let resolve = |name: &str| match name {
            "BLOCK_SIZE" => 256,
            "n_elements" => 1000,
            other => panic!("unexpected param {other}"),
        };
        let traffic = mem_traffic(&spec, 4, resolve);
        assert_eq!(traffic, 2 * 4 * 256 * 4);
    }

    #[test]
    fn mem_traffic_double_counts_inout_tensor() {
        // An InOut tensor is present in both `inputs` and `outputs` (see
        // #[tiled_kernel]'s codegen for PtrArgKind::InOut) -- traffic must reflect
        // both the read and the write, not just one.
        let spec = KernelTileSpec {
            inputs: &[X],
            outputs: &[X],
            loop_spec: ::core::option::Option::None,
        };
        let resolve = |name: &str| match name {
            "BLOCK_SIZE" => 128,
            "n_elements" => 128,
            other => panic!("unexpected param {other}"),
        };
        let single = 128u64 * 4;
        assert_eq!(mem_traffic(&spec, 4, resolve), 2 * single);
    }

    #[test]
    fn mem_traffic_multi_axis_multiplies_across_axes() {
        // GEMM-shaped: [BLOCK_M, BLOCK_K] tile over [M, K] = [64, 64] extent
        // with BLOCK_M=BLOCK_K=32 -> 2*2=4 tiles of 32*32 elements each.
        const A: TensorTileSpec = TensorTileSpec {
            param: "a_ptr",
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
                    block_const: "BLOCK_K",
                    extent_param: "K",
                    window: None,
                },
            ],
            reduction_axis: Some(1),
            untiled_dims: &[],
        };
        let spec = KernelTileSpec {
            inputs: &[A],
            outputs: &[],
            loop_spec: ::core::option::Option::None,
        };
        let resolve = |name: &str| match name {
            "BLOCK_M" | "BLOCK_K" => 32,
            "M" | "K" => 64,
            other => panic!("unexpected param {other}"),
        };
        assert_eq!(mem_traffic(&spec, 4, resolve), 4 * 32 * 32 * 4);
    }

    fn resolve_elu(name: &str) -> i64 {
        match name {
            "BLOCK_SIZE" => 1024,
            "n_elements" => 1_000_000,
            other => panic!("unexpected param {other}"),
        }
    }

    #[test]
    fn cost_model_penalizes_under_parallel_launch() {
        let spec = KernelTileSpec {
            inputs: &[X],
            outputs: &[Y],
            loop_spec: ::core::option::Option::None,
        };
        let cost_model = CostModel {
            sm_count: 64,
            saturation_ctas_per_sm: 4,
            under_parallel_penalty: 2.0,
            window_penalty: 1.5,
            shared_mem_budget_bytes: 49_152,
            shared_mem_occupancy_penalty: 1.0,
        };
        let raw = mem_traffic(&spec, 4, resolve_elu) as f64;

        // 64*4 = 256 CTAs needed to saturate; 10 CTAs is well under that.
        let under = cost_model.penalized_traffic(&spec, 4, resolve_elu, 10);
        assert_eq!(under, raw * 2.0);

        // 1000 CTAs comfortably saturates every SM.
        let saturated = cost_model.penalized_traffic(&spec, 4, resolve_elu, 1000);
        assert_eq!(saturated, raw);
    }

    #[test]
    fn cost_model_penalizes_windowed_tensor() {
        const WINDOWED: TensorTileSpec = TensorTileSpec {
            param: "x_ptr",
            rank: 1,
            axes: &[TileAxisBinding {
                dim: 0,
                block_const: "BLOCK_OW",
                extent_param: "W",
                window: Some(TileWindow {
                    stride_const: "STRIDE_W",
                    pad_const: "PAD_W",
                    kernel_size_const: "KW",
                }),
            }],
            reduction_axis: None,
            untiled_dims: &[],
        };
        let spec = KernelTileSpec {
            inputs: &[WINDOWED],
            outputs: &[],
            loop_spec: ::core::option::Option::None,
        };
        let resolve = |name: &str| match name {
            "BLOCK_OW" => 32,
            "W" => 320,
            "STRIDE_W" => 1,
            "PAD_W" => 1,
            "KW" => 3,
            other => panic!("unexpected param {other}"),
        };
        let cost_model = CostModel {
            sm_count: 1,
            saturation_ctas_per_sm: 1,
            under_parallel_penalty: 1.0,
            window_penalty: 1.7,
            shared_mem_budget_bytes: 49_152,
            shared_mem_occupancy_penalty: 1.0,
        };
        let raw = mem_traffic(&spec, 4, resolve) as f64;
        // 100 CTAs saturates a 1-SM device, so only the window penalty fires.
        assert_eq!(
            cost_model.penalized_traffic(&spec, 4, resolve, 100),
            raw * 1.7
        );
    }

    #[test]
    fn cost_model_penalizes_shared_mem_constrained_tile() {
        let spec = KernelTileSpec {
            inputs: &[X],
            outputs: &[Y],
            loop_spec: None,
        };
        let cost_model = CostModel {
            sm_count: 64,
            saturation_ctas_per_sm: 4,
            under_parallel_penalty: 1.0,
            window_penalty: 1.0,
            shared_mem_budget_bytes: 49_152,
            shared_mem_occupancy_penalty: 3.0,
        };
        // Comfortably many CTAs, so `under_parallel_penalty` never fires --
        // isolates the shared-memory term.
        let raw = cost_model.penalized_traffic(&spec, 4, resolve_elu, 1_000_000);

        // 49152 / 4096 = 12 CTAs/SM by the shared-memory budget alone --
        // above saturation_ctas_per_sm=4, so no penalty.
        let roomy =
            cost_model.penalized_traffic_with_shared_mem(&spec, 4, resolve_elu, 1_000_000, 4096);
        assert_eq!(roomy, raw);

        // 49152 / 20000 = 2 CTAs/SM by the shared-memory budget -- below
        // saturation_ctas_per_sm=4, so the penalty fires.
        let cramped =
            cost_model.penalized_traffic_with_shared_mem(&spec, 4, resolve_elu, 1_000_000, 20_000);
        assert_eq!(cramped, raw * 3.0);
    }

    /// GEMM-shaped spec (`a_ptr: [M,K]`, `b_ptr: [K,N]`, `c_ptr: [M,N]`
    /// InOut, `K` the reduction axis), mirroring `matmul_forward`'s real
    /// `#[tile(...)]` declaration. Built locally rather than imported from
    /// `teeny-kernels` (which depends on this crate, not the reverse).
    fn gemm_shaped_spec() -> KernelTileSpec {
        const A: TensorTileSpec = TensorTileSpec {
            param: "a_ptr",
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
                    block_const: "BLOCK_K",
                    extent_param: "K",
                    window: None,
                },
            ],
            reduction_axis: Some(1),
            untiled_dims: &[],
        };
        const B: TensorTileSpec = TensorTileSpec {
            param: "b_ptr",
            rank: 2,
            axes: &[
                TileAxisBinding {
                    dim: 0,
                    block_const: "BLOCK_K",
                    extent_param: "K",
                    window: None,
                },
                TileAxisBinding {
                    dim: 1,
                    block_const: "BLOCK_N",
                    extent_param: "N",
                    window: None,
                },
            ],
            reduction_axis: Some(0),
            untiled_dims: &[],
        };
        const C: TensorTileSpec = TensorTileSpec {
            param: "c_ptr",
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
        };
        KernelTileSpec {
            inputs: &[A, B, C], // c_ptr is InOut: present in both lists
            outputs: &[C],
            loop_spec: None,
        }
    }

    #[test]
    fn propagate_within_kernel_flat_elementwise_is_identity() {
        let spec = KernelTileSpec {
            inputs: &[X],
            outputs: &[Y],
            loop_spec: None,
        };
        let resolved = propagate_within_kernel(&spec, "y_ptr", &[1000]);
        assert_eq!(resolved.get("n_elements"), Some(&1000));
    }

    #[test]
    fn propagate_within_kernel_gemm_resolves_m_and_n_not_k() {
        let spec = gemm_shaped_spec();
        // Chosen output tile for c_ptr: M=256, N=512.
        let resolved = propagate_within_kernel(&spec, "c_ptr", &[256, 512]);
        assert_eq!(resolved.get("M"), Some(&256));
        assert_eq!(resolved.get("N"), Some(&512));
        // K is c_ptr's reduction axis on neither of its own axes -- it's
        // absent from c_ptr entirely, so it can't be derived from c_ptr's
        // chosen shape alone. This is the whole point: reduction-axis tile
        // size is chosen independently (teenygrad-3w0.9/.11), not
        // propagated top-down from the output.
        assert_eq!(resolved.get("K"), None);
        assert_eq!(resolved.len(), 2);
    }

    #[test]
    #[should_panic(expected = "not a declared output")]
    fn propagate_within_kernel_panics_on_unknown_output_param() {
        let spec = gemm_shaped_spec();
        propagate_within_kernel(&spec, "nonexistent_ptr", &[1, 2]);
    }

    #[test]
    #[should_panic(expected = "chosen_output has")]
    fn propagate_within_kernel_panics_on_shape_length_mismatch() {
        let spec = gemm_shaped_spec();
        propagate_within_kernel(&spec, "c_ptr", &[256]); // c_ptr has 2 axes
    }
}
