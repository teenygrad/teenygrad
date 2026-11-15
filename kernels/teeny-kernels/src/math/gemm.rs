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

//! 2-D matrix multiply (MatMul / Gemm) Triton kernels.
//!
//! Tiled GEMM using `T::make_tensor_descriptor` + `T::dot` for Tensor Core
//! utilisation — one CTA computes one `[BLOCK_M, BLOCK_N]` (or `[BLOCK_M,
//! BLOCK_K]` / `[BLOCK_K, BLOCK_N]` for the backward kernels) output tile,
//! accumulating over `K`/`N`/`M`-tiles with `T::dot` rather than a scalar
//! multiply-and-reduce per element. Same swizzled-pid / tensor-descriptor
//! structure as [`crate::nn::mlp::linear`]'s `linear_forward`/`linear_backward`.
//!
//! Grid: one CTA per output tile. Block: `[128, 1, 1]`.

#![allow(non_snake_case)]

use teeny_core::dtype::{AddOffsets, Comparison, Num, Tensor};
use teeny_core::graph::MatMulPlan;
use teeny_macros::kernel;
use teeny_triton::triton::{PaddingOption, *};

// ── MatMul Forward ──────────────────────────────
//
// C[b, M, N] = A[b, M, K] @ B[b, K, N]  for each b in [0, BATCH)
//
// Kernel grid: one CTA per [BLOCK_M, BLOCK_N] output tile per batch, X pids
// swizzled by GROUP_M for L2 locality (same scheme as linear_forward) and the
// batch on Y.
//
// The batch is addressed by skipping rows, not by moving the base pointer: each
// operand is described as one tall 2-D matrix, A as [BATCH * M, K], B as
// [BATCH * K, N] and C as [BATCH * M, N], which is how a batched MatMul is
// already laid out, so no copy is needed to get there. A_BATCH_ROWS and
// B_BATCH_ROWS are the rows to skip per batch; 0 means the operand is shared
// across the batch and every b reads the same matrix.
//
// BLOCK_K must be at most K: the accumulation loop reads its last K tile back
// from inside the operands so that no tile hangs off the end of the contracted
// axis, which needs one whole tile to fit. pick_gemm_tile_sizes in the graph
// lowering holds to that, and k_tail_arguments works out where that last tile
// starts and which of its lanes are new.
//
// A tile near the end of a batch would otherwise straddle into the next batch's
// first rows, which the dot would then sum into the result. Each descriptor is
// therefore bounded at the current batch's last row rather than at the buffer's,
// so the straddling part reads as the descriptor's zero padding and contributes
// nothing, and the store masks it off instead of writing it over the next
// batch's output.

/// Forward: C = A @ B, batched
// ANCHOR: matmul_forward
#[kernel]
pub fn matmul_forward<
    T: Triton<I32Tensor = <T as Triton>::Tensor<i32>>,
    D: Num,
    const BLOCK_M: i32,
    const BLOCK_N: i32,
    const BLOCK_K: i32,
    const GROUP_M: i32,
>(
    a_ptr: InPtr<T::Pointer<D>>,
    b_ptr: InPtr<T::Pointer<D>>,
    c_ptr: InOutPtr<T::Pointer<D>>,
    M: i32,
    N: i32,
    K: i32,
    A_BATCH_ROWS: i32,
    B_BATCH_ROWS: i32,
    FULL_TILES: i32,
    TAIL_OFF: i32,
    TAIL_KEEP_FROM: i32,
) where
    T::I32Tensor: core::ops::Add<i32, Output = T::I32Tensor>,
    T::I32Tensor: core::ops::Sub<i32, Output = T::I32Tensor>,
{
    let pid = T::program_id(Axis::X);
    let pid_b = T::program_id(Axis::Y);
    let a_row0 = pid_b * A_BATCH_ROWS;
    let b_row0 = pid_b * B_BATCH_ROWS;
    let c_row0 = pid_b * M;
    let num_pid_m = T::cdiv(M, BLOCK_M);
    let num_pid_n = T::cdiv(N, BLOCK_N);
    let num_pid_in_group = GROUP_M * num_pid_n;
    let group_id = pid / num_pid_in_group;
    let first_pid_m = group_id * GROUP_M;
    let remaining_m = num_pid_m - first_pid_m;
    let group_size_m = if remaining_m < GROUP_M {
        remaining_m
    } else {
        GROUP_M
    };
    let pid_in_group = pid % num_pid_in_group;
    let pid_m = first_pid_m + (pid_in_group % group_size_m);
    let pid_n = pid_in_group / group_size_m;

    let a_desc = T::make_tensor_descriptor(
        a_ptr,
        &[a_row0 + M, K],
        &[K, 1],
        &[BLOCK_M, BLOCK_K],
        Some(PaddingOption::Zero),
    );
    let b_desc = T::make_tensor_descriptor(
        b_ptr,
        &[b_row0 + K, N],
        &[N, 1],
        &[BLOCK_K, BLOCK_N],
        Some(PaddingOption::Zero),
    );

    let mut acc = T::zeros::<D>(&[BLOCK_M, BLOCK_N]);

    // Whole K tiles only. A tile that hangs off the end of the contracted axis
    // cannot be relied on: PaddingOption::Zero fills a partial tile along M or N,
    // but along K the RISC-V backend reads the whole tile as zero, dropping the
    // terms that are in bounds, so the loop never asks for one. FULL_TILES is
    // K / BLOCK_K, computed on the host along with the tail's two arguments.
    for k in 0..FULL_TILES {
        let a = T::load_tensor_descriptor(a_desc, &[a_row0 + pid_m * BLOCK_M, k * BLOCK_K]);
        let b = T::load_tensor_descriptor(b_desc, &[b_row0 + k * BLOCK_K, pid_n * BLOCK_N]);
        // IEEE, not TF32: an imported model's MatMul/Gemm must reproduce its FP32 maths.
        // TF32's 10-bit mantissa put ResNet-18's and MobileNetV3's logits ~1e-3 off
        // onnxruntime; IEEE brings them to ~1e-6.
        acc = T::dot::<D, D>(a, b, Some(acc), InputPrecision::IEEE, None);
    }

    // The K tail: one more whole tile, read back from TAIL_OFF, which the host
    // picks so that the tile lies inside the operands and starts on a 16-byte
    // boundary. It overlaps the lanes the loop has already summed, and those are
    // zeroed in A so the dot does not count them twice: a zero there makes the
    // product zero whatever B holds, so B needs no mask of its own.
    //
    // There is no `if` around this. When K divides BLOCK_K exactly the tail is
    // nothing but overlap, and TAIL_KEEP_FROM is then BLOCK_K, so the mask is zero
    // everywhere and the dot adds nothing. One wasted tile beats a branch: a
    // scalar `if` inside this loop is mis-lowered, which cost a first attempt at
    // this fix its clamp, and a partial tile is what the backend drops.
    let lane = T::arange(0, BLOCK_K);
    let keep = T::maximum(
        T::minimum(lane - TAIL_KEEP_FROM + 1, T::full::<i32>(&[BLOCK_K], 1)),
        T::full::<i32>(&[BLOCK_K], 0),
    );
    let keep_d = T::broadcast_to(
        T::expand_dims(T::cast::<i32, D>(keep, None, false), 0),
        &[BLOCK_M, BLOCK_K],
    );
    let a_tail = T::load_tensor_descriptor(a_desc, &[a_row0 + pid_m * BLOCK_M, TAIL_OFF]);
    let b_tail = T::load_tensor_descriptor(b_desc, &[b_row0 + TAIL_OFF, pid_n * BLOCK_N]);
    acc = T::dot::<D, D>(
        a_tail * keep_d,
        b_tail,
        Some(acc),
        InputPrecision::IEEE,
        None,
    );

    let c_desc = T::make_tensor_descriptor(
        c_ptr,
        &[c_row0 + M, N],
        &[N, 1],
        &[BLOCK_M, BLOCK_N],
        Some(PaddingOption::Zero),
    );
    T::store_tensor_descriptor(c_desc, &[c_row0 + pid_m * BLOCK_M, pid_n * BLOCK_N], acc);
}
// ANCHOR_END: matmul_forward

/// Backward: dA = dC @ B^T
///
/// Grid: one CTA per `[BLOCK_M, BLOCK_K]` tile of dA.
/// dA\[m, k\] = sum_n dC\[m, n\] * B\[k, n\]  (B\[k, n\] = B^T\[n, k\])
#[kernel]
pub fn matmul_backward_da<
    T: Triton,
    D: Num,
    const BLOCK_M: i32,
    const BLOCK_N: i32,
    const BLOCK_K: i32,
    const GROUP_M: i32,
>(
    dc_ptr: InPtr<T::Pointer<D>>,
    b_ptr: InPtr<T::Pointer<D>>,
    da_ptr: InPtr<T::Pointer<D>>,
    M: i32,
    N: i32,
    K: i32,
) {
    let pid = T::program_id(Axis::X);
    let num_pid_k = T::cdiv(K, BLOCK_K);
    let pid_k = pid % num_pid_k;
    let pid_m = pid / num_pid_k;

    let dc_desc = T::make_tensor_descriptor(
        dc_ptr,
        &[M, N],
        &[N, 1],
        &[BLOCK_M, BLOCK_N],
        Some(PaddingOption::Zero),
    );
    let b_desc = T::make_tensor_descriptor(
        b_ptr,
        &[K, N],
        &[N, 1],
        &[BLOCK_K, BLOCK_N],
        Some(PaddingOption::Zero),
    );

    let mut acc = T::zeros::<D>(&[BLOCK_M, BLOCK_K]);
    let n_tiles = T::cdiv(N, BLOCK_N);
    for n in 0..n_tiles {
        let dc = T::load_tensor_descriptor(dc_desc, &[pid_m * BLOCK_M, n * BLOCK_N]);
        let b = T::load_tensor_descriptor(b_desc, &[pid_k * BLOCK_K, n * BLOCK_N]);
        let b_t = T::trans(b, &[1, 0]);
        acc = T::dot::<D, D>(dc, b_t, Some(acc), InputPrecision::TF32, None);
    }

    let da_desc = T::make_tensor_descriptor(
        da_ptr,
        &[M, K],
        &[K, 1],
        &[BLOCK_M, BLOCK_K],
        Some(PaddingOption::Zero),
    );
    T::store_tensor_descriptor(da_desc, &[pid_m * BLOCK_M, pid_k * BLOCK_K], acc);
}

/// Backward: dB = A^T @ dC
///
/// Grid: one CTA per `[BLOCK_K, BLOCK_N]` tile of dB.
/// dB\[k, n\] = sum_m A\[m, k\] * dC\[m, n\]
#[kernel]
pub fn matmul_backward_db<
    T: Triton,
    D: Num,
    const BLOCK_M: i32,
    const BLOCK_N: i32,
    const BLOCK_K: i32,
    const GROUP_M: i32,
>(
    dc_ptr: InPtr<T::Pointer<D>>,
    a_ptr: InPtr<T::Pointer<D>>,
    db_ptr: OutPtr<T::Pointer<D>>,
    M: i32,
    N: i32,
    K: i32,
) {
    let pid = T::program_id(Axis::X);
    let num_pid_n = T::cdiv(N, BLOCK_N);
    let pid_n = pid % num_pid_n;
    let pid_k = pid / num_pid_n;

    let dc_desc = T::make_tensor_descriptor(
        dc_ptr,
        &[M, N],
        &[N, 1],
        &[BLOCK_M, BLOCK_N],
        Some(PaddingOption::Zero),
    );
    let a_desc = T::make_tensor_descriptor(
        a_ptr,
        &[M, K],
        &[K, 1],
        &[BLOCK_M, BLOCK_K],
        Some(PaddingOption::Zero),
    );

    let mut acc = T::zeros::<D>(&[BLOCK_K, BLOCK_N]);
    let m_tiles = T::cdiv(M, BLOCK_M);
    for m in 0..m_tiles {
        let dc = T::load_tensor_descriptor(dc_desc, &[m * BLOCK_M, pid_n * BLOCK_N]);
        let a = T::load_tensor_descriptor(a_desc, &[m * BLOCK_M, pid_k * BLOCK_K]);
        let a_t = T::trans(a, &[1, 0]);
        acc = T::dot::<D, D>(a_t, dc, Some(acc), InputPrecision::TF32, None);
    }

    let db_desc = T::make_tensor_descriptor(
        db_ptr,
        &[K, N],
        &[N, 1],
        &[BLOCK_K, BLOCK_N],
        Some(PaddingOption::Zero),
    );
    T::store_tensor_descriptor(db_desc, &[pid_k * BLOCK_K, pid_n * BLOCK_N], acc);
}

// ── RuntimeOp for MatMul ──────────────────────────────────────────────────────

/// Swizzle group size for `matmul_forward`'s pid decomposition — see `linear_forward`.
const GROUP_M: i32 = 8;

/// Whether a tensor-descriptor tile that hangs off the end of the contracted axis
/// loads the part of it that is in bounds.
///
/// CUDA's tensor descriptors fill the rest with zeros, as `PaddingOption::Zero`
/// asks. The RISC-V backend reads the whole tile as zero instead, dropping the
/// terms that were in bounds, so the kernel's last K tile has to be read from
/// somewhere it fits entirely. The two backends also differ in what they allow:
/// CUDA wants a descriptor offset along the innermost axis to be 16-byte aligned
/// and faults with an illegal instruction otherwise, which is why the backend that
/// can use a partial tile is the one that has to.
pub(crate) const PARTIAL_K_TILE_IS_PADDED: bool = cfg!(feature = "cuda");

/// Where `matmul_forward`'s extra K tile starts, and from which of its lanes it
/// contributes.
///
/// The kernel sums `k / block_k` whole tiles and then one more, whose lanes below
/// the second return value are zeroed in A so that nothing is counted twice. There
/// are two ways to place that tile, and [`PARTIAL_K_TILE_IS_PADDED`] decides which:
///
/// - Where the next tile would fall, `(k / block_k) * block_k`, hanging off the end
///   of the axis by `block_k - k % block_k` lanes. Nothing repeats, so no lane is
///   masked, and the lanes past the end read as zero and add nothing. This offset is
///   a multiple of `block_k` and so always aligned.
/// - Read back to `k - block_k`, the last whole tile of the axis, which repeats the
///   lanes from there to `(k / block_k) * block_k` and masks them off. No tile is
///   ever partial, at the cost of an offset that is not generally aligned.
///
/// When `k` divides by `block_k` exactly there is no tail: the read-back form is
/// used, its mask comes out zero everywhere, and the tile adds nothing.
///
/// `None` when `block_k` is wider than `k`, which leaves no whole tile to read back
/// to; `MatMulRuntimeOp::new` passes that on rather than placing a tile that would
/// be wrong on one of the two backends.
fn k_tail_arguments(k: usize, block_k: usize) -> Option<(usize, usize)> {
    if block_k == 0 {
        return None;
    }
    let covered = (k / block_k) * block_k;
    if PARTIAL_K_TILE_IS_PADDED && covered < k {
        // Covers a contraction narrower than the tile too: the loop then runs no
        // whole tiles and this one partial tile is the whole contraction.
        return Some((covered, 0));
    }
    if block_k > k {
        return None;
    }
    let off = k - block_k;
    Some((off, covered - off))
}

fn plan_k_tiles(plan: &MatMulPlan, block_k: i32) -> usize {
    plan.k / (block_k as usize).max(1)
}

pub struct MatMulRuntimeOp<D: Num + Send + Sync + 'static> {
    pub fwd_kernel: MatmulForward<D>,
    pub bwd_da_kernel: MatmulBackwardDa<D>,
    pub bwd_db_kernel: MatmulBackwardDb<D>,
    /// Whole K tiles the accumulation loop runs, `k / block_k`.
    pub full_tiles: i32,
    /// Where the extra K tile starts; see [`k_tail_arguments`].
    pub tail_off: i32,
    /// The first lane of that tile the loop has not already summed.
    pub tail_keep_from: i32,
    /// How the operands' shapes map onto the batched GEMM.
    ///
    /// The extents come from here rather than from the shapes `pack_args` is handed,
    /// because reading `m` and `k` off the ends of the left operand's shape is only
    /// right for a rank-2 operand: a batched product's leading axes are either
    /// further batches or extra rows, and which one decides where every element is.
    pub plan: MatMulPlan,
}

impl<D: Num + Send + Sync + 'static> MatMulRuntimeOp<D> {
    /// `None` if `block_k` is wider than the plan's contracted extent, which leaves
    /// the forward kernel's last K tile nowhere inside the operands to be read from.
    pub fn new(block_m: i32, block_n: i32, block_k: i32, plan: MatMulPlan) -> Option<Self> {
        let (tail_off, tail_keep_from) = k_tail_arguments(plan.k, block_k as usize)?;
        let full_tiles = plan_k_tiles(&plan, block_k) as i32;
        Some(Self {
            fwd_kernel: MatmulForward::<D>::new(block_m, block_n, block_k, GROUP_M),
            bwd_da_kernel: MatmulBackwardDa::<D>::new(block_m, block_n, block_k, GROUP_M),
            bwd_db_kernel: MatmulBackwardDb::<D>::new(block_m, block_n, block_k, GROUP_M),
            full_tiles,
            plan,
            tail_off: tail_off as i32,
            tail_keep_from: tail_keep_from as i32,
        })
    }

    pub fn forward_source(&self) -> &str {
        &self.fwd_kernel.source
    }
    pub fn kernel_name(&self) -> &str {
        self.fwd_kernel.name
    }
}

impl<D: Num + Send + Sync + 'static> teeny_core::model::RuntimeOp for MatMulRuntimeOp<D> {
    fn n_activation_inputs(&self) -> usize {
        2
    }

    fn param_shapes(&self, _: &[&[usize]], _: &[usize]) -> Vec<Vec<usize>> {
        vec![]
    }

    fn pack_args(
        &self,
        inputs: &[(teeny_core::model::RawPtr, &[usize])],
        _: &[teeny_core::model::RawPtr],
        output: teeny_core::model::RawPtr,
        output_shape: &[usize],
        _: i32,
        visitor: &mut dyn teeny_core::device::program::ArgVisitor,
    ) {
        let _ = output_shape;
        visitor.visit_ptr(inputs[0].0); // a_ptr
        visitor.visit_ptr(inputs[1].0); // b_ptr
        visitor.visit_ptr(output); // c_ptr
        visitor.visit_i32(self.plan.m as i32);
        visitor.visit_i32(self.plan.n as i32);
        visitor.visit_i32(self.plan.k as i32);
        visitor.visit_i32(self.plan.a_batch_rows as i32);
        visitor.visit_i32(self.plan.b_batch_rows as i32);
        visitor.visit_i32(self.full_tiles);
        visitor.visit_i32(self.tail_off);
        visitor.visit_i32(self.tail_keep_from);
    }

    fn grid(&self, _output_shape: &[usize]) -> [u32; 3] {
        let pm = (self.plan.m as u32).div_ceil(self.fwd_kernel.block_m as u32);
        let pn = (self.plan.n as u32).div_ceil(self.fwd_kernel.block_n as u32);
        [pm * pn, self.plan.batch as u32, 1]
    }

    /// Only for an unbatched product.
    ///
    /// The two backward kernels are 2-D: they take no batch and no row stride, and a
    /// batched product's gradient would need both. Reporting no backward is what keeps
    /// a batched forward from being paired with a gradient computed over the first
    /// matrix alone. Graph lowering builds no backward for `MatMul` either way.
    #[cfg(feature = "training")]
    fn has_backward(&self) -> bool {
        self.plan.batch == 1
    }

    // For backward we pack args for dA kernel. The lowering handles dB separately.
    // This is a simplified backward that only handles dA = dC @ B^T.
    #[cfg(feature = "training")]
    fn pack_backward_args(
        &self,
        inputs: &[(teeny_core::model::RawPtr, &[usize])],
        _: &[teeny_core::model::RawPtr],
        _: teeny_core::model::RawPtr,
        output_shape: &[usize],
        grad_output: teeny_core::model::RawPtr,
        _: i32,
        grad_inputs: &[teeny_core::model::RawPtr],
        _: &[teeny_core::model::RawPtr],
        visitor: &mut dyn teeny_core::device::program::ArgVisitor,
    ) {
        let m = inputs[0].1.first().copied().unwrap_or(1) as i32;
        let k = inputs[0].1.last().copied().unwrap_or(1) as i32;
        let n = output_shape.last().copied().unwrap_or(1) as i32;
        visitor.visit_ptr(grad_output); // dc_ptr
        visitor.visit_ptr(inputs[1].0); // b_ptr
        visitor.visit_ptr(grad_inputs[0]); // da_ptr
        visitor.visit_i32(m);
        visitor.visit_i32(n);
        visitor.visit_i32(k);
    }

    #[cfg(feature = "training")]
    fn backward_grid(&self, input_shapes: &[&[usize]], _: &[usize]) -> [u32; 3] {
        let m = input_shapes
            .first()
            .and_then(|s| s.first())
            .copied()
            .unwrap_or(1) as u32;
        let k = input_shapes
            .first()
            .and_then(|s| s.last())
            .copied()
            .unwrap_or(1) as u32;
        let pm = m.div_ceil(self.bwd_da_kernel.block_m as u32);
        let pk = k.div_ceil(self.bwd_da_kernel.block_k as u32);
        [pm * pk, 1, 1]
    }
}

#[cfg(test)]
mod k_tail_tests {
    use super::{PARTIAL_K_TILE_IS_PADDED, k_tail_arguments};

    /// Whichever placement the backend calls for, the tail tile has to reach the
    /// end of the contracted axis and start no later than the lanes the loop
    /// already summed, or terms go missing.
    #[test]
    fn test_the_tail_tile_reaches_the_end_without_leaving_a_gap() {
        for block_k in [1usize, 2, 4, 8, 16, 32] {
            for k in block_k..block_k * 4 + 7 {
                // k >= block_k throughout, so both placements are available.
                let (off, keep_from) = k_tail_arguments(k, block_k).expect("fits");
                let covered = (k / block_k) * block_k;
                assert!(
                    off + block_k >= k,
                    "k={k} block_k={block_k}: tile at {off} stops before the end"
                );
                assert!(
                    off <= covered,
                    "k={k} block_k={block_k}: tile at {off} starts past the {covered} \
                     lanes the loop summed"
                );
                assert_eq!(
                    keep_from,
                    covered.saturating_sub(off),
                    "k={k} block_k={block_k}: the mask and the overlap disagree"
                );
            }
        }
    }

    /// A tile is either whole or aligned, as the backend requires.
    #[test]
    fn test_the_tail_tile_suits_the_backend() {
        for block_k in [4usize, 8, 16, 32] {
            for k in block_k..block_k * 3 + 5 {
                let (off, _) = k_tail_arguments(k, block_k).expect("fits");
                if PARTIAL_K_TILE_IS_PADDED {
                    assert_eq!(
                        off % 4,
                        0,
                        "k={k} block_k={block_k}: offset {off} unaligned"
                    );
                } else {
                    assert!(
                        off + block_k <= k,
                        "k={k} block_k={block_k}: tile at {off} is partial"
                    );
                }
            }
        }
    }

    #[test]
    fn test_an_exact_multiple_has_a_tail_that_adds_nothing() {
        let (off, keep_from) = k_tail_arguments(64, 32).expect("fits");
        assert_eq!((off, keep_from), (32, 32));
    }

    /// A tile wider than the contraction can only be a partial one, so it is the
    /// backends that pad it that take it.
    #[test]
    fn test_a_tile_wider_than_the_contraction_needs_padding() {
        if PARTIAL_K_TILE_IS_PADDED {
            assert_eq!(k_tail_arguments(6, 8), Some((0, 0)));
        } else {
            assert_eq!(k_tail_arguments(6, 8), None);
        }
        assert_eq!(k_tail_arguments(8, 8), Some((0, 8)));
    }
}

// ---------------------------------------------------------------------------
// Row kernel: any K and N
// ---------------------------------------------------------------------------

/// A batched `MatMul` without tensor descriptors: each program computes `BLOCK_N`
/// outputs of one row, lanes along `N`, with a scalar loop over `K` that broadcasts
/// `a[m, k]` against row `k` of `B`.
///
/// [`matmul_forward`]'s descriptors need 16-byte row strides, so on CUDA it is wrong
/// whenever `K` or `N` is not a multiple of 4 floats (teenygrad#143); the lowering uses
/// this kernel for those shapes instead. It reads through plain masked pointers, so every
/// shape is exact on every backend. Batches and shared operands follow [`MatMulPlan`].
///
/// Grid: `(M * cdiv(N, BLOCK_N), batch)`.
#[allow(clippy::too_many_arguments)]
#[kernel]
pub fn matmul_rows_forward<T: Triton, D: Num, const BLOCK_N: i32>(
    a_ptr: InPtr<T::Pointer<D>>,
    b_ptr: InPtr<T::Pointer<D>>,
    c_ptr: OutPtr<T::Pointer<D>>,
    M: i32,
    N: i32,
    K: i32,
    A_BATCH_ROWS: i32,
    B_BATCH_ROWS: i32,
) where
    T: Triton<I32Tensor = <T as Triton>::Tensor<i32>>,
    T::I32Tensor: Tensor<i32, 1>,
    T::I32Tensor: core::ops::Add<i32, Output = T::I32Tensor>,
    T::I32Tensor: core::ops::Mul<i32, Output = T::I32Tensor>,
    T::I32Tensor: Comparison<i32, BoolTensor = T::BoolTensor>,
    T::Pointer<D>: AddOffsets<i32, 1, T::I32Tensor, Output = T::Tensor<T::Pointer<D>>>,
{
    let pid = T::program_id(Axis::X);
    let pid_b = T::program_id(Axis::Y);
    let tiles = (N + BLOCK_N - 1) / BLOCK_N;
    let m = pid / tiles;
    let tile = pid % tiles;
    let n = T::arange(0, BLOCK_N) + tile * BLOCK_N;
    let nm = n.lt(N);
    let izero = T::zeros::<i32>(&[BLOCK_N]);
    let zeros = T::zeros::<D>(&[BLOCK_N]);
    let a_row = (pid_b * A_BATCH_ROWS + m) * K;
    let b_row0 = pid_b * B_BATCH_ROWS;
    let mut acc = T::zeros::<D>(&[BLOCK_N]);
    for k in 0..K {
        let a = T::load(
            a_ptr.add_offsets(izero + (a_row + k)),
            None,
            None,
            &[],
            None,
            None,
            None,
            false,
        );
        let b = T::load(
            b_ptr.add_offsets(n + (b_row0 + k) * N),
            Some(nm),
            Some(zeros),
            &[],
            None,
            None,
            None,
            false,
        );
        acc = acc + a * b;
    }
    T::store(
        c_ptr.add_offsets(n + (pid_b * M + m) * N),
        acc,
        Some(nm),
        &[],
        None,
        None,
    );
}

/// Runtime dispatch for [`matmul_rows_forward`].
pub struct MatMulRowsRuntimeOp {
    plan: MatMulPlan,
    block_n: i32,
}

impl MatMulRowsRuntimeOp {
    /// The op for `plan`.
    pub fn new(plan: MatMulPlan, block_n: i32) -> Self {
        Self { plan, block_n }
    }
}

impl teeny_core::model::RuntimeOp for MatMulRowsRuntimeOp {
    fn n_activation_inputs(&self) -> usize {
        2
    }

    fn param_shapes(&self, _: &[&[usize]], _: &[usize]) -> Vec<Vec<usize>> {
        vec![]
    }

    fn pack_args(
        &self,
        inputs: &[(teeny_core::model::RawPtr, &[usize])],
        _: &[teeny_core::model::RawPtr],
        output: teeny_core::model::RawPtr,
        _: &[usize],
        _: i32,
        visitor: &mut dyn teeny_core::device::program::ArgVisitor,
    ) {
        visitor.visit_ptr(inputs[0].0);
        visitor.visit_ptr(inputs[1].0);
        visitor.visit_ptr(output);
        for v in [
            self.plan.m,
            self.plan.n,
            self.plan.k,
            self.plan.a_batch_rows,
            self.plan.b_batch_rows,
        ] {
            visitor.visit_i32(v as i32);
        }
    }

    fn grid(&self, _: &[usize]) -> [u32; 3] {
        let tiles = self.plan.n.div_ceil(self.block_n as usize);
        [(self.plan.m * tiles) as u32, self.plan.batch as u32, 1]
    }
}
