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

#![allow(non_snake_case)]

use teeny_macros::kernel;
use teeny_triton::triton::{
    types::{AddOffsets, Comparison},
    *,
};

// ── muon_frob_norm_sq ─────────────────────────────────────────────────────────

/// Parallel Frobenius squared-norm via atomic reduction.
///
/// Each block sums `x[block]²` and atomically adds to `out_ptr[0]`.
/// Caller must **zero** `out_ptr[0]` before launch.
/// After launch: `out_ptr[0] = ||X||_F²`.
///
/// Grid: `[ceil(n_elements / BLOCK_SIZE), 1, 1]`.
#[kernel]
pub fn muon_frob_norm_sq<T: Triton, const BLOCK_SIZE: i32>(
    x_ptr: T::Pointer<f32>,
    out_ptr: T::Pointer<f32>,
    n_elements: i32,
) where
    T::I32Tensor: types::Tensor<i32, 1>,
    T::I32Tensor: Comparison<i32, BoolTensor = T::BoolTensor>,
    T::Pointer<f32>: AddOffsets<i32, 1, T::I32Tensor, Output = T::Tensor<T::Pointer<f32>>>,
{
    let pid = T::program_id(Axis::X);
    let offsets = T::arange(0, BLOCK_SIZE) + pid * BLOCK_SIZE;
    let mask = offsets.lt(n_elements);

    let x = T::load(x_ptr.add_offsets(offsets), Some(mask), Some(T::zeros::<f32>(&[BLOCK_SIZE])), &[], None, None, None, false);
    // Reduce x² within this block → scalar, then expand to tensor<1xf32> for atomic_add
    let partial = T::expand_dims(T::sum(x * x, Some(0), false), 0);

    // Atomic-add the block's partial sum to the single output accumulator
    let out_off: T::I32Tensor = T::arange(0, 1);
    let _ = T::atomic_add(out_ptr.add_offsets(out_off), partial, None, None, None);
}

// ── muon_ns_xtx ───────────────────────────────────────────────────────────────

/// Gram matrix: `T = X·Xᵀ` (!TRANSPOSE) or `T = Xᵀ·X` (TRANSPOSE).
///
/// Both cases are expressed as `A @ Aᵀ` with different views of X:
/// - `!TRANSPOSE` → A = X `[M, N]`,   output T `[M, M]`, contraction K = N.
/// - `TRANSPOSE`  → A = Xᵀ `[N, M]`, output T `[N, N]`, contraction K = M.
///
/// `stride_xm` is the **row stride of X** (typically `= N` for row-major `[M, N]`).
///
/// Grid: `[ceil(R/BLOCK_R)² grouped by GROUP_R, 1, 1]` where R = M or N.
#[kernel]
pub fn muon_ns_xtx<
    T: Triton,
    const TRANSPOSE: bool,
    const BLOCK_R: i32,
    const BLOCK_K: i32,
    const GROUP_R: i32,
>(
    x_ptr: T::Pointer<f32>,
    t_ptr: T::Pointer<f32>,
    M: i32,
    N: i32,
    stride_xm: i32,
) {
    // R: dimension of the square output; K: contraction dim.
    // For !TRANSPOSE: A = X[M, N],   stride(N, 1),   R=M, K=N.
    // For  TRANSPOSE: A = Xᵀ[N, M], stride(1, N),   R=N, K=M.
    let R            = if TRANSPOSE { N }        else { M };
    let K            = if TRANSPOSE { M }        else { N };
    let a_stride_row = if TRANSPOSE { 1 }        else { stride_xm };
    let a_stride_col = if TRANSPOSE { stride_xm } else { 1 };

    let pid = T::program_id(Axis::X);
    let num_pid_r = T::cdiv(R, BLOCK_R);
    let num_pid_in_group = GROUP_R * num_pid_r;
    let group_id = pid / num_pid_in_group;
    let first_pid_r = group_id * GROUP_R;
    let remaining = num_pid_r - first_pid_r;
    let group_size = if remaining < GROUP_R { remaining } else { GROUP_R };
    let pid_in_group = pid % num_pid_in_group;
    let pid_rm = first_pid_r + (pid_in_group % group_size);
    let pid_rn = pid_in_group / group_size;

    // A and B both view the same matrix (symmetric Gram product)
    let a_desc = T::make_tensor_descriptor(x_ptr, &[R, K], &[a_stride_row, a_stride_col], &[BLOCK_R, BLOCK_K], Some(PaddingOption::Zero));
    let b_desc = T::make_tensor_descriptor(x_ptr, &[R, K], &[a_stride_row, a_stride_col], &[BLOCK_R, BLOCK_K], Some(PaddingOption::Zero));

    let mut acc = T::zeros::<f32>(&[BLOCK_R, BLOCK_R]);
    let k_tiles = T::cdiv(K, BLOCK_K);
    for k in 0..k_tiles {
        let a   = T::load_tensor_descriptor(a_desc, &[pid_rm * BLOCK_R, k * BLOCK_K]);
        let b   = T::load_tensor_descriptor(b_desc, &[pid_rn * BLOCK_R, k * BLOCK_K]);
        let b_t = T::trans(b, &[1, 0]);
        acc = T::dot::<f32, f32>(a, b_t, Some(acc), None, None);
    }

    // Output T: [R × R], row-major
    let t_desc = T::make_tensor_descriptor(t_ptr, &[R, R], &[R, 1], &[BLOCK_R, BLOCK_R], Some(PaddingOption::Zero));
    T::store_tensor_descriptor(t_desc, &[pid_rm * BLOCK_R, pid_rn * BLOCK_R], acc);
}

// ── muon_ns_step ──────────────────────────────────────────────────────────────

/// One Newton-Schulz step (in-place): `X ← a·X + b·(T·X)` or `X ← a·X + b·(X·T)`.
///
/// Both cases are expressed as `A @ B` (no explicit transpose on B) with unit
/// inner strides so TMA descriptors remain valid:
///
/// - `!TRANSPOSE` (`T·X`): A = T `[M, K]`, B = X `[K, N]`, K = M.
/// - `TRANSPOSE`  (`X·T`): A = X `[M, K]`, B = T `[K, N]`, K = N.
///
/// The GEMM result is fused with the elementwise update in one store.
///
/// - `stride_tm`: row stride of T (= M for !TRANSPOSE, = N for TRANSPOSE).
/// - `stride_xm = N` (row-major X, always `[M, N]`).
///
/// Grid: `[ceil(M/BLOCK_M) × ceil(N/BLOCK_N) grouped by GROUP_M, 1, 1]`.
#[kernel]
pub fn muon_ns_step<
    T: Triton,
    const TRANSPOSE: bool,
    const BLOCK_M: i32,
    const BLOCK_N: i32,
    const BLOCK_K: i32,
    const GROUP_M: i32,
>(
    t_ptr: T::Pointer<f32>,
    x_ptr: T::Pointer<f32>,
    M: i32,
    N: i32,
    stride_tm: i32,
    stride_xm: i32,
    a: f32,
    b: f32,
) {
    // Inner (contraction) dimension
    let K = if TRANSPOSE { N } else { M };

    let pid = T::program_id(Axis::X);
    let num_pid_m = T::cdiv(M, BLOCK_M);
    let num_pid_n = T::cdiv(N, BLOCK_N);
    let num_pid_in_group = GROUP_M * num_pid_n;
    let group_id = pid / num_pid_in_group;
    let first_pid_m = group_id * GROUP_M;
    let remaining = num_pid_m - first_pid_m;
    let group_size = if remaining < GROUP_M { remaining } else { GROUP_M };
    let pid_in_group = pid % num_pid_in_group;
    let pid_m = first_pid_m + (pid_in_group % group_size);
    let pid_n = pid_in_group / group_size;

    // Descriptors for the GEMM (A @ B, inner dim K, both with unit inner stride):
    //   !TRANSPOSE: A = T[M, K] strides(stride_tm,1),  B = X[K, N] strides(stride_xm,1), K=M
    //    TRANSPOSE: A = X[M, K] strides(stride_xm,1),  B = T[K, N] strides(stride_tm,1), K=N
    let (a_desc, b_desc) = if TRANSPOSE {
        let ad = T::make_tensor_descriptor(x_ptr, &[M, K], &[stride_xm, 1], &[BLOCK_M, BLOCK_K], Some(PaddingOption::Zero));
        let bd = T::make_tensor_descriptor(t_ptr, &[K, N], &[stride_tm, 1], &[BLOCK_K, BLOCK_N], Some(PaddingOption::Zero));
        (ad, bd)
    } else {
        let ad = T::make_tensor_descriptor(t_ptr, &[M, K], &[stride_tm, 1], &[BLOCK_M, BLOCK_K], Some(PaddingOption::Zero));
        let bd = T::make_tensor_descriptor(x_ptr, &[K, N], &[stride_xm, 1], &[BLOCK_K, BLOCK_N], Some(PaddingOption::Zero));
        (ad, bd)
    };

    let mut acc = T::zeros::<f32>(&[BLOCK_M, BLOCK_N]);
    let k_tiles = T::cdiv(K, BLOCK_K);
    for k in 0..k_tiles {
        let av = T::load_tensor_descriptor(a_desc, &[pid_m * BLOCK_M, k * BLOCK_K]);
        let bv = T::load_tensor_descriptor(b_desc, &[k * BLOCK_K, pid_n * BLOCK_N]);
        acc = T::dot::<f32, f32>(av, bv, Some(acc), None, None);
    }

    // Fused elementwise: X_new = a * X_tile + b * GEMM_result
    let x_desc = T::make_tensor_descriptor(x_ptr, &[M, N], &[stride_xm, 1], &[BLOCK_M, BLOCK_N], Some(PaddingOption::Zero));
    let x_tile = T::load_tensor_descriptor(x_desc, &[pid_m * BLOCK_M, pid_n * BLOCK_N]);

    let a_t    = T::full::<f32>(&[BLOCK_M, BLOCK_N], a);
    let b_t    = T::full::<f32>(&[BLOCK_M, BLOCK_N], b);
    let result = a_t * x_tile + b_t * acc;
    T::store_tensor_descriptor(x_desc, &[pid_m * BLOCK_M, pid_n * BLOCK_N], result);
}

// ── muon_update ───────────────────────────────────────────────────────────────

/// Parameter update: `W -= lr · G_orth`.
///
/// Grid: `[ceil(n_elements / BLOCK_SIZE), 1, 1]`.
#[kernel]
pub fn muon_update<T: Triton, const BLOCK_SIZE: i32>(
    params_ptr: T::Pointer<f32>,
    grad_ptr: T::Pointer<f32>,
    n_elements: i32,
    lr: f32,
) where
    T::I32Tensor: types::Tensor<i32, 1>,
    T::I32Tensor: Comparison<i32, BoolTensor = T::BoolTensor>,
    T::Pointer<f32>: AddOffsets<i32, 1, T::I32Tensor, Output = T::Tensor<T::Pointer<f32>>>,
{
    let pid = T::program_id(Axis::X);
    let offsets = T::arange(0, BLOCK_SIZE) + pid * BLOCK_SIZE;
    let mask = offsets.lt(n_elements);

    let p    = T::load(params_ptr.add_offsets(offsets), Some(mask), None, &[], None, None, None, false);
    let g    = T::load(grad_ptr.add_offsets(offsets),   Some(mask), None, &[], None, None, None, false);
    let lr_t = T::full::<f32>(&[BLOCK_SIZE], lr);

    T::store(params_ptr.add_offsets(offsets), p - lr_t * g, Some(mask), &[], None, None);
}
