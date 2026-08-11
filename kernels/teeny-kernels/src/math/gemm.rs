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

use teeny_core::dtype::Num;
use teeny_macros::kernel;
use teeny_triton::triton::{PaddingOption, *};

// ── MatMul Forward ────────────────────────────────────────────────────────────
//
// C[M, N] = A[M, K] @ B[K, N]
//
// Kernel grid: one CTA per [BLOCK_M, BLOCK_N] output tile, pids swizzled by
// GROUP_M for L2 locality (same scheme as linear_forward).

/// Forward: C = A @ B
// ANCHOR: matmul_forward
#[kernel]
pub fn matmul_forward<
    T: Triton,
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
) {
    let pid = T::program_id(Axis::X);
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
        &[M, K],
        &[K, 1],
        &[BLOCK_M, BLOCK_K],
        Some(PaddingOption::Zero),
    );
    let b_desc = T::make_tensor_descriptor(
        b_ptr,
        &[K, N],
        &[N, 1],
        &[BLOCK_K, BLOCK_N],
        Some(PaddingOption::Zero),
    );

    let mut acc = T::zeros::<D>(&[BLOCK_M, BLOCK_N]);
    let k_tiles = T::cdiv(K, BLOCK_K);
    for k in 0..k_tiles {
        let a = T::load_tensor_descriptor(a_desc, &[pid_m * BLOCK_M, k * BLOCK_K]);
        let b = T::load_tensor_descriptor(b_desc, &[k * BLOCK_K, pid_n * BLOCK_N]);
        acc = T::dot::<D, D>(a, b, Some(acc), InputPrecision::TF32, None);
    }

    let c_desc = T::make_tensor_descriptor(
        c_ptr,
        &[M, N],
        &[N, 1],
        &[BLOCK_M, BLOCK_N],
        Some(PaddingOption::Zero),
    );
    T::store_tensor_descriptor(c_desc, &[pid_m * BLOCK_M, pid_n * BLOCK_N], acc);
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

pub struct MatMulRuntimeOp<D: Num + Send + Sync + 'static> {
    pub fwd_kernel: MatmulForward<D>,
    pub bwd_da_kernel: MatmulBackwardDa<D>,
    pub bwd_db_kernel: MatmulBackwardDb<D>,
}

impl<D: Num + Send + Sync + 'static> MatMulRuntimeOp<D> {
    pub fn new(block_m: i32, block_n: i32, block_k: i32) -> Self {
        Self {
            fwd_kernel: MatmulForward::<D>::new(block_m, block_n, block_k, GROUP_M),
            bwd_da_kernel: MatmulBackwardDa::<D>::new(block_m, block_n, block_k, GROUP_M),
            bwd_db_kernel: MatmulBackwardDb::<D>::new(block_m, block_n, block_k, GROUP_M),
        }
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
        // A: [M, K], B: [K, N], C: [M, N]
        let m = inputs[0].1.first().copied().unwrap_or(1) as i32;
        let k = inputs[0].1.last().copied().unwrap_or(1) as i32;
        let n = output_shape.last().copied().unwrap_or(1) as i32;
        visitor.visit_ptr(inputs[0].0); // a_ptr
        visitor.visit_ptr(inputs[1].0); // b_ptr
        visitor.visit_ptr(output); // c_ptr
        visitor.visit_i32(m);
        visitor.visit_i32(n);
        visitor.visit_i32(k);
    }

    fn grid(&self, output_shape: &[usize]) -> [u32; 3] {
        let m = output_shape.first().copied().unwrap_or(1) as u32;
        let n = output_shape.last().copied().unwrap_or(1) as u32;
        let pm = m.div_ceil(self.fwd_kernel.block_m as u32);
        let pn = n.div_ceil(self.fwd_kernel.block_n as u32);
        [pm * pn, 1, 1]
    }

    #[cfg(feature = "training")]
    fn has_backward(&self) -> bool {
        true
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
