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
//! This implementation is a simple row-per-CTA dot-product kernel. Each CTA
//! computes one row of the output matrix C by loading a row of A and a column
//! of B (both of length K) and reducing.
//!
//! Grid: [M, 1, 1] — one CTA per output row.
//! Block: [BLOCK_K, 1, 1]
//!
//! For production use, replace with a tiled GEMM using `T::make_block_ptr` and
//! `T::dot` for Tensor Core utilisation.

#![allow(non_snake_case)]

use teeny_core::dtype::Float;
use teeny_macros::kernel;
use teeny_triton::triton::{
    types::{AddOffsets, Comparison},
    *,
};

// ── MatMul Forward ────────────────────────────────────────────────────────────
//
// C[M, N] = A[M, K] @ B[K, N]
//
// Kernel grid: [M * N, 1, 1]  — one CTA per output element.
// BLOCK_K is the tile size over the K dimension.
//
// Per CTA:
//   pid = m * N + n  (flat output index)
//   C[m, n] = sum_k A[m, k] * B[k, n]

/// Forward: C = A @ B
#[kernel]
pub fn matmul_forward<T: Triton, D: Float, const BLOCK_K: i32>(
    a_ptr: T::Pointer<D>,
    b_ptr: T::Pointer<D>,
    c_ptr: T::Pointer<D>,
    M: i32,
    N: i32,
    K: i32,
) where
    T::I32Tensor: types::Tensor<i32, 1>,
    T::I32Tensor: Comparison<i32, BoolTensor = T::BoolTensor>,
    T::Pointer<D>: AddOffsets<i32, 1, T::I32Tensor, Output = T::Tensor<T::Pointer<D>>>,
{
    let pid = T::program_id(Axis::X);
    let n = pid % N;
    let m = pid / N;

    if m >= M { return; }

    // Load row m of A: A[m, 0..K]
    let k_offsets = T::arange(0, BLOCK_K);
    let a_offsets = k_offsets + m * K;
    let k_mask    = k_offsets.lt(K);
    let zero_fill = T::zeros::<D>(&[BLOCK_K]);

    let a_row = T::load(
        a_ptr.add_offsets(a_offsets),
        Some(k_mask),
        Some(zero_fill),
        &[], None, None, None, false,
    );

    // Load column n of B: B[0..K, n]  — stored in row-major so B[k, n] = B[k*N + n]
    // We must load with stride N: offsets = k*N + n  for k in 0..K
    let b_col_offsets = k_offsets * N + n;
    let b_col_mask    = k_offsets.lt(K);
    let b_col = T::load(
        b_ptr.add_offsets(b_col_offsets),
        Some(b_col_mask),
        Some(zero_fill),
        &[], None, None, None, false,
    );

    // dot product: sum(a_row * b_col)
    let dot = T::sum(a_row * b_col, Some(0), true);

    let c_offset = T::arange(0, 1) + (m * N + n);
    T::store(c_ptr.add_offsets(c_offset), dot, None, &[], None, None);
}

/// Backward: dA = dC @ B^T,  dB = A^T @ dC
///
/// This kernel computes one element of dA per CTA.
/// Grid: [M * K, 1, 1]
/// dA[m, k] = sum_n dC[m, n] * B[k, n]  (B[k, n] = B^T[n, k])
#[kernel]
pub fn matmul_backward_da<T: Triton, D: Float, const BLOCK_N: i32>(
    dc_ptr: T::Pointer<D>,
    b_ptr:  T::Pointer<D>,
    da_ptr: T::Pointer<D>,
    M: i32,
    N: i32,
    K: i32,
) where
    T::I32Tensor: types::Tensor<i32, 1>,
    T::I32Tensor: Comparison<i32, BoolTensor = T::BoolTensor>,
    T::Pointer<D>: AddOffsets<i32, 1, T::I32Tensor, Output = T::Tensor<T::Pointer<D>>>,
{
    let pid = T::program_id(Axis::X);
    let k = pid % K;
    let m = pid / K;
    if m >= M { return; }

    // Load row m of dC: dC[m, 0..N]
    let n_offsets = T::arange(0, BLOCK_N);
    let dc_offsets = n_offsets + m * N;
    let n_mask     = n_offsets.lt(N);
    let zero_fill  = T::zeros::<D>(&[BLOCK_N]);

    let dc_row = T::load(
        dc_ptr.add_offsets(dc_offsets),
        Some(n_mask),
        Some(zero_fill),
        &[], None, None, None, false,
    );

    // Load row k of B (= column k of B^T): B[k, 0..N]
    let b_row_offsets = n_offsets + k * N;
    let b_row = T::load(
        b_ptr.add_offsets(b_row_offsets),
        Some(n_mask),
        Some(zero_fill),
        &[], None, None, None, false,
    );

    let dot = T::sum(dc_row * b_row, Some(0), true);

    let da_offset = T::arange(0, 1) + (m * K + k);
    T::store(da_ptr.add_offsets(da_offset), dot, None, &[], None, None);
}

/// Backward: dB[k, n] = sum_m A[m, k] * dC[m, n]
/// Grid: [K * N, 1, 1]
#[kernel]
pub fn matmul_backward_db<T: Triton, D: Float, const BLOCK_M: i32>(
    dc_ptr: T::Pointer<D>,
    a_ptr:  T::Pointer<D>,
    db_ptr: T::Pointer<D>,
    M: i32,
    N: i32,
    K: i32,
) where
    T::I32Tensor: types::Tensor<i32, 1>,
    T::I32Tensor: Comparison<i32, BoolTensor = T::BoolTensor>,
    T::Pointer<D>: AddOffsets<i32, 1, T::I32Tensor, Output = T::Tensor<T::Pointer<D>>>,
{
    let pid = T::program_id(Axis::X);
    let n = pid % N;
    let k = pid / N;
    if k >= K { return; }

    // Load column n of dC: dC[0..M, n]
    let m_offsets = T::arange(0, BLOCK_M);
    let dc_col_offsets = m_offsets * N + n;
    let m_mask     = m_offsets.lt(M);
    let zero_fill  = T::zeros::<D>(&[BLOCK_M]);

    let dc_col = T::load(
        dc_ptr.add_offsets(dc_col_offsets),
        Some(m_mask),
        Some(zero_fill),
        &[], None, None, None, false,
    );

    // Load column k of A: A[0..M, k]
    let a_col_offsets = m_offsets * K + k;
    let a_col = T::load(
        a_ptr.add_offsets(a_col_offsets),
        Some(m_mask),
        Some(zero_fill),
        &[], None, None, None, false,
    );

    let dot = T::sum(dc_col * a_col, Some(0), true);

    let db_offset = T::arange(0, 1) + (k * N + n);
    T::store(db_ptr.add_offsets(db_offset), dot, None, &[], None, None);
}

// ── RuntimeOp for MatMul ──────────────────────────────────────────────────────

pub struct MatMulRuntimeOp<D: Float + Send + Sync + 'static> {
    pub fwd_kernel: MatmulForward<D>,
    pub bwd_da_kernel: MatmulBackwardDa<D>,
    pub bwd_db_kernel: MatmulBackwardDb<D>,
}

impl<D: Float + Send + Sync + 'static> MatMulRuntimeOp<D> {
    pub fn new(block_size: i32) -> Self {
        Self {
            fwd_kernel: MatmulForward::<D>::new(block_size),
            bwd_da_kernel: MatmulBackwardDa::<D>::new(block_size),
            bwd_db_kernel: MatmulBackwardDb::<D>::new(block_size),
        }
    }

    pub fn forward_source(&self) -> &str { &self.fwd_kernel.source }
    pub fn kernel_name(&self) -> &str { &self.fwd_kernel.name }
}

impl<D: Float + Send + Sync + 'static> teeny_core::model::RuntimeOp for MatMulRuntimeOp<D> {
    fn n_activation_inputs(&self) -> usize { 2 }

    fn param_shapes(&self, _: &[&[usize]], _: &[usize]) -> Vec<Vec<usize>> { vec![] }

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
        visitor.visit_ptr(output);      // c_ptr
        visitor.visit_i32(m);
        visitor.visit_i32(n);
        visitor.visit_i32(k);
    }

    fn block(&self) -> [u32; 3] { [self.fwd_kernel.block_k as u32, 1, 1] }

    fn grid(&self, output_shape: &[usize]) -> [u32; 3] {
        let m = output_shape.first().copied().unwrap_or(1) as u32;
        let n = output_shape.last().copied().unwrap_or(1) as u32;
        [m * n, 1, 1]
    }

    #[cfg(feature = "training")]
    fn has_backward(&self) -> bool { true }

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
        visitor.visit_ptr(grad_output);    // dc_ptr
        visitor.visit_ptr(inputs[1].0);    // b_ptr
        visitor.visit_ptr(grad_inputs[0]); // da_ptr
        visitor.visit_i32(m);
        visitor.visit_i32(n);
        visitor.visit_i32(k);
    }

    #[cfg(feature = "training")]
    fn backward_block(&self) -> [u32; 3] { [self.bwd_da_kernel.block_n as u32, 1, 1] }

    #[cfg(feature = "training")]
    fn backward_grid(&self, input_shapes: &[&[usize]], _: &[usize]) -> [u32; 3] {
        let m = input_shapes.first().and_then(|s| s.first()).copied().unwrap_or(1) as u32;
        let k = input_shapes.first().and_then(|s| s.last()).copied().unwrap_or(1) as u32;
        [m * k, 1, 1]
    }
}
