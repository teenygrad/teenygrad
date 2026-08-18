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

//! LayerNorm Triton kernels.
//!
//! Layout: input `x` is `[M, N]` row-major where M = product of batch / outer
//! dimensions and N = product of the normalized dimensions.  Each CTA handles
//! one row (one sample), reading all N elements in `BLOCK_N`-wide tiles.
//!
//! Forward: y[m, n] = (x[m, n] − mean_m) / sqrt(var_m + eps) * γ[n] + β[n]
//!
//! Training launches a single forward kernel that also writes out the saved
//! `mean` and `rstd` buffers for the backward pass.

#![allow(non_snake_case)]

use teeny_core::dtype::Float;
use teeny_macros::kernel;
use teeny_triton::triton::{
    types::{AddOffsets, Comparison},
    *,
};

// ─── Inference ───────────────────────────────────────────────────────────────

/// Forward pass using pre-computed running statistics (inference only).
///
/// Grid: `[M]` — one CTA per row.
#[kernel]
pub fn layer_norm_forward_inference<T: Triton, D: Float, const BLOCK_N: i32>(
    x_ptr: InPtr<T::Pointer<D>>,
    y_ptr: OutPtr<T::Pointer<D>>,
    weight_ptr: InPtr<T::Pointer<D>>,
    bias_ptr: InPtr<T::Pointer<D>>,
    _M: i32,
    N: i32,
    eps: f32,
) where
    T::I32Tensor: types::Tensor<i32, 1>,
    T::I32Tensor: Comparison<i32, BoolTensor = T::BoolTensor>,
    T::Pointer<D>: AddOffsets<i32, 1, T::I32Tensor, Output = T::Tensor<T::Pointer<D>>>,
{
    let row = T::program_id(Axis::X);
    let row_start = row * N;

    // ── Pass 1: accumulate mean ───────────────────────────────────────────────
    let zeros = T::zeros::<D>(&[BLOCK_N]);
    let zero_1 = T::zeros::<D>(&[1]);
    let mut sum = zero_1;
    let mut n_start: i32 = 0;
    while n_start < N {
        let col_offs = T::arange(0, BLOCK_N) + n_start;
        let mask = col_offs.lt(N);
        let x_tile = T::load(
            x_ptr.add_offsets(col_offs + row_start),
            Some(mask),
            Some(zeros),
            &[],
            None,
            None,
            None,
            false,
        );
        sum = sum + T::sum(x_tile, None, true);
        n_start += BLOCK_N;
    }
    let n_inv = T::cast::<f32, D>(T::full::<f32>(&[1], 1.0f32 / (N as f32)), None, false);
    let mean_1 = sum * n_inv;
    let mean = T::broadcast_to(mean_1, &[BLOCK_N]);

    // ── Pass 2: accumulate variance ───────────────────────────────────────────
    let mut var_sum = zero_1;
    n_start = 0;
    while n_start < N {
        let col_offs = T::arange(0, BLOCK_N) + n_start;
        let mask = col_offs.lt(N);
        let x_tile = T::load(
            x_ptr.add_offsets(col_offs + row_start),
            Some(mask),
            Some(zeros),
            &[],
            None,
            None,
            None,
            false,
        );
        // Mask the diff so out-of-bounds positions don't contribute mean^2 to variance.
        let diff = T::where_::<D>(mask, x_tile - mean, zeros);
        var_sum = var_sum + T::sum(diff * diff, None, true);
        n_start += BLOCK_N;
    }
    let eps_t = T::cast::<f32, D>(T::full::<f32>(&[1], eps), None, false);
    let rstd = T::broadcast_to(T::rsqrt(var_sum * n_inv + eps_t), &[BLOCK_N]);

    // ── Pass 3: normalise and apply affine transform ──────────────────────────
    n_start = 0;
    while n_start < N {
        let col_offs = T::arange(0, BLOCK_N) + n_start;
        let mask = col_offs.lt(N);
        let x_tile = T::load(
            x_ptr.add_offsets(col_offs + row_start),
            Some(mask),
            Some(zeros),
            &[],
            None,
            None,
            None,
            false,
        );
        let gamma = T::load(
            weight_ptr.add_offsets(col_offs),
            Some(mask),
            Some(zeros),
            &[],
            None,
            None,
            None,
            false,
        );
        let beta = T::load(
            bias_ptr.add_offsets(col_offs),
            Some(mask),
            Some(zeros),
            &[],
            None,
            None,
            None,
            false,
        );
        let y_tile = (x_tile - mean) * rstd * gamma + beta;
        T::store(
            y_ptr.add_offsets(col_offs + row_start),
            y_tile,
            Some(mask),
            &[],
            None,
            None,
        );
        n_start += BLOCK_N;
    }
}

// ─── Training forward ─────────────────────────────────────────────────────────

/// Forward pass that also saves per-row mean and rstd for the backward pass.
///
/// Grid: `[M]` — one CTA per row.
#[cfg(feature = "training")]
#[kernel]
pub fn layer_norm_forward<T: Triton, D: Float, const BLOCK_N: i32>(
    x_ptr: InPtr<T::Pointer<D>>,
    y_ptr: OutPtr<T::Pointer<D>>,
    weight_ptr: InPtr<T::Pointer<D>>,
    bias_ptr: InPtr<T::Pointer<D>>,
    mean_ptr: OutPtr<T::Pointer<D>>,
    rstd_ptr: OutPtr<T::Pointer<D>>,
    _M: i32,
    N: i32,
    eps: f32,
) where
    T::I32Tensor: types::Tensor<i32, 1>,
    T::I32Tensor: Comparison<i32, BoolTensor = T::BoolTensor>,
    T::Pointer<D>: AddOffsets<i32, 1, T::I32Tensor, Output = T::Tensor<T::Pointer<D>>>,
{
    let row = T::program_id(Axis::X);
    let row_start = row * N;
    let row_idx = T::arange(0, 1) + row;

    let zeros = T::zeros::<D>(&[BLOCK_N]);
    let zero_1 = T::zeros::<D>(&[1]);
    let n_inv = T::cast::<f32, D>(T::full::<f32>(&[1], 1.0f32 / (N as f32)), None, false);

    // ── Pass 1: mean ─────────────────────────────────────────────────────────
    let mut sum = zero_1;
    let mut n_start: i32 = 0;
    while n_start < N {
        let col_offs = T::arange(0, BLOCK_N) + n_start;
        let mask = col_offs.lt(N);
        let x_tile = T::load(
            x_ptr.add_offsets(col_offs + row_start),
            Some(mask),
            Some(zeros),
            &[],
            None,
            None,
            None,
            false,
        );
        sum = sum + T::sum(x_tile, None, true);
        n_start += BLOCK_N;
    }
    let mean_1 = sum * n_inv;
    let mean = T::broadcast_to(mean_1, &[BLOCK_N]);

    // ── Pass 2: variance ─────────────────────────────────────────────────────
    let mut var_sum = zero_1;
    n_start = 0;
    while n_start < N {
        let col_offs = T::arange(0, BLOCK_N) + n_start;
        let mask = col_offs.lt(N);
        let x_tile = T::load(
            x_ptr.add_offsets(col_offs + row_start),
            Some(mask),
            Some(zeros),
            &[],
            None,
            None,
            None,
            false,
        );
        // Mask the diff so out-of-bounds positions don't contribute mean^2 to variance.
        let diff = T::where_::<D>(mask, x_tile - mean, zeros);
        var_sum = var_sum + T::sum(diff * diff, None, true);
        n_start += BLOCK_N;
    }
    let eps_t = T::cast::<f32, D>(T::full::<f32>(&[1], eps), None, false);
    let rstd_1 = T::rsqrt(var_sum * n_inv + eps_t);
    let rstd = T::broadcast_to(rstd_1, &[BLOCK_N]);

    T::store(mean_ptr.add_offsets(row_idx), mean_1, None, &[], None, None);
    T::store(rstd_ptr.add_offsets(row_idx), rstd_1, None, &[], None, None);

    // ── Pass 3: normalise ─────────────────────────────────────────────────────
    n_start = 0;
    while n_start < N {
        let col_offs = T::arange(0, BLOCK_N) + n_start;
        let mask = col_offs.lt(N);
        let x_tile = T::load(
            x_ptr.add_offsets(col_offs + row_start),
            Some(mask),
            Some(zeros),
            &[],
            None,
            None,
            None,
            false,
        );
        let gamma = T::load(
            weight_ptr.add_offsets(col_offs),
            Some(mask),
            Some(zeros),
            &[],
            None,
            None,
            None,
            false,
        );
        let beta = T::load(
            bias_ptr.add_offsets(col_offs),
            Some(mask),
            Some(zeros),
            &[],
            None,
            None,
            None,
            false,
        );
        let y_tile = (x_tile - mean) * rstd * gamma + beta;
        T::store(
            y_ptr.add_offsets(col_offs + row_start),
            y_tile,
            Some(mask),
            &[],
            None,
            None,
        );
        n_start += BLOCK_N;
    }
}

// ─── Training backward ───────────────────────────────────────────────────────

/// Backward pass for LayerNorm.
///
/// Given saved `mean` and `rstd` from the forward pass:
/// ```text
/// xhat[m,n]    = (x[m,n] - mean[m]) * rstd[m]
/// dweight[n]   = Σ_m dy[m,n] * xhat[m,n]
/// dbias[n]     = Σ_m dy[m,n]
/// dx[m,n]      = rstd[m] * γ[n] * (dy[m,n]
///                  - (Σ_n dy[m,n]*γ[n]) / N
///                  - xhat[m,n] * (Σ_n dy[m,n]*γ[n]*xhat[m,n]) / N)
/// ```
///
/// Grid: `[M]` — one CTA per row.
#[cfg(feature = "training")]
#[kernel]
pub fn layer_norm_backward<T: Triton, D: Float, const BLOCK_N: i32>(
    dy_ptr: InPtr<T::Pointer<D>>,
    x_ptr: InPtr<T::Pointer<D>>,
    dx_ptr: OutPtr<T::Pointer<D>>,
    weight_ptr: InPtr<T::Pointer<D>>,
    dweight_ptr: InOutPtr<T::Pointer<D>>,
    dbias_ptr: InOutPtr<T::Pointer<D>>,
    mean_ptr: InPtr<T::Pointer<D>>,
    rstd_ptr: InPtr<T::Pointer<D>>,
    _M: i32,
    N: i32,
) where
    T::I32Tensor: types::Tensor<i32, 1>,
    T::I32Tensor: Comparison<i32, BoolTensor = T::BoolTensor>,
    T::Pointer<D>: AddOffsets<i32, 1, T::I32Tensor, Output = T::Tensor<T::Pointer<D>>>,
{
    let row = T::program_id(Axis::X);
    let row_start = row * N;
    let row_idx = T::arange(0, 1) + row;

    let zeros = T::zeros::<D>(&[BLOCK_N]);
    let zero_1 = T::zeros::<D>(&[1]);
    let n_inv = T::cast::<f32, D>(T::full::<f32>(&[1], 1.0f32 / (N as f32)), None, false);

    let rstd_1 = T::load(
        rstd_ptr.add_offsets(row_idx),
        None,
        None,
        &[],
        None,
        None,
        None,
        false,
    );
    let mean_1 = T::load(
        mean_ptr.add_offsets(row_idx),
        None,
        None,
        &[],
        None,
        None,
        None,
        false,
    );
    let rstd = T::broadcast_to(rstd_1, &[BLOCK_N]);
    let mean = T::broadcast_to(mean_1, &[BLOCK_N]);

    // ── Pass 1: accumulate row-level dot products ─────────────────────────────
    let mut sum_dy_gamma = zero_1;
    let mut sum_dy_gamma_xhat = zero_1;
    let mut n_start: i32 = 0;
    while n_start < N {
        let col_offs = T::arange(0, BLOCK_N) + n_start;
        let mask = col_offs.lt(N);
        let x_tile = T::load(
            x_ptr.add_offsets(col_offs + row_start),
            Some(mask),
            Some(zeros),
            &[],
            None,
            None,
            None,
            false,
        );
        let dy_tile = T::load(
            dy_ptr.add_offsets(col_offs + row_start),
            Some(mask),
            Some(zeros),
            &[],
            None,
            None,
            None,
            false,
        );
        let gamma = T::load(
            weight_ptr.add_offsets(col_offs),
            Some(mask),
            Some(zeros),
            &[],
            None,
            None,
            None,
            false,
        );
        let xhat = (x_tile - mean) * rstd;
        sum_dy_gamma = sum_dy_gamma + T::sum(dy_tile * gamma, None, true);
        sum_dy_gamma_xhat = sum_dy_gamma_xhat + T::sum(dy_tile * gamma * xhat, None, true);
        n_start += BLOCK_N;
    }
    let c1 = T::broadcast_to(sum_dy_gamma * n_inv, &[BLOCK_N]);
    let c2 = T::broadcast_to(sum_dy_gamma_xhat * n_inv, &[BLOCK_N]);

    // ── Pass 2: compute dx and accumulate dweight / dbias ────────────────────
    n_start = 0;
    while n_start < N {
        let col_offs = T::arange(0, BLOCK_N) + n_start;
        let mask = col_offs.lt(N);
        let x_tile = T::load(
            x_ptr.add_offsets(col_offs + row_start),
            Some(mask),
            Some(zeros),
            &[],
            None,
            None,
            None,
            false,
        );
        let dy_tile = T::load(
            dy_ptr.add_offsets(col_offs + row_start),
            Some(mask),
            Some(zeros),
            &[],
            None,
            None,
            None,
            false,
        );
        let gamma = T::load(
            weight_ptr.add_offsets(col_offs),
            Some(mask),
            Some(zeros),
            &[],
            None,
            None,
            None,
            false,
        );
        let dw_old = T::load(
            dweight_ptr.add_offsets(col_offs),
            Some(mask),
            Some(zeros),
            &[],
            None,
            None,
            None,
            false,
        );
        let db_old = T::load(
            dbias_ptr.add_offsets(col_offs),
            Some(mask),
            Some(zeros),
            &[],
            None,
            None,
            None,
            false,
        );

        let xhat = (x_tile - mean) * rstd;
        let dx_tile = rstd * gamma * (dy_tile - c1 - xhat * c2);

        T::store(
            dx_ptr.add_offsets(col_offs + row_start),
            dx_tile,
            Some(mask),
            &[],
            None,
            None,
        );
        T::store(
            dweight_ptr.add_offsets(col_offs),
            dw_old + dy_tile * xhat,
            Some(mask),
            &[],
            None,
            None,
        );
        T::store(
            dbias_ptr.add_offsets(col_offs),
            db_old + dy_tile,
            Some(mask),
            &[],
            None,
            None,
        );
        n_start += BLOCK_N;
    }
}

// ─── Inference RuntimeOp ──────────────────────────────────────────────────────

/// RuntimeOp for LayerNorm inference.
///
/// Parameter layout (2 params): `[weight, bias]`, each of shape `[N]` where
/// N is the last (normalized) dimension of the input.
pub struct LayerNormForwardInferenceRuntimeOp<D: Float + Send + Sync + 'static> {
    fwd: LayerNormForwardInference<D>,
    #[allow(dead_code)]
    block_n: i32,
    eps: f32,
}

impl<D: Float + Send + Sync + 'static> LayerNormForwardInferenceRuntimeOp<D> {
    pub fn new(block_n: i32, eps: f32) -> Self {
        Self {
            fwd: LayerNormForwardInference::<D>::new(block_n),
            block_n,
            eps,
        }
    }

    pub fn forward_source(&self) -> &str {
        &self.fwd.source
    }
    pub fn kernel_name(&self) -> &str {
        self.fwd.name
    }
}

impl<D: Float + Send + Sync + 'static> teeny_core::model::RuntimeOp
    for LayerNormForwardInferenceRuntimeOp<D>
{
    fn n_activation_inputs(&self) -> usize {
        1
    }

    fn param_shapes(&self, input_shapes: &[&[usize]], _output_shape: &[usize]) -> Vec<Vec<usize>> {
        // N = last dim of input
        let n = *input_shapes[0].last().unwrap();
        vec![vec![n], vec![n]]
    }

    fn param_names(&self) -> &'static [&'static str] {
        &["weight", "bias"]
    }

    fn pack_args(
        &self,
        inputs: &[(teeny_core::model::RawPtr, &[usize])],
        params: &[teeny_core::model::RawPtr],
        output: teeny_core::model::RawPtr,
        _output_shape: &[usize],
        _output_row_stride: i32,
        visitor: &mut dyn teeny_core::device::program::ArgVisitor,
    ) {
        let shape = inputs[0].1;
        let n = *shape.last().unwrap() as i32;
        let total: usize = shape.iter().product();
        let m = (total as i32) / n;

        visitor.visit_ptr(inputs[0].0); // x
        visitor.visit_ptr(output); // y
        visitor.visit_ptr(params[0]); // weight (gamma)
        visitor.visit_ptr(params[1]); // bias (beta)
        visitor.visit_i32(m); // M
        visitor.visit_i32(n); // N
        visitor.visit_f32(self.eps); // eps
    }

    fn grid(&self, output_shape: &[usize]) -> [u32; 3] {
        // one CTA per row; M = product of all dims except last
        let n = *output_shape.last().unwrap();
        let total: usize = output_shape.iter().product();
        let m = total / n;
        [m as u32, 1, 1]
    }

    fn has_backward(&self) -> bool {
        false
    }
}
