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

// ── CosineEmbeddingLoss ───────────────────────────────────────────────────────

/// Cosine embedding loss forward (per-row).
///
/// ```text
/// cos_sim = dot(x1[n], x2[n]) / (||x1[n]|| * ||x2[n]||)
/// out[n]  = 1 - cos_sim              if y[n] ==  1
///         = max(0, cos_sim - margin) if y[n] == -1
/// ```
///
/// Grid: `[n_rows, 1, 1]`.  `BLOCK_SIZE` must equal `next_power_of_two(n_dim)`.
///
/// Norms computed as `exp(0.5 * log(sq))` to avoid device-library calls from
/// `math.rsqrt`/`math.sqrt` on scalar f32 values.
#[kernel]
pub fn cosine_embedding_loss_forward<T: Triton, const BLOCK_SIZE: i32>(
    x1_ptr: T::Pointer<f32>,
    x2_ptr: T::Pointer<f32>,
    y_ptr: T::Pointer<f32>,
    out_ptr: T::Pointer<f32>,
    _n_rows: i32,
    n_dim: i32,
    margin: f32,
) where
    T::I32Tensor: types::Tensor<i32, 1>,
    T::I32Tensor: Comparison<i32, BoolTensor = T::BoolTensor>,
    T::Pointer<f32>: AddOffsets<i32, 1, T::I32Tensor, Output = T::Tensor<T::Pointer<f32>>>,
{
    let pid = T::program_id(Axis::X);
    let row_base = pid * n_dim;
    let col_offs: T::I32Tensor = T::arange(0, BLOCK_SIZE);
    let row_offs: T::I32Tensor = col_offs + row_base;
    let in_row = col_offs.lt(n_dim);
    let zeros = T::zeros::<f32>(&[BLOCK_SIZE]);

    let x1 = T::load(x1_ptr.add_offsets(row_offs), Some(in_row), Some(zeros), &[], None, None, None, false);
    let x2 = T::load(x2_ptr.add_offsets(row_offs), Some(in_row), Some(zeros), &[], None, None, None, false);

    // Reduction scalars (tt.reduce returns f32 scalar in MLIR)
    let dot_raw = T::sum(x1 * x2, Some(0), true);
    let sq1_raw = T::sum(x1 * x1, Some(0), true);
    let sq2_raw = T::sum(x2 * x2, Some(0), true);

    // Force scalars to tensor<1xf32> so math ops lower to vectorized PTX instructions
    // rather than calling device library functions (__nv_rsqrtf etc.)
    let dot_t = T::zeros::<f32>(&[1]) + dot_raw;
    let sq1_t = T::zeros::<f32>(&[1]) + sq1_raw;
    let sq2_t = T::zeros::<f32>(&[1]) + sq2_raw;

    // sqrt(sq) = exp(0.5 * log(sq)) — uses only native log/exp PTX approximations
    let half = T::full::<f32>(&[1], 0.5_f32);
    let norm1 = T::exp(half * T::log(sq1_t));
    let norm2 = T::exp(half * T::log(sq2_t));
    let cos_sim = dot_t / (norm1 * norm2);

    // Load y[pid] → tensor<1xf32>
    let y_off: T::I32Tensor = T::arange(0, 1) + pid;
    let y: T::Tensor<f32> = T::load(
        y_ptr.add_offsets(y_off),
        None, None, &[], None, None, None, false,
    );

    let zeros1 = T::zeros::<f32>(&[1]);
    let margin_t = T::full::<f32>(&[1], margin);

    let y_is_pos = T::gt(y, zeros1);
    let hinge = T::maximum(cos_sim - margin_t, zeros1);
    let loss = T::where_(y_is_pos, T::full::<f32>(&[1], 1.0_f32) - cos_sim, hinge);

    let out_off: T::I32Tensor = T::arange(0, 1) + pid;
    T::store(out_ptr.add_offsets(out_off), loss, None, &[], None, None);
}

/// Cosine embedding loss backward (per-row).
///
/// Let `c = cos_sim`, `n1 = ||x1||`, `n2 = ||x2||`, `r1 = 1/n1`, `r2 = 1/n2`.
/// ```text
/// dc/dx1[k] = (x2[k]*r2 - c*x1[k]*r1) * r1
/// dc/dx2[k] = (x1[k]*r1 - c*x2[k]*r2) * r2
///
/// coeff = -dy  if y ==  1
///       =  dy  if y == -1 and cos_sim > margin
///       =   0  otherwise
///
/// dx1 = coeff * dc/dx1,   dx2 = coeff * dc/dx2
/// ```
///
/// Grid: `[n_rows, 1, 1]`.  `BLOCK_SIZE` must equal `next_power_of_two(n_dim)`.
#[kernel]
pub fn cosine_embedding_loss_backward<T: Triton, const BLOCK_SIZE: i32>(
    dy_ptr: T::Pointer<f32>,
    x1_ptr: T::Pointer<f32>,
    x2_ptr: T::Pointer<f32>,
    y_ptr: T::Pointer<f32>,
    dx1_ptr: T::Pointer<f32>,
    dx2_ptr: T::Pointer<f32>,
    _n_rows: i32,
    n_dim: i32,
    margin: f32,
) where
    T::I32Tensor: types::Tensor<i32, 1>,
    T::I32Tensor: Comparison<i32, BoolTensor = T::BoolTensor>,
    T::Pointer<f32>: AddOffsets<i32, 1, T::I32Tensor, Output = T::Tensor<T::Pointer<f32>>>,
{
    let pid = T::program_id(Axis::X);
    let row_base = pid * n_dim;
    let col_offs: T::I32Tensor = T::arange(0, BLOCK_SIZE);
    let row_offs: T::I32Tensor = col_offs + row_base;
    let in_row = col_offs.lt(n_dim);
    let zeros = T::zeros::<f32>(&[BLOCK_SIZE]);

    let x1 = T::load(x1_ptr.add_offsets(row_offs), Some(in_row), Some(zeros), &[], None, None, None, false);
    let x2 = T::load(x2_ptr.add_offsets(row_offs), Some(in_row), Some(zeros), &[], None, None, None, false);

    // Reductions → scalar f32 in MLIR; force to tensor<1xf32>
    let dot_t  = T::zeros::<f32>(&[1]) + T::sum(x1 * x2, Some(0), true);
    let sq1_t  = T::zeros::<f32>(&[1]) + T::sum(x1 * x1, Some(0), true);
    let sq2_t  = T::zeros::<f32>(&[1]) + T::sum(x2 * x2, Some(0), true);

    let neg_half = T::full::<f32>(&[1], -0.5_f32);

    // inv_normN = 1/||xN|| = exp(-0.5 * log(sq)) — avoids rsqrt device function
    let inv_norm1 = T::exp(neg_half * T::log(sq1_t));
    let inv_norm2 = T::exp(neg_half * T::log(sq2_t));
    let cos_sim = dot_t * inv_norm1 * inv_norm2;

    // Load dy, y → tensor<1xf32>
    let scalar_off: T::I32Tensor = T::arange(0, 1) + pid;
    let dy: T::Tensor<f32> = T::load(
        dy_ptr.add_offsets(scalar_off),
        None, None, &[], None, None, None, false,
    );
    let y: T::Tensor<f32> = T::load(
        y_ptr.add_offsets(scalar_off),
        None, None, &[], None, None, None, false,
    );

    let zeros1 = T::zeros::<f32>(&[1]);
    let margin_t = T::full::<f32>(&[1], margin);

    let y_is_pos    = T::gt(y, zeros1);
    let cos_gt_margin = T::gt(cos_sim, margin_t);
    let neg_dy = T::full::<f32>(&[1], -1.0_f32) * dy;

    // coeff: tensor<1xf32>
    let coeff = T::where_(y_is_pos, neg_dy, T::where_(cos_gt_margin, dy, zeros1));

    // Broadcast tensor<1xf32> values to tensor<BLOCK_SIZExf32> before mixing with x1/x2
    let inv_norm1_b = T::broadcast_to(inv_norm1, &[BLOCK_SIZE]);
    let inv_norm2_b = T::broadcast_to(inv_norm2, &[BLOCK_SIZE]);
    let cos_sim_b   = T::broadcast_to(cos_sim,   &[BLOCK_SIZE]);
    let coeff_b     = T::broadcast_to(coeff,     &[BLOCK_SIZE]);

    // Gradient of cos wrt x1 and x2
    let d_cos_dx1 = (x2 * inv_norm2_b - cos_sim_b * x1 * inv_norm1_b) * inv_norm1_b;
    let d_cos_dx2 = (x1 * inv_norm1_b - cos_sim_b * x2 * inv_norm2_b) * inv_norm2_b;

    let dx1 = coeff_b * d_cos_dx1;
    let dx2 = coeff_b * d_cos_dx2;

    T::store(dx1_ptr.add_offsets(row_offs), dx1, Some(in_row), &[], None, None);
    T::store(dx2_ptr.add_offsets(row_offs), dx2, Some(in_row), &[], None, None);
}

// ── TripletMarginLoss ─────────────────────────────────────────────────────────

/// Triplet margin loss forward (per-row).
///
/// `d(a,p) = sqrt(||a-p||^2 + eps)`,  `d(a,n) = sqrt(||a-n||^2 + eps)`
/// `out[i] = max(0, d(a,p) - d(a,n) + margin)`
///
/// Grid: `[n_rows, 1, 1]`.  `BLOCK_SIZE` must equal `next_power_of_two(n_dim)`.
#[kernel]
pub fn triplet_margin_loss_forward<T: Triton, const BLOCK_SIZE: i32>(
    anchor_ptr: T::Pointer<f32>,
    positive_ptr: T::Pointer<f32>,
    negative_ptr: T::Pointer<f32>,
    out_ptr: T::Pointer<f32>,
    _n_rows: i32,
    n_dim: i32,
    margin: f32,
    eps: f32,
) where
    T::I32Tensor: types::Tensor<i32, 1>,
    T::I32Tensor: Comparison<i32, BoolTensor = T::BoolTensor>,
    T::Pointer<f32>: AddOffsets<i32, 1, T::I32Tensor, Output = T::Tensor<T::Pointer<f32>>>,
{
    let pid = T::program_id(Axis::X);
    let row_base = pid * n_dim;
    let col_offs: T::I32Tensor = T::arange(0, BLOCK_SIZE);
    let row_offs: T::I32Tensor = col_offs + row_base;
    let in_row = col_offs.lt(n_dim);
    let zeros = T::zeros::<f32>(&[BLOCK_SIZE]);

    let a = T::load(anchor_ptr.add_offsets(row_offs),   Some(in_row), Some(zeros), &[], None, None, None, false);
    let p = T::load(positive_ptr.add_offsets(row_offs), Some(in_row), Some(zeros), &[], None, None, None, false);
    let n = T::load(negative_ptr.add_offsets(row_offs), Some(in_row), Some(zeros), &[], None, None, None, false);

    let diff_ap = a - p;
    let diff_an = a - n;

    // sq + eps: scalar + tensor<1xf32> → tensor<1xf32>
    let eps_t = T::full::<f32>(&[1], eps);
    let sq_ap = T::sum(diff_ap * diff_ap, Some(0), true) + eps_t;
    let sq_an = T::sum(diff_an * diff_an, Some(0), true) + eps_t;

    // sqrt(sq) = exp(0.5 * log(sq)) — avoids device library rsqrt/sqrt calls
    let half = T::full::<f32>(&[1], 0.5_f32);
    let d_ap = T::exp(half * T::log(sq_ap));
    let d_an = T::exp(half * T::log(sq_an));

    let margin_t = T::full::<f32>(&[1], margin);
    let zeros1 = T::zeros::<f32>(&[1]);
    let loss = T::maximum(d_ap - d_an + margin_t, zeros1);

    let out_off: T::I32Tensor = T::arange(0, 1) + pid;
    T::store(out_ptr.add_offsets(out_off), loss, None, &[], None, None);
}

/// Triplet margin loss backward (per-row).
///
/// When active (d(a,p) - d(a,n) + margin > 0):
/// ```text
/// da[k] = dy * ((a[k]-p[k])/d(a,p) - (a[k]-n[k])/d(a,n))
/// dp[k] = dy * (p[k]-a[k])/d(a,p)
/// dn[k] = dy * (a[k]-n[k])/d(a,n)
/// ```
///
/// Grid: `[n_rows, 1, 1]`.  `BLOCK_SIZE` must equal `next_power_of_two(n_dim)`.
#[kernel]
pub fn triplet_margin_loss_backward<T: Triton, const BLOCK_SIZE: i32>(
    dy_ptr: T::Pointer<f32>,
    anchor_ptr: T::Pointer<f32>,
    positive_ptr: T::Pointer<f32>,
    negative_ptr: T::Pointer<f32>,
    da_ptr: T::Pointer<f32>,
    dp_ptr: T::Pointer<f32>,
    dn_ptr: T::Pointer<f32>,
    _n_rows: i32,
    n_dim: i32,
    margin: f32,
    eps: f32,
) where
    T::I32Tensor: types::Tensor<i32, 1>,
    T::I32Tensor: Comparison<i32, BoolTensor = T::BoolTensor>,
    T::Pointer<f32>: AddOffsets<i32, 1, T::I32Tensor, Output = T::Tensor<T::Pointer<f32>>>,
{
    let pid = T::program_id(Axis::X);
    let row_base = pid * n_dim;
    let col_offs: T::I32Tensor = T::arange(0, BLOCK_SIZE);
    let row_offs: T::I32Tensor = col_offs + row_base;
    let in_row = col_offs.lt(n_dim);
    let zeros = T::zeros::<f32>(&[BLOCK_SIZE]);

    let a = T::load(anchor_ptr.add_offsets(row_offs),   Some(in_row), Some(zeros), &[], None, None, None, false);
    let p = T::load(positive_ptr.add_offsets(row_offs), Some(in_row), Some(zeros), &[], None, None, None, false);
    let n = T::load(negative_ptr.add_offsets(row_offs), Some(in_row), Some(zeros), &[], None, None, None, false);

    let diff_ap = a - p;
    let diff_an = a - n;

    // sq + eps → tensor<1xf32>; sqrt = exp(0.5*log), inv_sqrt = exp(-0.5*log)
    let eps_t    = T::full::<f32>(&[1], eps);
    let sq_ap    = T::sum(diff_ap * diff_ap, Some(0), true) + eps_t;
    let sq_an    = T::sum(diff_an * diff_an, Some(0), true) + eps_t;

    let half     = T::full::<f32>(&[1],  0.5_f32);
    let neg_half = T::full::<f32>(&[1], -0.5_f32);

    let d_ap     = T::exp(half     * T::log(sq_ap));  // tensor<1xf32>
    let d_an     = T::exp(half     * T::log(sq_an));  // tensor<1xf32>
    let inv_d_ap = T::exp(neg_half * T::log(sq_ap));  // tensor<1xf32>
    let inv_d_an = T::exp(neg_half * T::log(sq_an));  // tensor<1xf32>

    let margin_t = T::full::<f32>(&[1], margin);
    let zeros1   = T::zeros::<f32>(&[1]);
    // active: gradient flows only when triplet margin is positive
    let active = T::gt(d_ap - d_an + margin_t, zeros1);

    // Load dy → tensor<1xf32>
    let scalar_off: T::I32Tensor = T::arange(0, 1) + pid;
    let dy: T::Tensor<f32> = T::load(
        dy_ptr.add_offsets(scalar_off),
        None, None, &[], None, None, None, false,
    );

    // Effective dy: zero out gradient for inactive triplets
    let eff_dy     = T::where_(active, dy, zeros1);
    let neg_eff_dy = T::full::<f32>(&[1], -1.0_f32) * eff_dy;

    // Broadcast tensor<1xf32> to tensor<BLOCK_SIZExf32> for element-wise ops
    let inv_d_ap_b  = T::broadcast_to(inv_d_ap,  &[BLOCK_SIZE]);
    let inv_d_an_b  = T::broadcast_to(inv_d_an,  &[BLOCK_SIZE]);
    let eff_dy_b    = T::broadcast_to(eff_dy,    &[BLOCK_SIZE]);
    let neg_eff_b   = T::broadcast_to(neg_eff_dy, &[BLOCK_SIZE]);

    // Unit direction vectors (diff / distance)
    let unit_ap = diff_ap * inv_d_ap_b;
    let unit_an = diff_an * inv_d_an_b;

    let da = eff_dy_b  * (unit_ap - unit_an);
    let dp = neg_eff_b * unit_ap;
    let dn = eff_dy_b  * unit_an;

    T::store(da_ptr.add_offsets(row_offs), da, Some(in_row), &[], None, None);
    T::store(dp_ptr.add_offsets(row_offs), dp, Some(in_row), &[], None, None);
    T::store(dn_ptr.add_offsets(row_offs), dn, Some(in_row), &[], None, None);
}
