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

//! Flash Attention 2 — forward and backward kernels.
//!
//! **Layout**: all tensors are stored as `[BATCH * N_HEADS, N_CTX, HEAD_DIM]`
//! row-major (contiguous).  The caller is responsible for reshaping
//! `[B, H, N, D]` PyTorch tensors to this flat 3-D layout before calling.
//!
//! **Algorithm**: each CTA processes one `(batch, head, q_row)` triple.
//! The kernel iterates over all `N_CTX_K` key/value rows with the online
//! softmax recurrence (Flash Attention paper, Dao et al. 2022/2023), so
//! the full `N_CTX_Q × N_CTX_K` attention matrix is never materialised.
//! Memory is O(N_CTX × HEAD_DIM) per CTA rather than O(N_CTX²).
//!
//! HEAD_DIM must be a power of two and is a compile-time const so the
//! inner vector loads are always fully unmasked.

use core::marker::PhantomData;
use teeny_macros::kernel;
use teeny_triton::triton::{
    types::{AddOffsets, Comparison, Tensor},
    *,
};

// ── Forward ───────────────────────────────────────────────────────────────────

/// Flash Attention 2 forward pass.
///
/// Inputs  (all `[BH, N_CTX, HEAD_DIM]` row-major, where `BH = BATCH * N_HEADS`):
///   `q_ptr`, `k_ptr`, `v_ptr`
///
/// Outputs:
///   `o_ptr`  — attention output   `[BH, N_CTX_Q, HEAD_DIM]`
///   `l_ptr`  — log-sum-exp        `[BH, N_CTX_Q]`  (saved for backward)
///
/// Grid: `(N_CTX_Q, BH, 1)` — one CTA per `(batch_head, q_row)` pair.
#[kernel]
pub fn flash_attention2_forward<T: Triton, const HEAD_DIM: i32>(
    q_ptr: T::Pointer<f32>,
    k_ptr: T::Pointer<f32>,
    v_ptr: T::Pointer<f32>,
    o_ptr: T::Pointer<f32>,
    l_ptr: T::Pointer<f32>,
    n_ctx_q: i32,
    n_ctx_k: i32,
    softmax_scale: f32, // 1 / sqrt(HEAD_DIM)
    neg_inf: f32,       // f32::NEG_INFINITY — passed explicitly (no_core has no float constants)
) where
    T::I32Tensor: Tensor<i32, 1>,
    T::I32Tensor: Comparison<i32, BoolTensor = T::BoolTensor>,
    T::Pointer<f32>: AddOffsets<i32, 1, T::I32Tensor, Output = T::Tensor<T::Pointer<f32>>>,
{
    let pid_m = T::program_id(Axis::X); // query-row index  [0, n_ctx_q)
    let pid_bh = T::program_id(Axis::Y); // (batch, head)    [0, BH)

    let kv_bh_base = pid_bh * n_ctx_k * HEAD_DIM;
    let q_row_base = pid_bh * n_ctx_q * HEAD_DIM + pid_m * HEAD_DIM;
    let o_row_base = pid_bh * n_ctx_q * HEAD_DIM + pid_m * HEAD_DIM;
    let l_row_base = pid_bh * n_ctx_q + pid_m;

    // HEAD_DIM lane offsets — no masking needed (HEAD_DIM is a power of two).
    let d = T::arange(0, HEAD_DIM);

    // Load Q[pid_bh, pid_m, :]
    let q_vec = T::load(
        q_ptr.add_offsets(d + q_row_base),
        None,
        None,
        &[],
        None,
        None,
        None,
        false,
    );

    // Online-softmax running state — all kept as [HEAD_DIM] tensors so that
    // all scf.for iter-args have the same shape (Triton requires this).
    let mut acc = T::zeros::<f32>(&[HEAD_DIM]);
    let mut m_i = T::full::<f32>(&[HEAD_DIM], neg_inf);
    let mut l_i = T::zeros::<f32>(&[HEAD_DIM]);
    let scale_t = T::full::<f32>(&[HEAD_DIM], softmax_scale);

    for k_row in 0..n_ctx_k {
        let kv_row_base = kv_bh_base + k_row * HEAD_DIM;

        let k_vec = T::load(
            k_ptr.add_offsets(d + kv_row_base),
            None,
            None,
            &[],
            None,
            None,
            None,
            false,
        );
        let v_vec = T::load(
            v_ptr.add_offsets(d + kv_row_base),
            None,
            None,
            &[],
            None,
            None,
            None,
            false,
        );

        // Scaled dot-product: score = sum(q · k) * scale  → [HD] (scalar replicated)
        let qk = T::sum(q_vec * k_vec, Some(0), true) * scale_t;

        // Online softmax recurrence
        let m_new = T::maximum(m_i, qk); // [HD] running max (all elements equal)
        let exp_diff = T::exp(m_i - m_new); // [HD] correction factor
        let p = T::exp(qk - m_new); // [HD] unnorm weight for this k

        l_i = exp_diff * l_i + p;
        acc = exp_diff * acc + p * v_vec;
        m_i = m_new;
    }

    // Normalise output and compute logsumexp for backward.
    let o_row = acc / l_i; // [HD] / [HD] → [HD]
    // All elements of m_i and l_i are equal (replicated scalar). Sum and divide
    // by HEAD_DIM to recover the scalar as tensor<1xf32> for the l_ptr store.
    let l_save_sum = T::sum(m_i + T::log(l_i), Some(0), false); // scalar f32
    let l_save = l_save_sum / T::full::<f32>(&[1], HEAD_DIM as f32); // tensor<1xf32>

    T::store(o_ptr.add_offsets(d + o_row_base), o_row, None, &[], None, None);
    T::store(
        l_ptr.add_offsets(T::arange(0, 1) + l_row_base),
        l_save,
        None,
        &[],
        None,
        None,
    );
}

// ── Backward — dQ ─────────────────────────────────────────────────────────────

/// Flash Attention 2 backward: computes `dQ`.
///
/// For each query row `q`, iterates over all key rows `k` and accumulates:
/// ```text
/// dQ_q += dS_{qk} * K_k * scale
/// where dS_{qk} = p_{qk} * (dO_q · V_k − D_q)
///       p_{qk}  = exp(Q_q · K_k * scale − L_q)   (recomputed attention)
///       D_q     = sum(O_q * dO_q)                 (per-row scalar)
/// ```
///
/// Grid: `(N_CTX_Q, BH, 1)` — same grid shape as the forward pass.
#[kernel]
pub fn flash_attention2_backward_dq<T: Triton, const HEAD_DIM: i32>(
    q_ptr: T::Pointer<f32>,
    k_ptr: T::Pointer<f32>,
    v_ptr: T::Pointer<f32>,
    o_ptr: T::Pointer<f32>,
    do_ptr: T::Pointer<f32>,
    l_ptr: T::Pointer<f32>,
    dq_ptr: T::Pointer<f32>,
    n_ctx_q: i32,
    n_ctx_k: i32,
    softmax_scale: f32,
) where
    T::I32Tensor: Tensor<i32, 1>,
    T::I32Tensor: Comparison<i32, BoolTensor = T::BoolTensor>,
    T::Pointer<f32>: AddOffsets<i32, 1, T::I32Tensor, Output = T::Tensor<T::Pointer<f32>>>,
{
    let pid_m = T::program_id(Axis::X);
    let pid_bh = T::program_id(Axis::Y);

    let q_row_base = pid_bh * n_ctx_q * HEAD_DIM + pid_m * HEAD_DIM;
    let kv_bh_base = pid_bh * n_ctx_k * HEAD_DIM;
    let l_row_base = pid_bh * n_ctx_q + pid_m;

    let d = T::arange(0, HEAD_DIM);
    let scale_t = T::full::<f32>(&[HEAD_DIM], softmax_scale);

    let q_vec = T::load(
        q_ptr.add_offsets(d + q_row_base),
        None,
        None,
        &[],
        None,
        None,
        None,
        false,
    );
    let o_vec = T::load(
        o_ptr.add_offsets(d + q_row_base),
        None,
        None,
        &[],
        None,
        None,
        None,
        false,
    );
    let do_vec = T::load(
        do_ptr.add_offsets(d + q_row_base),
        None,
        None,
        &[],
        None,
        None,
        None,
        false,
    );

    // D_q = rowsum(O_q * dO_q)  — scalar
    let d_q = T::sum(o_vec * do_vec, Some(0), false);

    // Load logsumexp L_q (scalar stored as 1-element vec); reduce to scalar.
    let l_q_raw = T::load(
        l_ptr.add_offsets(T::arange(0, 1) + l_row_base),
        None,
        None,
        &[],
        None,
        None,
        None,
        false,
    );
    let l_q = T::sum(l_q_raw, Some(0), false); // scalar f32

    let mut dq_acc = T::zeros::<f32>(&[HEAD_DIM]);

    for k_row in 0..n_ctx_k {
        let kv_row_base = kv_bh_base + k_row * HEAD_DIM;

        let k_vec = T::load(
            k_ptr.add_offsets(d + kv_row_base),
            None,
            None,
            &[],
            None,
            None,
            None,
            false,
        );
        let v_vec = T::load(
            v_ptr.add_offsets(d + kv_row_base),
            None,
            None,
            &[],
            None,
            None,
            None,
            false,
        );

        // Recompute attention weight: p = exp(qk * scale - L_q) — [HEAD_DIM] (scalar replicated)
        let qk = T::sum(q_vec * k_vec, Some(0), false) * scale_t;
        let p = T::exp(qk - l_q);

        // dS = p * (dO · V_k - D_q)
        let do_dot_v = T::sum(do_vec * v_vec, Some(0), false);
        let ds = p * (do_dot_v - d_q);

        // dQ += dS * K_k * scale
        dq_acc = dq_acc + ds * k_vec * scale_t;
    }

    T::store(
        dq_ptr.add_offsets(d + q_row_base),
        dq_acc,
        None,
        &[],
        None,
        None,
    );
}

// ── Backward — dK / dV ────────────────────────────────────────────────────────

/// Flash Attention 2 backward: computes `dK` and `dV` for one key row.
///
/// For each key row `n`, iterates over all query rows `m` and accumulates:
/// ```text
/// dV_n += p_{mn} * dO_m
/// dK_n += dS_{mn} * Q_m * scale
/// where p_{mn}  = exp(Q_m · K_n * scale − L_m)
///       dS_{mn} = p_{mn} * (dO_m · V_n − D_m)
///       D_m     = sum(O_m * dO_m)
/// ```
///
/// Each CTA owns an exclusive key row so no atomic operations are needed.
///
/// Grid: `(N_CTX_K, BH, 1)`.
#[kernel]
pub fn flash_attention2_backward_dkv<T: Triton, const HEAD_DIM: i32>(
    q_ptr: T::Pointer<f32>,
    k_ptr: T::Pointer<f32>,
    v_ptr: T::Pointer<f32>,
    o_ptr: T::Pointer<f32>,
    do_ptr: T::Pointer<f32>,
    l_ptr: T::Pointer<f32>,
    dk_ptr: T::Pointer<f32>,
    dv_ptr: T::Pointer<f32>,
    n_ctx_q: i32,
    n_ctx_k: i32,
    softmax_scale: f32,
) where
    T::I32Tensor: Tensor<i32, 1>,
    T::I32Tensor: Comparison<i32, BoolTensor = T::BoolTensor>,
    T::Pointer<f32>: AddOffsets<i32, 1, T::I32Tensor, Output = T::Tensor<T::Pointer<f32>>>,
{
    let pid_n = T::program_id(Axis::X); // key-row index  [0, n_ctx_k)
    let pid_bh = T::program_id(Axis::Y); // (batch, head)  [0, BH)

    let q_bh_base = pid_bh * n_ctx_q * HEAD_DIM;
    let kv_row_base = pid_bh * n_ctx_k * HEAD_DIM + pid_n * HEAD_DIM;
    let l_bh_base = pid_bh * n_ctx_q;

    let d = T::arange(0, HEAD_DIM);
    let scale_t = T::full::<f32>(&[HEAD_DIM], softmax_scale);

    // Load K_n and V_n — fixed for this CTA.
    let k_vec = T::load(
        k_ptr.add_offsets(d + kv_row_base),
        None,
        None,
        &[],
        None,
        None,
        None,
        false,
    );
    let v_vec = T::load(
        v_ptr.add_offsets(d + kv_row_base),
        None,
        None,
        &[],
        None,
        None,
        None,
        false,
    );

    let mut dk_acc = T::zeros::<f32>(&[HEAD_DIM]);
    let mut dv_acc = T::zeros::<f32>(&[HEAD_DIM]);

    for q_row in 0..n_ctx_q {
        let q_row_base = q_bh_base + q_row * HEAD_DIM;
        let l_row_base = l_bh_base + q_row;

        let q_vec_m = T::load(
            q_ptr.add_offsets(d + q_row_base),
            None,
            None,
            &[],
            None,
            None,
            None,
            false,
        );
        let o_vec_m = T::load(
            o_ptr.add_offsets(d + q_row_base),
            None,
            None,
            &[],
            None,
            None,
            None,
            false,
        );
        let do_vec_m = T::load(
            do_ptr.add_offsets(d + q_row_base),
            None,
            None,
            &[],
            None,
            None,
            None,
            false,
        );
        // Load logsumexp L_m; reduce tensor<1xf32> to scalar.
        let l_m_raw = T::load(
            l_ptr.add_offsets(T::arange(0, 1) + l_row_base),
            None,
            None,
            &[],
            None,
            None,
            None,
            false,
        );
        let l_m = T::sum(l_m_raw, Some(0), false); // scalar f32

        // D_m = rowsum(O_m * dO_m) — scalar
        let d_m = T::sum(o_vec_m * do_vec_m, Some(0), false);

        // Recompute p_{mn} = exp(Q_m · K_n * scale - L_m) — [HEAD_DIM] (scalar replicated)
        let qk = T::sum(q_vec_m * k_vec, Some(0), false) * scale_t;
        let p = T::exp(qk - l_m);

        // dV += p * dO_m
        dv_acc = dv_acc + p * do_vec_m;

        // dS = p * (dO_m · V_n - D_m)
        let do_dot_v = T::sum(do_vec_m * v_vec, Some(0), false);
        let ds = p * (do_dot_v - d_m);

        // dK += dS * Q_m * scale
        dk_acc = dk_acc + ds * q_vec_m * scale_t;
    }

    T::store(
        dk_ptr.add_offsets(d + kv_row_base),
        dk_acc,
        None,
        &[],
        None,
        None,
    );
    T::store(
        dv_ptr.add_offsets(d + kv_row_base),
        dv_acc,
        None,
        &[],
        None,
        None,
    );
}

// ── Op wrapper ────────────────────────────────────────────────────────────────

pub struct FlashAttention2Op<'a> {
    pub forward: FlashAttention2Forward,
    pub backward_dq: FlashAttention2BackwardDq,
    pub backward_dkv: FlashAttention2BackwardDkv,
    _marker: PhantomData<&'a ()>,
}

impl<'a> FlashAttention2Op<'a> {
    pub fn new(head_dim: i32) -> Self {
        Self {
            forward: FlashAttention2Forward::new(head_dim),
            backward_dq: FlashAttention2BackwardDq::new(head_dim),
            backward_dkv: FlashAttention2BackwardDkv::new(head_dim),
            _marker: PhantomData,
        }
    }
}
