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

#![allow(non_snake_case)]

use core::ops::BitAnd;

use teeny_macros::kernel;
use teeny_triton::triton::{
    types::{AddOffsets, Comparison, Tensor},
    *,
};

/// Fused Conv2d(1×1, stride=1, pad=0, groups=1) + BatchNorm2d + SiLU using GEMM.
///
/// A 1×1 stride=1 no-padding convolution is mathematically equivalent to:
///   Y[N, M] = W[N, K] @ X[K, M]
/// where:
///   N = C_OUT, K = C_IN, M = OH * OW  (batch handled in grid)
///
/// Input X is stored NCHW = [B, K, OH, OW], viewed as 2-D [K, OH*OW] per batch
/// (column-major spatial: stride_K = OH*OW, stride_spatial = 1).
///
/// Weight W is [C_OUT, C_IN] row-major.
///
/// Output Y is viewed as [B*C_OUT, y_row_stride] where `y_row_stride >= M` and
/// `y_row_stride` is a multiple of `BLOCK_M`. TMA stores a full `[BLOCK_N,
/// BLOCK_M]` tile; without that padding the last M-tile writes past `M` into
/// the next channel (silent overwrite). Runtime allocates the padded buffer
/// and depads back to tight NCHW.
///
/// T::dot uses TF32 Tensor Cores on sm_87+ (Jetson Orin) for ~8× throughput
/// vs direct scalar accumulation.
///
/// Restrictions (enforced by dispatch in graph/mod.rs):
///   - kernel_h == 1, kernel_w == 1
///   - stride_h == 1, stride_w == 1
///   - padding_h == 0, padding_w == 0
///   - groups == 1
///
/// BN parameters must be precomputed (same convention as conv2d_bn_silu_forward).
///
/// Grid: pid = b * num_pid_per_batch + group_id * (GROUP_M * num_pid_n)
///             + pid_in_group  (same L2-locality grouping as Triton's matmul tutorial)
///
/// Inference-only; no backward pass.
#[kernel]
pub fn conv2d_bn_silu_gemm_forward<
    T: Triton,
    const BLOCK_M: i32,
    const BLOCK_N: i32,
    const BLOCK_K: i32,
    const GROUP_M: i32,
>(
    x_ptr: InPtr<T::Pointer<f32>>,
    w_ptr: InPtr<T::Pointer<f32>>,
    bn_scale_ptr: InPtr<T::Pointer<f32>>,
    bn_shift_ptr: InPtr<T::Pointer<f32>>,
    y_ptr: OutPtr<T::Pointer<f32>>,
    B: i32,
    C_IN: i32,
    C_OUT: i32,
    M: i32, // OH * OW per batch (tight spatial extent)
    // y_row_stride: allocated column width per (b, c_out) row.
    //   Must satisfy: y_row_stride >= M AND divisible by BLOCK_M.
    //   Prevents the last M-tile's TMA store from spilling into the next channel.
    y_row_stride: i32,
) where
    T::I32Tensor: Tensor<i32, 1>,
    T::I32Tensor: Comparison<i32, BoolTensor = T::BoolTensor>,
    T::BoolTensor: BitAnd<Output = T::BoolTensor>,
    T::Pointer<f32>: AddOffsets<i32, 1, T::I32Tensor, Output = T::Tensor<T::Pointer<f32>>>,
{
    let pid = T::program_id(Axis::X);

    let num_pid_m = T::cdiv(M, BLOCK_M);
    let num_pid_n = T::cdiv(C_OUT, BLOCK_N);
    let pids_per_batch = num_pid_m * num_pid_n;

    // Decode batch dimension from pid
    let b = pid / pids_per_batch;
    let pid_local = pid % pids_per_batch;

    // L2-locality grouping (same as Triton matmul tutorial)
    let num_pid_in_group = GROUP_M * num_pid_n;
    let group_id = pid_local / num_pid_in_group;
    let first_pid_m = group_id * GROUP_M;
    let remaining_m = num_pid_m - first_pid_m;
    let group_size_m = if remaining_m < GROUP_M {
        remaining_m
    } else {
        GROUP_M
    };
    let pid_in_group = pid_local % num_pid_in_group;
    let pid_m = first_pid_m + (pid_in_group % group_size_m);
    let pid_n = pid_in_group / group_size_m;

    // ── Input descriptor: X viewed as [B*C_IN, M] ────────────────────────────
    // NCHW layout: x[b, c_in, oh, ow] = x_flat[b*(C_IN*M) + c_in*M + oh*OW + ow]
    // For batch b: the C_IN rows for that batch start at row b*C_IN.
    // stride along K (row) = M (spatial extent per channel)
    // stride along M (col) = 1
    let x_desc = T::make_tensor_descriptor(
        x_ptr,
        &[B * C_IN, M],
        &[M, 1],
        &[BLOCK_K, BLOCK_M],
        Some(PaddingOption::Zero),
    );

    // ── Weight descriptor: W [C_OUT, C_IN] row-major ─────────────────────────
    let w_desc = T::make_tensor_descriptor(
        w_ptr,
        &[C_OUT, C_IN],
        &[C_IN, 1],
        &[BLOCK_N, BLOCK_K],
        Some(PaddingOption::Zero),
    );

    // ── GEMM: acc [BLOCK_N, BLOCK_M] = sum_k W_tile @ X_tile ─────────────────
    let mut acc = T::zeros::<f32>(&[BLOCK_N, BLOCK_M]);
    let k_tiles = T::cdiv(C_IN, BLOCK_K);
    for k in 0..k_tiles {
        // x_tile: [BLOCK_K, BLOCK_M] — rows are the C_IN channels for batch b
        let x_tile = T::load_tensor_descriptor(x_desc, &[b * C_IN + k * BLOCK_K, pid_m * BLOCK_M]);

        // w_tile: [BLOCK_N, BLOCK_K]
        let w_tile = T::load_tensor_descriptor(w_desc, &[pid_n * BLOCK_N, k * BLOCK_K]);

        // [BLOCK_N, BLOCK_K] @ [BLOCK_K, BLOCK_M] → [BLOCK_N, BLOCK_M]
        // TF32 precision is what actually routes this dot to the tensor-core MMA
        // path (see getMmaTypeDot in Triton's MMAv2.cpp) — IEEE forces the
        // software FMA fallback and silently disables tensor cores entirely.
        acc = T::dot::<f32, f32>(w_tile, x_tile, Some(acc), InputPrecision::TF32, None);
    }

    // ── BatchNorm epilog ──────────────────────────────────────────────────────
    let bn_off = T::arange(0, BLOCK_N) + pid_n * BLOCK_N;
    let bn_n_mask = bn_off.lt(C_OUT);
    let bn_scale = T::load(
        bn_scale_ptr.add_offsets(bn_off),
        Some(bn_n_mask),
        Some(T::zeros::<f32>(&[BLOCK_N])),
        &[],
        None,
        None,
        None,
        false,
    );
    let bn_shift = T::load(
        bn_shift_ptr.add_offsets(bn_off),
        Some(bn_n_mask),
        Some(T::zeros::<f32>(&[BLOCK_N])),
        &[],
        None,
        None,
        None,
        false,
    );
    let scale_2d = T::broadcast_to(T::expand_dims(bn_scale, 1), &[BLOCK_N, BLOCK_M]);
    let shift_2d = T::broadcast_to(T::expand_dims(bn_shift, 1), &[BLOCK_N, BLOCK_M]);
    let bn_out = scale_2d * acc + shift_2d;

    // ── SiLU epilog: y = x * sigmoid(x) ─────────────────────────────────────
    let y = bn_out * T::sigmoid(bn_out);

    // ── Store: y NCHW [B, C_OUT, OH, OW] viewed as [B*C_OUT, y_row_stride] ───
    // Valid columns are [0, M); columns [M, y_row_stride) are padding so the
    // last BLOCK_M-wide TMA store cannot spill into the next channel's row.
    let y_desc = T::make_tensor_descriptor(
        y_ptr,
        &[B * C_OUT, y_row_stride],
        &[y_row_stride, 1],
        &[BLOCK_N, BLOCK_M],
        Some(PaddingOption::Zero),
    );
    T::store_tensor_descriptor(y_desc, &[b * C_OUT + pid_n * BLOCK_N, pid_m * BLOCK_M], y);
}

// ── RuntimeOp ────────────────────────────────────────────────────────────────
//
// Params layout: [weight [C_OUT, C_IN], bn_scale [C_OUT], bn_shift [C_OUT]]
// pack_args order: x_ptr, w_ptr, bn_scale_ptr, bn_shift_ptr, y_ptr,
//                  B, C_IN, C_OUT, M (= OH * OW), y_row_stride

impl teeny_core::model::RuntimeOp for Conv2dBnSiluGemmForward {
    fn n_activation_inputs(&self) -> usize {
        1
    }

    fn param_shapes(&self, input_shapes: &[&[usize]], output_shape: &[usize]) -> Vec<Vec<usize>> {
        let c_in = input_shapes[0][1];
        let c_out = output_shape[1];
        // 1×1 weight: [C_OUT, C_IN] (no KH/KW dims needed — both are 1)
        vec![vec![c_out, c_in], vec![c_out], vec![c_out]]
    }

    fn param_names(&self) -> &'static [&'static str] {
        &["weight", "bn_scale", "bn_shift"]
    }

    // Row = one (b, c_out) channel flattened over OH*OW. Pad to a multiple of
    // BLOCK_M so the last TMA store tile stays inside the channel's row.
    fn forward_output_row_elems(&self, output_shape: &[usize]) -> usize {
        output_shape[2] * output_shape[3]
    }

    fn forward_output_row_stride(&self, output_shape: &[usize]) -> usize {
        let m = self.forward_output_row_elems(output_shape);
        m.next_multiple_of(self.block_m as usize)
    }

    fn pack_args(
        &self,
        inputs: &[(teeny_core::model::RawPtr, &[usize])],
        params: &[teeny_core::model::RawPtr],
        output: teeny_core::model::RawPtr,
        output_shape: &[usize],
        output_row_stride: i32,
        visitor: &mut dyn teeny_core::device::program::ArgVisitor,
    ) {
        let input_shape = inputs[0].1;
        let b = input_shape[0] as i32;
        let c_in = input_shape[1] as i32;
        let c_out = output_shape[1] as i32;
        let m = (output_shape[2] * output_shape[3]) as i32; // OH * OW
        visitor.visit_ptr(inputs[0].0); // x_ptr
        visitor.visit_ptr(params[0]); // w_ptr
        visitor.visit_ptr(params[1]); // bn_scale_ptr
        visitor.visit_ptr(params[2]); // bn_shift_ptr
        visitor.visit_ptr(output); // y_ptr
        visitor.visit_i32(b);
        visitor.visit_i32(c_in);
        visitor.visit_i32(c_out);
        visitor.visit_i32(m);
        visitor.visit_i32(output_row_stride); // y_row_stride
    }

    fn grid(&self, output_shape: &[usize]) -> [u32; 3] {
        let b = output_shape[0];
        let m = output_shape[2] * output_shape[3]; // OH * OW
        let c_out = output_shape[1];
        let pm = m.div_ceil(self.block_m as usize);
        let pn = c_out.div_ceil(self.block_n as usize);
        [(b * pm * pn) as u32, 1, 1]
    }
}
