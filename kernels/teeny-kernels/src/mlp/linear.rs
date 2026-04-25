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

use teeny_core::dtype::{AddOffsets, Comparison, Float, Tensor};
use teeny_macros::kernel;
use teeny_triton::triton::{Axis, PaddingOption, Triton};

#[kernel]
pub fn linear_forward<
    T: Triton,
    D: Float,
    const USE_BIAS: bool,
    const BLOCK_M: i32,
    const BLOCK_N: i32,
    const BLOCK_K: i32,
    const GROUP_M: i32,
>(
    x_ptr: T::Pointer<D>,
    w_ptr: T::Pointer<D>,
    b_ptr: T::Pointer<D>,
    y_ptr: T::Pointer<D>,
    M: i32,
    N: i32,
    K: i32,
    stride_xm: i32,
    stride_xk: i32,
    stride_wn: i32,
    stride_wk: i32,
    stride_ym: i32,
    stride_yn: i32,
) where
    T::I32Tensor: Tensor<i32, 1>,
    T::I32Tensor: Comparison<i32, BoolTensor = T::BoolTensor>,
    T::Pointer<D>: AddOffsets<i32, 1, T::I32Tensor, Output = T::Tensor<T::Pointer<D>>>,
{
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

    let x_desc = T::make_tensor_descriptor(
        x_ptr,
        &[M, K],
        &[stride_xm, stride_xk],
        &[BLOCK_M, BLOCK_K],
        Some(PaddingOption::Zero),
    );
    let w_desc = T::make_tensor_descriptor(
        w_ptr,
        &[N, K],
        &[stride_wn, stride_wk],
        &[BLOCK_N, BLOCK_K],
        Some(PaddingOption::Zero),
    );

    let mut acc = T::zeros::<D>(&[BLOCK_M, BLOCK_N]);
    let k_tiles = T::cdiv(K, BLOCK_K);
    for k in 0..k_tiles {
        let x = T::load_tensor_descriptor(x_desc, &[pid_m * BLOCK_M, k * BLOCK_K]);
        let w = T::load_tensor_descriptor(w_desc, &[pid_n * BLOCK_N, k * BLOCK_K]);
        let w_t = T::trans(w, &[1, 0]);
        acc = T::dot::<D, D>(x, w_t, Some(acc), None, None);
    }

    if USE_BIAS {
        let offs_bn = T::arange(0, BLOCK_N) + pid_n * BLOCK_N;
        let bias_mask = offs_bn.lt(N);
        let bias = T::load(
            b_ptr.add_offsets(offs_bn),
            Some(bias_mask),
            Some(T::zeros::<D>(&[BLOCK_N])),
            &[],
            None,
            None,
            None,
            false,
        );
        let bias = T::expand_dims(bias, 0);
        let bias = T::broadcast_to(bias, &[BLOCK_M, BLOCK_N]);
        acc = acc + bias;
    }

    let y_desc = T::make_tensor_descriptor(
        y_ptr,
        &[M, N],
        &[stride_ym, stride_yn],
        &[BLOCK_M, BLOCK_N],
        Some(PaddingOption::Zero),
    );

    T::store_tensor_descriptor(y_desc, &[pid_m * BLOCK_M, pid_n * BLOCK_N], acc);
}

#[kernel]
pub fn linear_backward<
    T: Triton,
    D: Float,
    const USE_BIAS: bool,
    const BLOCK_M: i32,
    const BLOCK_N: i32,
    const BLOCK_K: i32,
    const GROUP_M: i32,
>(
    x_ptr: T::Pointer<D>,
    w_ptr: T::Pointer<D>,
    dy_ptr: T::Pointer<D>,
    dx_ptr: T::Pointer<D>,
    dw_ptr: T::Pointer<D>,
    db_ptr: T::Pointer<D>,
    M: i32,
    N: i32,
    K: i32,
    stride_xm: i32,
    stride_xk: i32,
    stride_wk: i32,
    stride_wn: i32,
    stride_dym: i32,
    stride_dyn: i32,
    stride_dxm: i32,
    stride_dxk: i32,
    stride_dwk: i32,
    stride_dwn: i32,
    stride_dbn: i32,
) where
    T::I32Tensor: Tensor<i32, 1>,
    T::I32Tensor: Comparison<i32, BoolTensor = T::BoolTensor>,
    T::Pointer<D>: AddOffsets<i32, 1, T::I32Tensor, Output = T::Tensor<T::Pointer<D>>>,
{
    // 3D grid: pid encodes (pid_m, pid_n, pid_k) as a flat index.
    // pid_k = pid % num_pid_k
    // pid_n = (pid / num_pid_k) % num_pid_n
    // pid_m = pid / (num_pid_k * num_pid_n)
    //
    // Each CTA computes exactly ONE (BLOCK_M, BLOCK_K) tile of dx and ONE (BLOCK_N, BLOCK_K)
    // tile of dw, using single-level inner loops over N and M respectively.
    // Guards (pid_n == 0 for dx, pid_m == 0 for dw) prevent multiple CTAs writing the same tile.
    let pid = T::program_id(Axis::X);
    let num_pid_k = T::cdiv(K, BLOCK_K);
    let num_pid_n = T::cdiv(N, BLOCK_N);
    let pid_k = pid % num_pid_k;
    let pid_tmp = pid / num_pid_k;
    let pid_n = pid_tmp % num_pid_n;
    let pid_m = pid_tmp / num_pid_n;

    let x_desc = T::make_tensor_descriptor(
        x_ptr,
        &[M, K],
        &[stride_xm, stride_xk],
        &[BLOCK_M, BLOCK_K],
        Some(PaddingOption::Zero),
    );
    let w_desc = T::make_tensor_descriptor(
        w_ptr,
        &[N, K],
        &[stride_wn, stride_wk],
        &[BLOCK_N, BLOCK_K],
        Some(PaddingOption::Zero),
    );
    let dy_desc = T::make_tensor_descriptor(
        dy_ptr,
        &[M, N],
        &[stride_dym, stride_dyn],
        &[BLOCK_M, BLOCK_N],
        Some(PaddingOption::Zero),
    );

    // -----------------
    // Compute dx = dy @ W for tile (pid_m, pid_k).
    // Only CTAs with pid_n == 0 write; others skip to avoid racing writes.
    // -----------------
    let dx_desc = T::make_tensor_descriptor(
        dx_ptr,
        &[M, K],
        &[stride_dxm, stride_dxk],
        &[BLOCK_M, BLOCK_K],
        Some(PaddingOption::Zero),
    );
    if pid_n == 0 {
        let n_tiles = T::cdiv(N, BLOCK_N);
        let mut acc_dx = T::zeros::<D>(&[BLOCK_M, BLOCK_K]);
        for n in 0..n_tiles {
            let dy = T::load_tensor_descriptor(dy_desc, &[pid_m * BLOCK_M, n * BLOCK_N]);
            let w = T::load_tensor_descriptor(w_desc, &[n * BLOCK_N, pid_k * BLOCK_K]);
            acc_dx = T::dot::<D, D>(dy, w, Some(acc_dx), None, None);
        }
        T::store_tensor_descriptor(dx_desc, &[pid_m * BLOCK_M, pid_k * BLOCK_K], acc_dx);
    }

    // -----------------
    // Compute dw = dy.T @ x for tile (pid_n, pid_k).
    // Only CTAs with pid_m == 0 write; others skip to avoid racing writes.
    // -----------------
    let dw_desc = T::make_tensor_descriptor(
        dw_ptr,
        &[N, K],
        &[stride_dwn, stride_dwk],
        &[BLOCK_N, BLOCK_K],
        Some(PaddingOption::Zero),
    );
    if pid_m == 0 {
        let m_tiles = T::cdiv(M, BLOCK_M);
        let mut acc_dw = T::zeros::<D>(&[BLOCK_N, BLOCK_K]);
        for m in 0..m_tiles {
            let dy = T::load_tensor_descriptor(dy_desc, &[m * BLOCK_M, pid_n * BLOCK_N]);
            let x = T::load_tensor_descriptor(x_desc, &[m * BLOCK_M, pid_k * BLOCK_K]);
            let dy_t = T::trans(dy, &[1, 0]);
            acc_dw = T::dot::<D, D>(dy_t, x, Some(acc_dw), None, None);
        }
        T::store_tensor_descriptor(dw_desc, &[pid_n * BLOCK_N, pid_k * BLOCK_K], acc_dw);

        // db = sum(dy, dim=0): computed alongside dw (pid_m == 0), only for pid_k == 0.
        if USE_BIAS {
            if pid_k == 0 {
                let offs_bn = T::arange(0, BLOCK_N) + pid_n * BLOCK_N;
                let bias_mask = offs_bn.lt(N);
                let mut acc_db = T::zeros::<D>(&[BLOCK_N]);
                for m in 0..m_tiles {
                    let dy = T::load_tensor_descriptor(dy_desc, &[m * BLOCK_M, pid_n * BLOCK_N]);
                    let sum = T::sum::<D>(dy, Some(0), false);
                    acc_db = acc_db + sum;
                }
                let db_ptr_tile = db_ptr.add_offsets(offs_bn);
                T::store(db_ptr_tile, acc_db, Some(bias_mask), &[], None, None);
            }
        }
    }
}

pub struct LinearOp<'a, T: Float> {
    pub forward: LinearForward<T>,
    pub backward: LinearBackward<T>,
    _marker: core::marker::PhantomData<&'a ()>,
}
