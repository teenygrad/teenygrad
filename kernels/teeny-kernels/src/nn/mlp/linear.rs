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

use teeny_core::dtype::{AddOffsets, Comparison, Num, Tensor};
use teeny_macros::kernel;
use teeny_triton::triton::{Axis, PaddingOption, Triton};

#[kernel]
pub fn linear_forward<
    T: Triton,
    D: Num,
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
    D: Num,
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

impl<D: Num + Send + Sync + 'static> teeny_core::model::RuntimeOp for LinearForward<D> {
    fn n_activation_inputs(&self) -> usize { 1 }

    fn param_shapes(&self, input_shapes: &[&[usize]], output_shape: &[usize]) -> Vec<Vec<usize>> {
        // input_shapes[0] = [M, K], output_shape = [M, N]
        let k = input_shapes[0][1];
        let n = output_shape[1];
        // weight: [N, K]; optional bias: [N]
        if self.use_bias {
            vec![vec![n, k], vec![n]]
        } else {
            vec![vec![n, k]]
        }
    }

    fn pack_args(
        &self,
        inputs: &[(teeny_core::model::RawPtr, &[usize])],
        params: &[teeny_core::model::RawPtr],
        output: teeny_core::model::RawPtr,
        output_shape: &[usize],
        visitor: &mut dyn teeny_core::device::program::ArgVisitor,
    ) {
        // kernel args: x_ptr, w_ptr, b_ptr, y_ptr, M, N, K,
        //              stride_xm, stride_xk, stride_wn, stride_wk, stride_ym, stride_yn
        let m = output_shape[0] as i32;
        let n = output_shape[1] as i32;
        let k = inputs[0].1[1] as i32;
        let b_ptr = if self.use_bias { params[1] } else { core::ptr::null_mut() };
        visitor.visit_ptr(inputs[0].0);   // x_ptr
        visitor.visit_ptr(params[0]);     // w_ptr
        visitor.visit_ptr(b_ptr);         // b_ptr
        visitor.visit_ptr(output);        // y_ptr
        visitor.visit_i32(m);             // M
        visitor.visit_i32(n);             // N
        visitor.visit_i32(k);             // K
        visitor.visit_i32(k);             // stride_xm = K (row-major)
        visitor.visit_i32(1);             // stride_xk = 1
        visitor.visit_i32(k);             // stride_wn = K (w is [N,K])
        visitor.visit_i32(1);             // stride_wk = 1
        visitor.visit_i32(n);             // stride_ym = N
        visitor.visit_i32(1);             // stride_yn = 1
    }

    fn block(&self) -> [u32; 3] { [128, 1, 1] }

    fn grid(&self, output_shape: &[usize]) -> [u32; 3] {
        // pid encodes (pid_m, pid_n) grouped by GROUP_M
        let pm = output_shape[0].div_ceil(self.block_m as usize);
        let pn = output_shape[1].div_ceil(self.block_n as usize);
        [(pm * pn) as u32, 1, 1]
    }

    #[cfg(feature = "training")]
    fn has_backward(&self) -> bool { true }

    // linear_backward(x_ptr, w_ptr, dy_ptr, dx_ptr, dw_ptr, db_ptr,
    //                 M, N, K,
    //                 stride_xm, stride_xk, stride_wk, stride_wn,
    //                 stride_dym, stride_dyn, stride_dxm, stride_dxk,
    //                 stride_dwk, stride_dwn, stride_dbn)
    #[cfg(feature = "training")]
    #[allow(clippy::too_many_arguments)]
    fn pack_backward_args(
        &self,
        inputs: &[(teeny_core::model::RawPtr, &[usize])],
        params: &[teeny_core::model::RawPtr],
        _output: teeny_core::model::RawPtr,
        output_shape: &[usize],
        grad_output: teeny_core::model::RawPtr,
        grad_inputs: &[teeny_core::model::RawPtr],
        grad_params: &[teeny_core::model::RawPtr],
        visitor: &mut dyn teeny_core::device::program::ArgVisitor,
    ) {
        let m = output_shape[0] as i32;
        let n = output_shape[1] as i32;
        let k = inputs[0].1[1] as i32;
        let db_ptr = if self.use_bias { grad_params[1] } else { core::ptr::null_mut() };

        visitor.visit_ptr(inputs[0].0);    // x_ptr
        visitor.visit_ptr(params[0]);      // w_ptr
        visitor.visit_ptr(grad_output);    // dy_ptr
        visitor.visit_ptr(grad_inputs[0]); // dx_ptr
        visitor.visit_ptr(grad_params[0]); // dw_ptr
        visitor.visit_ptr(db_ptr);         // db_ptr (null if no bias)
        visitor.visit_i32(m);              // M
        visitor.visit_i32(n);              // N
        visitor.visit_i32(k);              // K
        visitor.visit_i32(k);              // stride_xm = K (x is [M,K] row-major)
        visitor.visit_i32(1);              // stride_xk = 1
        visitor.visit_i32(1);              // stride_wk = 1
        visitor.visit_i32(k);              // stride_wn = K (w is [N,K])
        visitor.visit_i32(n);              // stride_dym = N (dy is [M,N] row-major)
        visitor.visit_i32(1);              // stride_dyn = 1
        visitor.visit_i32(k);              // stride_dxm = K
        visitor.visit_i32(1);             // stride_dxk = 1
        visitor.visit_i32(1);              // stride_dwk = 1
        visitor.visit_i32(k);              // stride_dwn = K
        visitor.visit_i32(1);              // stride_dbn = 1
    }

    #[cfg(feature = "training")]
    fn backward_block(&self) -> [u32; 3] { [128, 1, 1] }

    // Grid: ceil(M/BM) * ceil(N/BN) * ceil(K/BK) CTAs
    #[cfg(feature = "training")]
    fn backward_grid(&self, input_shapes: &[&[usize]], output_shape: &[usize]) -> [u32; 3] {
        let m = output_shape[0].div_ceil(self.block_m as usize);
        let n = output_shape[1].div_ceil(self.block_n as usize);
        let k = input_shapes[0][1].div_ceil(self.block_k as usize);
        [(m * n * k) as u32, 1, 1]
    }
}

pub struct LinearOp<'a, T: Num> {
    pub forward: LinearForward<T>,
    pub backward: LinearBackward<T>,
    _marker: core::marker::PhantomData<&'a ()>,
}
