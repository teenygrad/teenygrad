/*
 * Copyright (c) 2025 Teenygrad. All rights reserved.
 *
 * This program is free software: you can redistribute it and/or modify it
 * under the terms of the GNU General Public License as published by the
 * Free Software Foundation, either version 3 of the License, or (at your
 * option) any later version.
 *
 * This program is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program. If not, see <https://www.gnu.org/licenses/>.
 */

// use teeny_core::tensor::{self, Tensor, shape::DynamicShape};
// use teeny_macros::kernel;
// use teeny_triton::triton::{self, ConstExpr};

// #[allow(non_snake_case)]
// #[allow(clippy::too_many_arguments)]
// #[allow(clippy::excessive_precision)]
// #[allow(clippy::approx_constant)]
// #[kernel]
// fn fwd_kernel(
//     Q: &Tensor,
//     K: &Tensor,
//     V: &Tensor,
//     sm_scale: f32,
//     L: &Tensor,
//     Out: &Tensor,
//     _stride_qz: usize,
//     stride_qh: usize,
//     stride_qm: usize,
//     stride_qk: usize,
//     _stride_kz: usize,
//     _stride_kh: usize,
//     stride_kn: usize,
//     stride_kk: usize,
//     _stride_vz: usize,
//     _stride_vh: usize,
//     stride_vn: usize,
//     stride_vk: usize,
//     _stride_oz: usize,
//     _stride_oh: usize,
//     stride_om: usize,
//     stride_on: usize,
//     _Z: usize,
//     _H: usize,
//     N_CTX: usize,
//     Z_H_N_CTX: usize,
//     BLOCK_M: ConstExpr<usize>,
//     BLOCK_DMODEL: ConstExpr<usize>,
//     BLOCK_N: ConstExpr<usize>,
//     IS_CAUSAL: ConstExpr<bool>,
// ) {
//     let start_m = triton::program_id(0);
//     let off_hz = triton::program_id(1);
//     let qvk_offset = off_hz * stride_qh;
//     let vk_offset = triton::floor_div(qvk_offset, stride_qm);

//     let mut K_block_ptr: Tensor = triton::make_block_ptr(triton::Block {
//         base: K,
//         shape: [BLOCK_DMODEL.0, Z_H_N_CTX].into(),
//         strides: [stride_kk, stride_kn].into(),
//         offsets: [0, vk_offset].into(),
//         block_shape: [BLOCK_DMODEL.0, BLOCK_N.0].into(),
//         order: [0, 1].into(),
//     });

//     let mut V_block_ptr = triton::make_block_ptr(triton::Block {
//         base: V,
//         shape: [Z_H_N_CTX, BLOCK_DMODEL.0].into(),
//         strides: [stride_vn, stride_vk].into(),
//         offsets: [vk_offset, 0].into(),
//         block_shape: [BLOCK_N.0, BLOCK_DMODEL.0].into(),
//         order: [1, 0].into(),
//     });

//     // // initialize offsets
//     let offs_m = &triton::arange::<f32>(0.0, BLOCK_M.0 as f32, 1.0) + (start_m * BLOCK_M.0) as f32;
//     // let offs_n = triton::arange::<f32>(0.0, BLOCK_N.0 as f32, 1.0);

//     // // initialize pointer to m and l
//     let mut m_i = triton::zeros::<DynamicShape, f32>([BLOCK_M.0].into()) - f32::INFINITY;
//     let mut l_i = triton::zeros::<DynamicShape, f32>([BLOCK_M.0].into());
//     let mut acc = triton::zeros::<DynamicShape, f32>([BLOCK_M.0, BLOCK_DMODEL.0].into());

//     // credits to: Adam P. Goucher (https://github.com/apgoucher):
//     // scale sm_scale by 1/log_2(e) and use
//     // 2^x instead of exp in the loop because CSE and LICM
//     // don't work as expected with `exp` in the loop
//     let qk_scale = sm_scale * 1.44269504;

//     // load q: it will stay in SRAM throughout
//     let offs_k = triton::arange::<f32>(0.0, BLOCK_DMODEL.0 as f32, 1.0);
//     let Q_ptrs = Q
//         + (qvk_offset as f32)
//         + triton::append_axis(&offs_m) * (stride_qm as f32)
//         + triton::prepend_axis(&offs_k) * (stride_qk as f32);
//     let q = triton::load(&Q_ptrs);

//     let q = q * qk_scale;
//     let lo = 0;
//     let hi = if IS_CAUSAL.0 {
//         (start_m + 1) * BLOCK_M.0
//     } else {
//         N_CTX
//     };

//     for _start_n in (lo..hi).step_by(BLOCK_N.0) {
//         // -- load k, v --
//         let k = triton::load(&K_block_ptr);
//         let v = triton::load(&V_block_ptr);

//         // -- compute qk ---
//         let mut qk = triton::zeros::<DynamicShape, f32>([BLOCK_M.0, BLOCK_N.0].into());
//         if IS_CAUSAL.0 {
//             // AXM - qk = tl.where(
//             //     offs_m[:, None] >= (start_n + offs_n[None, :]), qk, float("-inf")
//             // )
//         }
//         qk += triton::dot(&q, &k);
//         // -- compute scaling constant ---
//         let m_i_new = triton::maximum(&m_i, &triton::max(&qk, 1.0));
//         let alpha = triton::exp2(&triton::sub(&m_i, &m_i_new));
//         let p = triton::exp2(&triton::sub(&qk, &triton::append_axis(&m_i_new)));
//         // // -- scale and update acc --
//         acc *= triton::append_axis(&alpha);
//         acc += triton::dot(&p, &v);
//         // // -- update m_i and l_i --
//         l_i = l_i * alpha + triton::sum(&p, 1);
//         m_i = m_i_new;
//         // // update pointers
//         K_block_ptr = triton::advance(&K_block_ptr, [0, BLOCK_N.0].into());
//         V_block_ptr = triton::advance(&V_block_ptr, [BLOCK_N.0, 0].into());
//     }

//     // // write back l and m
//     acc = acc / triton::append_axis(&l_i);

//     let l_ptrs = L + ((off_hz * N_CTX) as f32) + offs_m;
//     triton::store(&l_ptrs, &(m_i + triton::log2(&l_i)));
//     // // write back O
//     let O_block_ptr = triton::make_block_ptr(triton::Block {
//         base: Out,
//         shape: [Z_H_N_CTX, BLOCK_DMODEL.0].into(),
//         strides: [stride_om, stride_on].into(),
//         offsets: [vk_offset + start_m * BLOCK_M.0, 0].into(),
//         block_shape: [BLOCK_M.0, BLOCK_DMODEL.0].into(),
//         order: [1, 0].into(),
//     });

//     // O_ptrs = Out + qvk_offset + offs_m[:, None] * stride_qm + offs_k[None, :] * stride_qk
//     triton::store(&O_block_ptr, &acc)
// }

// #[allow(non_snake_case)]
// #[allow(clippy::too_many_arguments)]
// #[allow(clippy::excessive_precision)]
// #[allow(clippy::approx_constant)]
// #[kernel]
// fn forward(
//     _ctx: usize,
//     q: Tensor,
//     k: Tensor,
//     v: Tensor,
//     causal: bool,
//     sm_scale: f32,
//     _sequence_parallel: bool,
// ) {
//     let BLOCK_M = 128;
//     let BLOCK_N = 64;
//     // shape constraints
//     // AXM - TO DO
//     let _Lq = q.shape_of(-1);
//     let Lk = k.shape_of(-1);
//     let _Lv = v.shape_of(-1);

//     // AXM - TO DO
//     // assert Lq == Lk and Lk == Lv;
//     // assert Lk in {16, 32, 64, 128};
//     let o = tensor::empty_like(&q);
//     let _grid = (
//         tensor::cdiv(q.shape()[2], BLOCK_M),
//         q.shape()[0] * q.shape()[1],
//         1,
//     );
//     let L = tensor::empty(&[q.shape[0] * q.shape[1], q.shape[2]]);
//     // let num_warps = if Lk <= 64 { 4 } else { 8 };
//     fwd_kernel(
//         &q,
//         &k,
//         &v,
//         sm_scale,
//         &L,
//         &o,
//         q.stride(0),
//         q.stride(1),
//         q.stride(2),
//         q.stride(3),
//         k.stride(0),
//         k.stride(1),
//         k.stride(2),
//         k.stride(3),
//         v.stride(0),
//         v.stride(1),
//         v.stride(2),
//         v.stride(3),
//         o.stride(0),
//         o.stride(1),
//         o.stride(2),
//         o.stride(3),
//         q.shape_of(0),
//         q.shape_of(1),
//         q.shape_of(2),
//         q.shape_of(0) * q.shape_of(1) * q.shape_of(2),
//         ConstExpr(BLOCK_M),
//         ConstExpr(BLOCK_N),
//         ConstExpr(Lk),
//         ConstExpr(causal),
//     )
// }
// //         ctx.save_for_backward(q, k, v, o, L)
// //         ctx.grid = grid
// //         ctx.sm_scale = sm_scale
// //         ctx.BLOCK_DMODEL = Lk
// //         ctx.causal = causal
// //         ctx.sequence_parallel = sequence_parallel
// //         return o

// // #[allow(non_snake_case)]
// // #[allow(clippy::too_many_arguments)]
// // #[allow(clippy::excessive_precision)]
// // #[allow(clippy::approx_constant)]
// // #[kernel]
// // fn bwd_preprocess(
// //     Out: DenseTensor<DynamicShape, f32>,
// //     DO: DenseTensor<DynamicShape, f32>,
// //     Delta: DenseTensor<DynamicShape, f32>,
// //     BLOCK_M: ConstExpr<usize>,
// //     D_HEAD: ConstExpr<usize>,
// // ) {
// //     let off_m =
// //         triton::program_id(0) * BLOCK_M.0 + triton::arange::<f32>(0.0, BLOCK_M.0 as f32, 1.0);
// //     let off_n = triton::arange::<f32>(0.0, D_HEAD.0 as f32, 1.0);
// //     // load
// //     let o = triton::load(
// //         &(Out + triton::append_axis(&off_m) * (D_HEAD.0 as f32) + triton::prepend_axis(&off_n)),
// //     );
// //     let d0 = triton::load(
// //         &(DO + triton::append_axis(&off_m) * (D_HEAD.0 as f32) + triton::prepend_axis(&off_n)),
// //     );
// //     // compute
// //     let delta = triton::sum(&(o * d0), 1);
// //     // write-back
// //     triton::store(&(Delta + off_m), &delta);
// // }

// // def _bwd_kernel_one_col_block(
// //     Q,
// //     K,
// //     V,
// //     sm_scale,
// //     qk_scale,  #
// //     Out,
// //     DO,  #
// //     DQ,
// //     DK,
// //     DV,  #
// //     L,  #
// //     D,  #
// //     Q_block_ptr,
// //     K_block_ptr,
// //     V_block_ptr,  #
// //     DO_block_ptr,
// //     DQ_block_ptr,
// //     DK_block_ptr,
// //     DV_block_ptr,  #
// //     stride_dqa,
// //     stride_qz,
// //     stride_qh,
// //     stride_qm,
// //     stride_qk,  #
// //     stride_kz,
// //     stride_kh,
// //     stride_kn,
// //     stride_kk,  #
// //     stride_vz,
// //     stride_vh,
// //     stride_vn,
// //     stride_vk,  #
// //     Z,
// //     H,
// //     N_CTX,  #
// //     off_h,
// //     off_z,
// //     off_hz,
// //     start_n,
// //     num_block,  #
// //     BLOCK_M: tl.constexpr,
// //     BLOCK_DMODEL: tl.constexpr,  #
// //     BLOCK_N: tl.constexpr,  #
// //     SEQUENCE_PARALLEL: tl.constexpr,  #
// //     CAUSAL: tl.constexpr,  #
// //     MMA_V3: tl.constexpr,  #
// // ):
// //     if CAUSAL:
// //         lo = start_n * BLOCK_M
// //     else:
// //         lo = 0

// //     Q_offset = (off_z * stride_qz + off_h * stride_qh) // stride_qm
// //     DQ_offset = off_z * stride_qz + off_h * stride_qh
// //     K_offset = (off_z * stride_kz + off_h * stride_kh) // stride_kn
// //     V_offset = (off_z * stride_vz + off_h * stride_vh) // stride_vn
// //     if SEQUENCE_PARALLEL:
// //         DQ_offset += stride_dqa * start_n
// //     DQ_offset = DQ_offset // stride_qm

// //     Q_block_ptr = tl.advance(Q_block_ptr, (lo + Q_offset, 0))
// //     K_block_ptr = tl.advance(K_block_ptr, (start_n * BLOCK_M + K_offset, 0))
// //     V_block_ptr = tl.advance(V_block_ptr, (start_n * BLOCK_M + V_offset, 0))
// //     DO_block_ptr = tl.advance(DO_block_ptr, (lo + Q_offset, 0))
// //     DQ_block_ptr = tl.advance(DQ_block_ptr, (lo + DQ_offset, 0))
// //     DK_block_ptr = tl.advance(DK_block_ptr, (start_n * BLOCK_M + K_offset, 0))
// //     DV_block_ptr = tl.advance(DV_block_ptr, (start_n * BLOCK_M + V_offset, 0))

// //     # initialize row/col offsets
// //     offs_n = start_n * BLOCK_M + tl.arange(0, BLOCK_M)
// //     offs_m = tl.arange(0, BLOCK_N)
// //     # pointer to row-wise quantities in value-like data
// //     D_ptrs = D + off_hz * N_CTX
// //     l_ptrs = L + off_hz * N_CTX
// //     # initialize dv amd dk
// //     dv = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
// //     dk = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
// //     # k and v stay in SRAM throughout
// //     k = tl.load(K_block_ptr)
// //     v = tl.load(V_block_ptr)
// //     # loop over rows
// //     for start_m in range(lo, num_block * BLOCK_M, BLOCK_M):
// //         offs_m_curr = start_m + offs_m
// //         # load q, k, v, do on-chip
// //         q = tl.load(Q_block_ptr)
// //         # recompute p = softmax(qk, dim=-1).T
// //         # NOTE: `do` is pre-divided by `l`; no normalization here
// //         if CAUSAL:
// //             qk = tl.where(
// //                 offs_m_curr[:, None] >= (offs_n[None, :]), float(0.0), float("-inf")
// //             )
// //         else:
// //             qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
// //         qk += tl.dot(q, tl.trans(k))
// //         qk *= qk_scale
// //         l_i = tl.load(l_ptrs + offs_m_curr)
// //         p = tl.math.exp2(qk - l_i[:, None])
// //         # compute dv
// //         do = tl.load(DO_block_ptr)
// //         dv += tl.dot(tl.trans(p.to(Q.dtype.element_ty)), do)
// //         # compute dp = dot(v, do)
// //         Di = tl.load(D_ptrs + offs_m_curr)
// //         # dp = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32) - Di[:, None]
// //         dp = tl.dot(do, tl.trans(v))
// //         # compute ds = p * (dp - delta[:, None])
// //         ds = (p * (dp - Di[:, None]) * sm_scale).to(Q.dtype.element_ty)
// //         # compute dk = dot(ds.T, q)
// //         dk += tl.dot(tl.trans(ds), q)
// //         # compute dq
// //         if not SEQUENCE_PARALLEL:
// //             dq = tl.load(DQ_block_ptr)
// //             dq += tl.dot(ds, k)
// //             tl.store(DQ_block_ptr, dq.to(Q.dtype.element_ty))
// //         elif SEQUENCE_PARALLEL:
// //             if MMA_V3:
// //                 dq = tl.dot(ds, k)
// //             else:
// //                 # not work with mma v3, because M % 64 != 0
// //                 dq = tl.trans(tl.dot(tl.trans(k), tl.trans(ds)))
// //             tl.store(DQ_block_ptr, dq.to(Q.dtype.element_ty))

// //         # increment pointers
// //         DQ_block_ptr = tl.advance(DQ_block_ptr, (BLOCK_M, 0))
// //         Q_block_ptr = tl.advance(Q_block_ptr, (BLOCK_M, 0))
// //         DO_block_ptr = tl.advance(DO_block_ptr, (BLOCK_M, 0))
// //     # write-back
// //     tl.store(DV_block_ptr, dv.to(V.dtype.element_ty))
// //     tl.store(DK_block_ptr, dk.to(K.dtype.element_ty))

// // @jit
// // def _bwd_kernel(
// //     Q,
// //     K,
// //     V,
// //     sm_scale,  #
// //     Out,
// //     DO,  #
// //     DQ,
// //     DK,
// //     DV,  #
// //     L,  #
// //     D,  #
// //     stride_dqa,
// //     stride_qz,
// //     stride_qh,
// //     stride_qm,
// //     stride_qk,  #
// //     stride_kz,
// //     stride_kh,
// //     stride_kn,
// //     stride_kk,  #
// //     stride_vz,
// //     stride_vh,
// //     stride_vn,
// //     stride_vk,  #
// //     Z,
// //     H,
// //     N_CTX,  #
// //     Z_H_N_CTX,  #
// //     SQ_Z_H_N_CTX,  #
// //     BLOCK_M: tl.constexpr,
// //     BLOCK_DMODEL: tl.constexpr,  #
// //     BLOCK_N: tl.constexpr,  #
// //     SEQUENCE_PARALLEL: tl.constexpr,  #
// //     CAUSAL: tl.constexpr,  #
// //     MMA_V3: tl.constexpr,  #
// // ):
// //     qk_scale = sm_scale * 1.44269504
// //     off_hz = tl.program_id(0)
// //     off_z = off_hz // H
// //     off_h = off_hz % H

// //     Q_block_ptr = tl.make_block_ptr(
// //         base=Q,
// //         shape=(Z_H_N_CTX, BLOCK_DMODEL),
// //         strides=(stride_qm, stride_qk),
// //         offsets=(0, 0),
// //         block_shape=(BLOCK_M, BLOCK_DMODEL),
// //         order=(1, 0),
// //     )
// //     K_block_ptr = tl.make_block_ptr(
// //         base=K,
// //         shape=(Z_H_N_CTX, BLOCK_DMODEL),
// //         strides=(stride_kn, stride_kk),
// //         offsets=(0, 0),
// //         block_shape=(BLOCK_M, BLOCK_DMODEL),
// //         order=(1, 0),
// //     )
// //     V_block_ptr = tl.make_block_ptr(
// //         base=V,
// //         shape=(Z_H_N_CTX, BLOCK_DMODEL),
// //         strides=(stride_vn, stride_vk),
// //         offsets=(0, 0),
// //         block_shape=(BLOCK_M, BLOCK_DMODEL),
// //         order=(1, 0),
// //     )
// //     DO_block_ptr = tl.make_block_ptr(
// //         base=DO,
// //         shape=(Z_H_N_CTX, BLOCK_DMODEL),
// //         strides=(stride_qm, stride_qk),
// //         offsets=(0, 0),
// //         block_shape=(BLOCK_M, BLOCK_DMODEL),
// //         order=(1, 0),
// //     )
// //     if SEQUENCE_PARALLEL:
// //         DQ_block_ptr = tl.make_block_ptr(
// //             base=DQ,
// //             shape=(SQ_Z_H_N_CTX, BLOCK_DMODEL),
// //             strides=(stride_qm, stride_qk),
// //             offsets=(0, 0),
// //             block_shape=(BLOCK_M, BLOCK_DMODEL),
// //             order=(1, 0),
// //         )
// //     else:
// //         DQ_block_ptr = tl.make_block_ptr(
// //             base=DQ,
// //             shape=(Z_H_N_CTX, BLOCK_DMODEL),
// //             strides=(stride_qm, stride_qk),
// //             offsets=(0, 0),
// //             block_shape=(BLOCK_M, BLOCK_DMODEL),
// //             order=(1, 0),
// //         )

// //     DK_block_ptr = tl.make_block_ptr(
// //         base=DK,
// //         shape=(Z_H_N_CTX, BLOCK_DMODEL),
// //         strides=(stride_kn, stride_kk),
// //         offsets=(0, 0),
// //         block_shape=(BLOCK_M, BLOCK_DMODEL),
// //         order=(1, 0),
// //     )
// //     DV_block_ptr = tl.make_block_ptr(
// //         base=DV,
// //         shape=(Z_H_N_CTX, BLOCK_DMODEL),
// //         strides=(stride_vn, stride_vk),
// //         offsets=(0, 0),
// //         block_shape=(BLOCK_M, BLOCK_DMODEL),
// //         order=(1, 0),
// //     )

// //     num_block_n = tl.cdiv(N_CTX, BLOCK_N)
// //     if not SEQUENCE_PARALLEL:
// //         for start_n in range(0, num_block_n):
// //             _bwd_kernel_one_col_block(
// //                 Q,
// //                 K,
// //                 V,
// //                 sm_scale,
// //                 qk_scale,
// //                 Out,
// //                 DO,  #
// //                 DQ,
// //                 DK,
// //                 DV,  #
// //                 L,  #
// //                 D,  #
// //                 Q_block_ptr,
// //                 K_block_ptr,
// //                 V_block_ptr,  #
// //                 DO_block_ptr,
// //                 DQ_block_ptr,
// //                 DK_block_ptr,
// //                 DV_block_ptr,  #
// //                 stride_dqa,
// //                 stride_qz,
// //                 stride_qh,
// //                 stride_qm,
// //                 stride_qk,  #
// //                 stride_kz,
// //                 stride_kh,
// //                 stride_kn,
// //                 stride_kk,  #
// //                 stride_vz,
// //                 stride_vh,
// //                 stride_vn,
// //                 stride_vk,  #
// //                 Z,
// //                 H,
// //                 N_CTX,  #
// //                 off_h,
// //                 off_z,
// //                 off_hz,
// //                 start_n,
// //                 num_block_n,  #
// //                 BLOCK_M=BLOCK_M,
// //                 BLOCK_DMODEL=BLOCK_DMODEL,  #
// //                 BLOCK_N=BLOCK_N,  #
// //                 SEQUENCE_PARALLEL=SEQUENCE_PARALLEL,  #
// //                 CAUSAL=CAUSAL,  #
// //                 MMA_V3=MMA_V3,  #
// //             )
// //     else:
// //         start_n = tl.program_id(1)
// //         _bwd_kernel_one_col_block(
// //             Q,
// //             K,
// //             V,
// //             sm_scale,
// //             qk_scale,
// //             Out,
// //             DO,  #
// //             DQ,
// //             DK,
// //             DV,  #
// //             L,  #
// //             D,  #
// //             Q_block_ptr,
// //             K_block_ptr,
// //             V_block_ptr,  #
// //             DO_block_ptr,
// //             DQ_block_ptr,
// //             DK_block_ptr,
// //             DV_block_ptr,  #
// //             stride_dqa,
// //             stride_qz,
// //             stride_qh,
// //             stride_qm,
// //             stride_qk,  #
// //             stride_kz,
// //             stride_kh,
// //             stride_kn,
// //             stride_kk,  #
// //             stride_vz,
// //             stride_vh,
// //             stride_vn,
// //             stride_vk,  #
// //             Z,
// //             H,
// //             N_CTX,  #
// //             off_h,
// //             off_z,
// //             off_hz,
// //             start_n,
// //             num_block_n,  #
// //             BLOCK_M=BLOCK_M,
// //             BLOCK_DMODEL=BLOCK_DMODEL,  #
// //             BLOCK_N=BLOCK_N,  #
// //             SEQUENCE_PARALLEL=SEQUENCE_PARALLEL,  #
// //             CAUSAL=CAUSAL,  #
// //             MMA_V3=MMA_V3,  #
// //         )

// //     @staticmethod
// //     def backward(ctx, do):
// //         capability = torch.cuda.get_device_capability()
// //         MMA_V3 = capability[0] >= 9
// //         BLOCK = 128

// //         if is_hip():
// //             # Bwd pass runs out of shared memory on HIP with larger block size.
// //             BLOCK = 64

// //         q, k, v, o, L = ctx.saved_tensors
// //         sequence_parallel = ctx.sequence_parallel
// //         seq_len_kv = k.shape[2]
// //         do = do.contiguous()
// //         if sequence_parallel:
// //             replicas = cdiv(seq_len_kv, BLOCK)
// //             new_dq_shape = (replicas,) + q.shape
// //             dq = torch.zeros(new_dq_shape, device=q.device, dtype=q.dtype)
// //         else:
// //             dq = torch.zeros_like(q, dtype=q.dtype)
// //         dk = torch.empty_like(k)
// //         dv = torch.empty_like(v)
// //         delta = torch.empty_like(L)
// //         _bwd_preprocess[(cdiv(q.shape[2], BLOCK) * ctx.grid[1],)](
// //             o,
// //             do,
// //             delta,
// //             BLOCK_M=BLOCK,
// //             D_HEAD=ctx.BLOCK_DMODEL,
// //         )
// //         _bwd_kernel[(ctx.grid[1], cdiv(seq_len_kv, BLOCK) if sequence_parallel else 1)](
// //             q,
// //             k,
// //             v,
// //             ctx.sm_scale,  #
// //             o,
// //             do,  #
// //             dq,
// //             dk,
// //             dv,  #
// //             L,  #
// //             delta,  #
// //             o.numel(),
// //             q.stride(0),
// //             q.stride(1),
// //             q.stride(2),
// //             q.stride(3),  #
// //             k.stride(0),
// //             k.stride(1),
// //             k.stride(2),
// //             k.stride(3),  #
// //             v.stride(0),
// //             v.stride(1),
// //             v.stride(2),
// //             v.stride(3),  #
// //             q.shape[0],
// //             q.shape[1],
// //             q.shape[2],  #
// //             q.shape[0] * q.shape[1] * q.shape[2],  #
// //             cdiv(seq_len_kv, BLOCK) * q.shape[0] * q.shape[1] * q.shape[2],  #
// //             BLOCK_M=BLOCK,
// //             BLOCK_N=BLOCK,  #
// //             BLOCK_DMODEL=ctx.BLOCK_DMODEL,  #
// //             SEQUENCE_PARALLEL=sequence_parallel,  #
// //             CAUSAL=ctx.causal,  #
// //             MMA_V3=MMA_V3,  #
// //             num_warps=8,  #
// //             num_stages=1,  #
// //         )

// //         if len(dq.shape) == 5:
// //             dq = dq.sum(dim=0)
// //         return dq, dk, dv, None, None, None

// // attention = _attention.apply
