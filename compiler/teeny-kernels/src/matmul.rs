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

// import torch

// from triton import Config, autotune, cdiv, heuristics, jit
// from triton import language as tl
// from .matmul_perf_model import early_config_prune, estimate_matmul_time

// _ordered_datatypes = [torch.int8, torch.float16, torch.bfloat16, torch.float32]

// def upcast_if_fp8(a):
//     if "fp8" in str(a):
//         return torch.float16
//     return a

// def get_higher_dtype(a, b):
//     a = upcast_if_fp8(a)
//     b = upcast_if_fp8(b)
//     if a is b:
//         return a

//     assert a in _ordered_datatypes
//     assert b in _ordered_datatypes

//     for d in _ordered_datatypes:
//         if a is d:
//             return b
//         if b is d:
//             return a

// def init_to_zero(name):
//     return lambda nargs: nargs[name].zero_()

// def get_configs_io_bound():
//     configs = []
//     for num_stages in [2, 3, 4, 5, 6]:
//         for block_m in [16, 32]:
//             for block_k in [32, 64]:
//                 for block_n in [32, 64, 128, 256]:
//                     num_warps = 2 if block_n <= 64 else 4
//                     configs.append(
//                         Config(
//                             {
//                                 "BLOCK_M": block_m,
//                                 "BLOCK_N": block_n,
//                                 "BLOCK_K": block_k,
//                                 "SPLIT_K": 1,
//                             },
//                             num_stages=num_stages,
//                             num_warps=num_warps,
//                         )
//                     )
//                     # split_k
//                     for split_k in [2, 4, 8, 16]:
//                         configs.append(
//                             Config(
//                                 {
//                                     "BLOCK_M": block_m,
//                                     "BLOCK_N": block_n,
//                                     "BLOCK_K": block_k,
//                                     "SPLIT_K": split_k,
//                                 },
//                                 num_stages=num_stages,
//                                 num_warps=num_warps,
//                                 pre_hook=init_to_zero("C"),
//                             )
//                         )
//     return configs

// @autotune(
//     configs=[
//         # basic configs for compute-bound matmuls
//         Config(
//             {"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 32, "SPLIT_K": 1},
//             num_stages=3,
//             num_warps=8,
//         ),
//         Config(
//             {"BLOCK_M": 256, "BLOCK_N": 128, "BLOCK_K": 32, "SPLIT_K": 1},
//             num_stages=3,
//             num_warps=8,
//         ),
//         Config(
//             {"BLOCK_M": 256, "BLOCK_N": 64, "BLOCK_K": 32, "SPLIT_K": 1},
//             num_stages=4,
//             num_warps=4,
//         ),
//         Config(
//             {"BLOCK_M": 64, "BLOCK_N": 256, "BLOCK_K": 32, "SPLIT_K": 1},
//             num_stages=4,
//             num_warps=4,
//         ),
//         Config(
//             {"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32, "SPLIT_K": 1},
//             num_stages=4,
//             num_warps=4,
//         ),
//         Config(
//             {"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 32, "SPLIT_K": 1},
//             num_stages=4,
//             num_warps=4,
//         ),
//         Config(
//             {"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 32, "SPLIT_K": 1},
//             num_stages=4,
//             num_warps=4,
//         ),
//         Config(
//             {"BLOCK_M": 128, "BLOCK_N": 32, "BLOCK_K": 32, "SPLIT_K": 1},
//             num_stages=4,
//             num_warps=4,
//         ),
//         Config(
//             {"BLOCK_M": 64, "BLOCK_N": 32, "BLOCK_K": 32, "SPLIT_K": 1},
//             num_stages=5,
//             num_warps=2,
//         ),
//         # good for int8
//         Config(
//             {"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 128, "SPLIT_K": 1},
//             num_stages=3,
//             num_warps=8,
//         ),
//         Config(
//             {"BLOCK_M": 256, "BLOCK_N": 128, "BLOCK_K": 128, "SPLIT_K": 1},
//             num_stages=3,
//             num_warps=8,
//         ),
//         Config(
//             {"BLOCK_M": 256, "BLOCK_N": 64, "BLOCK_K": 128, "SPLIT_K": 1},
//             num_stages=4,
//             num_warps=4,
//         ),
//         Config(
//             {"BLOCK_M": 64, "BLOCK_N": 256, "BLOCK_K": 128, "SPLIT_K": 1},
//             num_stages=4,
//             num_warps=4,
//         ),
//         Config(
//             {"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 128, "SPLIT_K": 1},
//             num_stages=4,
//             num_warps=4,
//         ),
//         Config(
//             {"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 64, "SPLIT_K": 1},
//             num_stages=4,
//             num_warps=4,
//         ),
//         Config(
//             {"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 64, "SPLIT_K": 1},
//             num_stages=4,
//             num_warps=4,
//         ),
//         Config(
//             {"BLOCK_M": 128, "BLOCK_N": 32, "BLOCK_K": 64, "SPLIT_K": 1},
//             num_stages=4,
//             num_warps=4,
//         ),
//         Config(
//             {"BLOCK_M": 64, "BLOCK_N": 32, "BLOCK_K": 64, "SPLIT_K": 1},
//             num_stages=5,
//             num_warps=2,
//         ),
//     ]
//     + get_configs_io_bound(),
//     key=["M", "N", "K"],
//     prune_configs_by={
//         "early_config_prune": early_config_prune,
//         "perf_model": estimate_matmul_time,
//         "top_k": 10,
//     },
// )
// @heuristics(
//     {
//         "EVEN_K": lambda args: args["K"] % (args["BLOCK_K"] * args["SPLIT_K"]) == 0,
//     }
// )
// @jit

// fn kernel(
//     A: &DenseTensor<DynamicShape, f32>,
//     B: &DenseTensor<DynamicShape, f32>,
//     C: &DenseTensor<DynamicShape, f32>,
//     M: usize,
//     N: usize,
//     K: usize,
//     stride_am: usize,
//     stride_ak: usize,
//     stride_bk: usize,
//     stride_bn: usize,
//     stride_cm: usize,
//     stride_cn: usize,
//     acc_dtype: usize,
//     input_precision: usize,
//     fp8_fast_accum: usize,
//     BLOCK_M: usize,
//     BLOCK_N: usize,
//     BLOCK_K: usize,
//     GROUP_M: usize,
//     SPLIT_K: usize,
//     EVEN_K: usize,
//     AB_DTYPE: usize,
// ) {
//     // matrix multiplication
//     let pid = triton::program_id(0);
//     let pid_z = triton::program_id(1);
//     let grid_m = triton::cdiv(M, BLOCK_M);
//     let grid_n = triton::cdiv(N, BLOCK_N);
//     // re-order program ID for better L2 performance
//     let width = GROUP_M * grid_n;
//     let group_id = pid / width;
//     let group_size = min(grid_m - group_id * GROUP_M, GROUP_M);
//     let pid_m = group_id * GROUP_M + (pid % group_size);
//     let pid_n = (pid % width) / group_size;
//     // do matrix multiplication
//     let rm = pid_m * BLOCK_M + triton::arange(0, BLOCK_M, 1);
//     let rn = pid_n * BLOCK_N + triton::arange(0, BLOCK_N, 1);
//     let ram = triton::max_contiguous(triton::multiple_of(rm % M, BLOCK_M), BLOCK_M);
//     let rbn = triton::max_contiguous(triton::multiple_of(rn % N, BLOCK_N), BLOCK_N);
//     let rk = pid_z * BLOCK_K + triton::arange(0, BLOCK_K, 1);
//     // pointers
//     let A = A + (ram[:, None] * stride_am + rk[None, :] * stride_ak);
//     let B = B + (rk[:, None] * stride_bk + rbn[None, :] * stride_bn);
//     let acc = triton::zeros((BLOCK_M, BLOCK_N), dtype=acc_dtype);
//     for k in range(0, triton::cdiv(K, BLOCK_K * SPLIT_K)) {
//       if EVEN_K {
//           let a = triton::load(A);
//           let b = triton::load(B);
//       } else {
//           let k_remaining = K - k * (BLOCK_K * SPLIT_K);
//           let _0 = triton::zeros((1, 1), dtype=C.dtype.element_ty);
//           let a = triton::load(A, mask=rk[None, :] < k_remaining, other=_0);
//           let b = triton::load(B, mask=rk[:, None] < k_remaining, other=_0);
//       }
//       if AB_DTYPE is not None {
//           a = a.to(AB_DTYPE);
//           b = b.to(AB_DTYPE);
//       }
//       if fp8_fast_accum {
//           acc = triton::dot(
//               a, b, acc, out_dtype=acc_dtype, input_precision=input_precision
//           )
//       } else {
//           acc += triton::dot(a, b, out_dtype=acc_dtype, input_precision=input_precision)
//       }
//       let A += BLOCK_K * SPLIT_K * stride_ak;
//       let B += BLOCK_K * SPLIT_K * stride_bk;
//     }

//     let acc = acc.to(C.dtype.element_ty)

//     // rematerialize rm and rn to save registers
//     let rm = pid_m * BLOCK_M + triton::arange(0, BLOCK_M, 1);
//     let rn = pid_n * BLOCK_N + triton::arange(0, BLOCK_N, 1);
//     let C = C + (rm[:, None] * stride_cm + rn[None, :] * stride_cn);
//     let mask = (rm < M)[:, None] & (rn < N)[None, :];

//     // handles write-back with reduction-splitting
//     if SPLIT_K == 1 {
//         triton::store(C, acc, mask=mask)
//     } else {
//         triton::atomic_add(C, acc, mask=mask)
//     }
// }

// fn matmul(a: &DenseTensor<DynamicShape, f32>, b: &DenseTensor<DynamicShape, f32>, acc_dtype: usize,
//   input_precision: usize, fp8_fast_accum: usize, output_dtype: usize) {
//     let device = a.device
//     // handle non-contiguous inputs if necessary
//     if a.stride(0) > 1 and a.stride(1) > 1 {
//         a = a.contiguous()
//     }
//     if b.stride(0) > 1 and b.stride(1) > 1 {
//         b = b.contiguous()
//     }
//     // checks constraints
//     assert a.shape[1] == b.shape[0], "incompatible dimensions {a.shape} and {b.shape}";
//     let M = a.shape[0];
//     let K = a.shape[1];
//     let N = b.shape[1];

//     // common type between a and b
//     let ab_dtype = get_higher_dtype(a.dtype, b.dtype)

//     // allocates output
//     if output_dtype is None {
//         output_dtype = ab_dtype
//     }

//     let c = torch.empty((M, N), device=device, dtype=output_dtype)

//     // Allowed types for acc_type given the types of a and b.
//     let supported_acc_dtypes = {
//         torch.float16: (torch.float32, torch.float16),
//         torch.bfloat16: (torch.float32, torch.bfloat16),
//         torch.float32: (torch.float32,),
//         torch.int8: (torch.int32,),
//     }

//     if acc_dtype is None {
//         acc_dtype = supported_acc_dtypes[ab_dtype][0]
//     } else {
//         assert isinstance(acc_dtype, torch.dtype), "acc_dtype must be a torch.dtype"
//         assert (
//             acc_dtype in supported_acc_dtypes[a.dtype]
//         ), "acc_dtype not compatible with the type of a"
//         assert (
//             acc_dtype in supported_acc_dtypes[b.dtype]
//         ), "acc_dtype not compatible with the type of b"

//     def to_tl_type(ty):
//         return getattr(tl, str(ty).split(".")[-1])

//     acc_dtype = to_tl_type(acc_dtype)
//     ab_dtype = to_tl_type(ab_dtype)
//     output_dtype = to_tl_type(output_dtype)

//     // Tensor cores support input with mixed float8 types.
//     if a.dtype in [tl.float8e4nv, tl.float8e5] and b.dtype in [
//         tl.float8e4nv,
//         tl.float8e5,
//     ] {
//         ab_dtype = None
//     }
//     // launch kernel
//     grid = lambda META: (
//         triton::cdiv(M, META["BLOCK_M"]) * triton::cdiv(N, META["BLOCK_N"]),
//         META["SPLIT_K"],
//     )

//     kernel(
//         a,
//         b,
//         c,
//         M,
//         N,
//         K,
//         a.stride(0),
//         a.stride(1),
//         b.stride(0),
//         b.stride(1),
//         c.stride(0),
//         c.stride(1),
//         acc_dtype,
//         input_precision,
//         fp8_fast_accum,
//         GROUP_M=8,
//         AB_DTYPE=ab_dtype,
//     )
//     return c
//   }
