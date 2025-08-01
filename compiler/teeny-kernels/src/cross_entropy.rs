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

// fn num_warps(n: usize) -> usize {
//     if n < 2048 {
//         4
//     } else if n < 8192 {
//         8
//     } else {
//         16
//     }
// }

// @heuristics({"num_warps": lambda nargs: num_warps(nargs["N"])})
// @heuristics({"BLOCK": lambda nargs: next_power_of_2(nargs["N"])})
// @jit
// fn forward(
//     LOGITS: &DenseTensor<DynamicShape, f32>,
//     PROBS: &DenseTensor<DynamicShape, f32>,
//     IDX: &DenseTensor<DynamicShape, i64>,
//     LOSS: &DenseTensor<DynamicShape, f32>,
//     N: usize,
//     BLOCK: usize,
// ) {
//     let row = triton::program_id(0);
//     let cols = triton::arange(0, BLOCK, 1);
//     let idx = triton::load(IDX + row);

//     // pointers to logit and probs
//     LOGITS = LOGITS + row * N + cols
//     WRIT_PROBS = PROBS + row * N + cols
//     READ_PROBS = PROBS + row * N + idx

//     // write-back negative log-probs
//     let logits = triton::load(LOGITS, mask=cols < N, other=-float("inf"));
//     let logits = logits.to(tl.float32);
//     let logits = logits - triton::max(logits, 0);
//     let probs = triton::log(triton::sum(triton::exp(logits), 0)) - logits;
//     triton::store(WRIT_PROBS, probs, mask=cols < N)

//     // There is a bug in the compiler, which fails to insert a barrier here.
//     // We add it explicitly for now. Will be fixed soon.
//     triton::debug_barrier()

//     // write-back loss
//     let probs = triton::load(READ_PROBS);
//     triton::store(LOSS + row, probs);
// }

// @heuristics({"num_warps": lambda nargs: num_warps(nargs["N"])})
// @heuristics({"BLOCK": lambda nargs: next_power_of_2(nargs["N"])})
// @jit
// def _backward(PROBS, IDX, DPROBS, N, BLOCK: tl.constexpr):
//     row = tl.program_id(0)
//     cols = tl.arange(0, BLOCK)
//     idx = tl.load(IDX + row)
//     # pointers to probs
//     PROBS = PROBS + row * N + cols
//     # We know d(-log(p[i])/dlogit[k] = -id_mat[i,k] + p[k]
//     # and we have -log(p[k]) stored in PROBS, so this is easy
//     probs = -tl.load(PROBS, mask=cols < N, other=float("inf"))
//     probs = tl.exp(probs.to(tl.float32))
//     delta = cols == idx
//     # write result in-place in PROBS
//     dout = tl.load(DPROBS + row)
//     din = (probs - delta) * dout
//     tl.store(PROBS, din.to(PROBS.dtype.element_ty), mask=cols < N)

// class _cross_entropy(torch.autograd.Function):

//     @classmethod
//     def forward(cls, ctx, logits, indices):
//         # make sure we can use triton
//         assert indices.dtype == torch.int64, "Indices are expected to be of type long."
//         # make kernel
//         device, dtype = logits.device, logits.dtype
//         n_cols = logits.shape[-1]
//         # run the kernel
//         result = torch.empty_like(indices, dtype=dtype, device=device)
//         neg_logprobs = torch.empty_like(logits, dtype=dtype, device=device)
//         grid = lambda opt: (logits.numel() // n_cols,)
//         _forward[grid](logits, neg_logprobs, indices, result, n_cols)
//         # save for backward
//         ctx.save_for_backward(neg_logprobs, indices)
//         return result

//     @classmethod
//     def backward(cls, ctx, dneg_logprobs):
//         """We know d(-log(p[i])/dlogit[k] = -id_mat[i,k] + p[k]
//         so we initialize the gradient as neg_logprobs, so we can just exponentiate
//         to get p[k], which is most of what we need...  neg_logprobs will be
//         modified in place to become the gradient we want
//         """
//         # load saved tensors
//         neg_logprobs, indices = ctx.saved_tensors
//         # run the kernel
//         # neg_logprobs will be modified in place to become our gradient:
//         n_cols = neg_logprobs.shape[-1]
//         grid = lambda opt: (neg_logprobs.numel() // n_cols,)
//         _backward[grid](neg_logprobs, indices, dneg_logprobs, n_cols)
//         return neg_logprobs, None

// cross_entropy = _cross_entropy.apply
