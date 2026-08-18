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

//! 2-D tensor transpose (ONNX `Transpose` on a rank-2 tensor).
//!
//! teenygrad-3w0.10 Step 1: resolves `Op::Transpose`'s `graph/mod.rs` TODO.
//! Rank-2 only (documented non-goal: arbitrary N-D `perm`, matching every
//! prior `3w0.*` phase's narrowing pattern).
//!
//! Tile load/store go through `T::make_tensor_descriptor` +
//! `T::load_tensor_descriptor`/`store_tensor_descriptor` — the real,
//! hardware-verified way this codebase builds a genuine rank-2 tile
//! (`math::gemm`'s `matmul_forward`/`flatten_forward` precedent), not the
//! broadcast-offset idiom (`row[:, None] * stride + col[None, :]`), which
//! the `Triton` trait surface here cannot actually construct (no primitive
//! turns `I32Tensor` into a 2-D `Tensor<D>` index). `T::trans` performs the
//! actual cross-thread transpose; it's already exercised on real
//! tensor-descriptor-loaded tiles by `math::gemm::matmul_backward_da`/`_db`
//! (real `#[cfg(feature = "cuda")]` tests), so this reuses an established,
//! verified primitive rather than inventing new lowering.
//!
//! Grid: one CTA per `[BLOCK_M, BLOCK_N]` input tile (`[BLOCK_N, BLOCK_M]`
//! output tile), flat pid decoded as `(pid_m, pid_n) = (pid / num_pid_n, pid % num_pid_n)`
//! — same scheme as `flatten_forward`.
//!
//! **Non-goal (found by direct hardware experimentation, not assumed):** `M`
//! and `N` must be exact multiples of `BLOCK_M`/`BLOCK_N`. `T::trans`
//! combined with a non-block-aligned tensor-descriptor store — a
//! combination this codebase had never exercised before (`matmul_backward_da`/
//! `_db`'s `T::trans` always feeds `T::dot`, never a direct store; `matmul_forward`'s
//! non-aligned-`N` test never calls `T::trans` at all) — silently produces
//! wrong values even at fully in-bounds, non-edge-tile positions (verified
//! with M=65/N=96, M=64/N=65, M=65/N=130, all wrong; M=32/32, 64/96, 128/256
//! all exactly correct). Root-causing that interaction is out of this
//! phase's scope; callers must ensure block-aligned shapes.

#![allow(non_snake_case)]

use teeny_core::dtype::Num;
use teeny_macros::kernel;
use teeny_triton::triton::{Axis, InPtr, OutPtr, PaddingOption, Triton};

/// `y[n, m] = x[m, n]` for a rank-2 `[M, N]` input.
// ANCHOR: transpose_2d_forward
#[kernel]
pub fn transpose_2d_forward<T: Triton, D: Num, const BLOCK_M: i32, const BLOCK_N: i32>(
    x_ptr: InPtr<T::Pointer<D>>,
    y_ptr: OutPtr<T::Pointer<D>>,
    M: i32,
    N: i32,
) {
    let pid = T::program_id(Axis::X);
    let num_pid_n = T::cdiv(N, BLOCK_N);
    let pid_m = pid / num_pid_n;
    let pid_n = pid % num_pid_n;

    let x_desc = T::make_tensor_descriptor(
        x_ptr,
        &[M, N],
        &[N, 1],
        &[BLOCK_M, BLOCK_N],
        Some(PaddingOption::Zero),
    );
    let tile = T::load_tensor_descriptor(x_desc, &[pid_m * BLOCK_M, pid_n * BLOCK_N]);
    let tile_t = T::trans(tile, &[1, 0]);

    let y_desc = T::make_tensor_descriptor(
        y_ptr,
        &[N, M],
        &[M, 1],
        &[BLOCK_N, BLOCK_M],
        Some(PaddingOption::Zero),
    );
    T::store_tensor_descriptor(y_desc, &[pid_n * BLOCK_N, pid_m * BLOCK_M], tile_t);
}
// ANCHOR_END: transpose_2d_forward

// ── RuntimeOp for Transpose ────────────────────────────────────────────────
//
// `#[kernel]` only auto-derives `RuntimeOp` for the simple
// `add_offsets`/`T::load`/`T::store` idiom (e.g. `flatten_forward`); a
// tensor-descriptor-based kernel's dynamic shape/stride/offset args aren't
// staticaly analyzable the same way, so — same precedent as
// `math::gemm::MatMulRuntimeOp` — this is hand-written.

use teeny_core::device::program::ArgVisitor;
use teeny_core::model::RawPtr;
use teeny_core::model::RuntimeOp;

pub struct TransposeRuntimeOp<D: Num + Send + Sync + 'static> {
    pub kernel: Transpose2dForward<D>,
}

impl<D: Num + Send + Sync + 'static> TransposeRuntimeOp<D> {
    pub fn new(block_m: i32, block_n: i32) -> Self {
        Self {
            kernel: Transpose2dForward::<D>::new(block_m, block_n),
        }
    }

    pub fn forward_source(&self) -> &str {
        &self.kernel.source
    }

    pub fn kernel_name(&self) -> &str {
        self.kernel.name
    }
}

impl<D: Num + Send + Sync + 'static> RuntimeOp for TransposeRuntimeOp<D> {
    fn n_activation_inputs(&self) -> usize {
        1
    }

    fn param_shapes(&self, _input_shapes: &[&[usize]], _output_shape: &[usize]) -> Vec<Vec<usize>> {
        vec![]
    }

    fn pack_args(
        &self,
        inputs: &[(RawPtr, &[usize])],
        _params: &[RawPtr],
        output: RawPtr,
        _output_shape: &[usize],
        _output_row_stride: i32,
        visitor: &mut dyn ArgVisitor,
    ) {
        // x: [M, N], y (output): [N, M].
        let m = inputs[0].1.first().copied().unwrap_or(1) as i32;
        let n = inputs[0].1.get(1).copied().unwrap_or(1) as i32;
        visitor.visit_ptr(inputs[0].0); // x_ptr
        visitor.visit_ptr(output); // y_ptr
        visitor.visit_i32(m);
        visitor.visit_i32(n);
    }

    fn grid(&self, output_shape: &[usize]) -> [u32; 3] {
        // output_shape is [N, M]; the input's [M, N] is what BLOCK_M/BLOCK_N tile.
        let n = output_shape.first().copied().unwrap_or(1) as u32;
        let m = output_shape.get(1).copied().unwrap_or(1) as u32;
        let pm = m.div_ceil(self.kernel.block_m as u32);
        let pn = n.div_ceil(self.kernel.block_n as u32);
        [pm * pn, 1, 1]
    }
}
