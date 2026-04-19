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

use teeny_core::dtype::Float;
use teeny_macros::kernel;
use teeny_triton::triton::{Axis, PaddingOption, Triton};

/// Copy a [B, N] tensor with arbitrary input strides to a contiguous row-major [B, N] output.
///
/// This is the forward pass of the flatten operation. In a neural network pipeline it is
/// used to materialise a contiguous copy of a potentially non-contiguous activation tensor
/// (e.g. when transitioning from convolutional to fully-connected layers). When the input is
/// already row-major contiguous (stride_ib = N, stride_in = 1) this is a simple memcpy; when
/// the input uses a different layout (e.g. column-major: stride_ib = 1, stride_in = B) the
/// kernel performs the necessary reordering so that downstream kernels can assume unit strides.
///
/// Grid: one flat 1D pid that encodes (pid_b, pid_n) = (pid / num_pid_n, pid % num_pid_n).
#[kernel]
pub fn flatten_forward<
    T: Triton,
    D: Float,
    const BLOCK_B: i32,
    const BLOCK_N: i32,
>(
    input_ptr: T::Pointer<D>,
    output_ptr: T::Pointer<D>,
    B: i32,
    N: i32,
    stride_ib: i32,
    stride_in: i32,
) {
    let pid = T::program_id(Axis::X);
    let num_pid_n = T::cdiv(N, BLOCK_N);
    let pid_b = pid / num_pid_n;
    let pid_n = pid % num_pid_n;

    let input_desc = T::make_tensor_descriptor(
        input_ptr,
        &[B, N],
        &[stride_ib, stride_in],
        &[BLOCK_B, BLOCK_N],
        Some(PaddingOption::Zero),
    );
    let output_desc = T::make_tensor_descriptor(
        output_ptr,
        &[B, N],
        &[N, 1],
        &[BLOCK_B, BLOCK_N],
        Some(PaddingOption::Zero),
    );

    let b_off = pid_b * BLOCK_B;
    let n_off = pid_n * BLOCK_N;
    let tile = T::load_tensor_descriptor(input_desc, &[b_off, n_off]);
    T::store_tensor_descriptor(output_desc, &[b_off, n_off], tile);
}

/// Copy a contiguous row-major [B, N] gradient back to an output buffer with arbitrary strides.
///
/// This is the backward pass of the flatten operation. The gradient dy arrives as a contiguous
/// row-major tensor (the upstream gradient matches the contiguous forward output); this kernel
/// writes it back to dx using the original input strides so that the gradient is in the same
/// memory layout as the forward input. When the forward input was already row-major contiguous
/// this is again a simple memcpy; when the forward input used a different layout the kernel
/// performs the inverse reordering.
#[kernel]
pub fn flatten_backward<
    T: Triton,
    D: Float,
    const BLOCK_B: i32,
    const BLOCK_N: i32,
>(
    dy_ptr: T::Pointer<D>,
    dx_ptr: T::Pointer<D>,
    B: i32,
    N: i32,
    stride_dxb: i32,
    stride_dxn: i32,
) {
    let pid = T::program_id(Axis::X);
    let num_pid_n = T::cdiv(N, BLOCK_N);
    let pid_b = pid / num_pid_n;
    let pid_n = pid % num_pid_n;

    let dy_desc = T::make_tensor_descriptor(
        dy_ptr,
        &[B, N],
        &[N, 1],
        &[BLOCK_B, BLOCK_N],
        Some(PaddingOption::Zero),
    );
    let dx_desc = T::make_tensor_descriptor(
        dx_ptr,
        &[B, N],
        &[stride_dxb, stride_dxn],
        &[BLOCK_B, BLOCK_N],
        Some(PaddingOption::Zero),
    );

    let b_off = pid_b * BLOCK_B;
    let n_off = pid_n * BLOCK_N;
    let tile = T::load_tensor_descriptor(dy_desc, &[b_off, n_off]);
    T::store_tensor_descriptor(dx_desc, &[b_off, n_off], tile);
}
