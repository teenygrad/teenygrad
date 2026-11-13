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

//! `MatmulForward` on shapes `test_gemm::test_matmul_forward` doesn't reach.
//!
//! That test uses m = 64, n = 48, k = 64, so every row stride is a multiple of
//! 16 bytes. `k` and `n` are the innermost extents of the A, B and C tensor
//! descriptors, and on CUDA a `k` or `n` that isn't a multiple of 4 f32 gives
//! wrong results (teenygrad#143). These tests pin down both sides of that line:
//! aligned shapes, including a ragged `m` and partial tiles, must stay exact,
//! and the ragged shapes are expected to pass once #143 is fixed.
#![cfg(feature = "hardware")]

use dotenv::dotenv;
use teeny_core::device::Device;
use teeny_core::device::buffer::Buffer;
use teeny_kernels::math::gemm::MatmulForward;

const GROUP_M: i32 = 8;
const PTX_LAUNCH_THREADS_X: u32 = 128;

// The kernel accumulates at TF32 precision. Correct results here stay below
// about 5e-4; the #143 failures are 0.38 or worse, so 1e-2 separates them.
const MAX_RELATIVE_ERROR: f32 = 1e-2;

/// `(m, n, k, block)`: the kernel uses `block` for `BLOCK_M`, `BLOCK_N` and
/// `BLOCK_K`.
type Shape = (usize, usize, usize, i32);

/// Worst error of `MatmulForward` on one shape, relative to `max(|expected|, 1)`
/// so that near-zero outputs don't dominate.
fn worst_relative_error((m, n, k, block): Shape) -> anyhow::Result<f32> {
    let device = teeny_runtime::open()?;

    // Distinct, non-repeating values so a wrong index or a dropped term shows up.
    let a: Vec<f32> = (0..m * k).map(|i| (i as f32 * 0.125) - 1.5).collect();
    let b: Vec<f32> = (0..k * n).map(|i| (i as f32 * 0.125) - 0.75).collect();

    let mut expected = vec![0.0f32; m * n];
    for mi in 0..m {
        for ni in 0..n {
            expected[mi * n + ni] = (0..k).map(|ki| a[mi * k + ki] * b[ki * n + ni]).sum();
        }
    }

    let mut a_buf = device.buffer::<f32>(m * k)?;
    let mut b_buf = device.buffer::<f32>(k * n)?;
    let c_buf = device.buffer::<f32>(m * n)?;
    a_buf.to_device(&a)?;
    b_buf.to_device(&b)?;

    let kernel = MatmulForward::<f32>::new(block, block, block, GROUP_M);
    let target = teeny_runtime::default_target(&device)?;
    let ptx = teeny_runtime::compile_kernel(&kernel, &target, true, false)?;
    let program = teeny_runtime::load_program::<MatmulForward<f32>>(&ptx)?;

    let tiles_m = (m as u32).div_ceil(block as u32);
    let tiles_n = (n as u32).div_ceil(block as u32);
    let cfg = teeny_runtime::launch_config_custom(
        [tiles_m * tiles_n, 1, 1],
        [PTX_LAUNCH_THREADS_X, 1, 1],
        [1, 1, 1],
    );
    device.launch(
        &program,
        &cfg,
        (
            a_buf.as_device_ptr(),
            b_buf.as_device_ptr(),
            c_buf.as_device_ptr(),
            m as i32,
            n as i32,
            k as i32,
        ),
    )?;

    let mut c = vec![0.0f32; m * n];
    c_buf.to_host(&mut c)?;
    Ok(c.iter()
        .zip(&expected)
        .map(|(got, want)| (got - want).abs() / want.abs().max(1.0))
        .fold(0.0, f32::max))
}

/// Runs every shape and fails once with all the bad ones, so a single run shows
/// the whole pattern rather than stopping at the first.
fn assert_exact(shapes: &[Shape]) -> anyhow::Result<()> {
    dotenv().ok();
    let mut failures = Vec::new();
    for &shape in shapes {
        let (m, n, k, block) = shape;
        // A launch error is reported but not propagated: CUDA error 716 poisons
        // the context, so every later shape would fail too and hide the pattern.
        match worst_relative_error(shape) {
            Ok(worst) if worst <= MAX_RELATIVE_ERROR => {}
            Ok(worst) => failures.push(format!(
                "[{m}, {k}] @ [{k}, {n}] block {block}: worst relative error {worst:.6}"
            )),
            Err(e) => failures.push(format!(
                "[{m}, {k}] @ [{k}, {n}] block {block}: launch failed: {e}"
            )),
        }
    }
    assert!(
        failures.is_empty(),
        "MatmulForward is wrong for:\n  {}",
        failures.join("\n  ")
    );
    Ok(())
}

#[test]
fn matmul_forward_aligned_inner_extents() -> anyhow::Result<()> {
    // n and k are multiples of 4; m is not, and n, k and m all leave partial
    // tiles, so this covers the descriptors' zero padding without #143.
    assert_exact(&[
        (33, 20, 24, 8),
        (17, 12, 16, 16),
        (5, 8, 8, 8),
        (100, 36, 40, 16),
    ])
}

#[test]
#[ignore = "wrong on CUDA when n or k is not a multiple of 4, see teenygrad#143"]
fn matmul_forward_ragged_inner_extents() -> anyhow::Result<()> {
    assert_exact(&[
        // Ragged n.
        (8, 5, 8, 8),
        (8, 7, 8, 8),
        (12, 10, 12, 16),
        (33, 30, 16, 16),
        (64, 50, 64, 32),
        // Ragged k.
        (8, 8, 5, 8),
        (8, 8, 6, 8),
        (8, 8, 9, 8),
        // Both.
        (17, 13, 21, 16),
    ])
}
