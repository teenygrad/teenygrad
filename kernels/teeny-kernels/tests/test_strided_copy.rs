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

//! `strided_copy_forward`, the one kernel behind Transpose, Slice and Expand, against
//! torch's result for the same operation (fixtures from
//! `tests/fixtures/strided_copy/generate.py`).

#![cfg(feature = "hardware")]

use dotenv::dotenv;
use teeny_core::device::Device;
use teeny_core::device::buffer::Buffer;
use teeny_core::model::RuntimeOp;
use teeny_kernels::nn::tensor::strided_copy::{StridedCopyForward, StridedCopyRuntimeOp};
use teeny_test::load_fixture;

const BLOCK: i32 = 128;

/// Runs the kernel for `op` on fixture `name` and checks it against torch's output.
fn check(name: &str, out_shape: &[usize], op: StridedCopyRuntimeOp<f32>) -> anyhow::Result<()> {
    dotenv().ok();
    let dir = env!("CARGO_MANIFEST_DIR");
    let input = load_fixture(dir, &format!("strided_copy/x_{name}.bin"));
    let want = load_fixture(dir, &format!("strided_copy/expected_{name}.bin"));
    let n: usize = out_shape.iter().product();
    assert_eq!(want.len(), n, "{name}: fixture and out_shape disagree");

    let device = teeny_runtime::open()?;
    let kernel = StridedCopyForward::<f32>::new(BLOCK);
    let target = teeny_runtime::default_target(&device)?;
    let path = teeny_runtime::compile_kernel(&kernel, &target, true, false)?;
    let program = teeny_runtime::load_program::<StridedCopyForward<f32>>(&path)?;

    let mut x_buf = device.buffer::<f32>(input.len())?;
    x_buf.to_device(&input)?;
    let mut y_buf = device.buffer::<f32>(n)?;
    y_buf.to_device(&vec![f32::NAN; n])?;

    // Scalars come from the RuntimeOp, so the test exercises the same mapping the
    // lowering uses rather than a hand-written duplicate.
    let a = op.kernel_scalars(n as i32);
    let grid = op.grid(out_shape);
    let cfg = teeny_runtime::launch_config_with_grid(grid[0] as usize, &program);
    device.launch(
        &program,
        &cfg,
        (
            x_buf.as_device_ptr(),
            y_buf.as_device_ptr(),
            a[0],
            a[1],
            a[2],
            a[3],
            a[4],
            a[5],
            a[6],
            a[7],
            a[8],
            a[9],
            a[10],
            a[11],
            a[12],
            a[13],
        ),
    )?;

    let mut got = vec![0.0f32; n];
    y_buf.to_host(&mut got)?;
    // A copy moves bits, so it must be exact.
    if let Some(i) = got
        .iter()
        .zip(&want)
        .position(|(g, w)| g.to_bits() != w.to_bits())
    {
        anyhow::bail!(
            "{name}: first mismatch at {i}: got {}, want {}",
            got[i],
            want[i]
        );
    }
    Ok(())
}

#[test]
fn test_strided_copy_transpose() -> anyhow::Result<()> {
    check(
        "transpose",
        &[2, 4, 5, 3],
        StridedCopyRuntimeOp::<f32>::transpose(BLOCK, &[2, 3, 4, 5], &[0, 2, 3, 1]),
    )?;
    check(
        "transpose_multi_block",
        &[33, 17, 9],
        StridedCopyRuntimeOp::<f32>::transpose(BLOCK, &[9, 33, 17], &[1, 2, 0]),
    )
}

#[test]
fn test_strided_copy_slice() -> anyhow::Result<()> {
    // x[:, 1:3, 0:4:2, 1:5:3]
    check(
        "slice",
        &[2, 2, 2, 2],
        StridedCopyRuntimeOp::<f32>::slice(
            BLOCK,
            &[2, 3, 4, 5],
            &[2, 2, 2, 2],
            &[0, 1, 0, 1],
            &[1, 1, 2, 3],
        ),
    )?;
    // x[:, ::-1, :, 0:5:2]: axis 1 starts at its last element and steps back.
    check(
        "slice_reversed",
        &[2, 3, 4, 3],
        StridedCopyRuntimeOp::<f32>::slice(
            BLOCK,
            &[2, 3, 4, 5],
            &[2, 3, 4, 3],
            &[0, 2, 0, 0],
            &[1, -1, 1, 2],
        ),
    )
}

#[test]
fn test_strided_copy_expand() -> anyhow::Result<()> {
    check(
        "expand",
        &[2, 3, 4, 5],
        StridedCopyRuntimeOp::<f32>::expand(BLOCK, &[1, 3, 1, 5], &[2, 3, 4, 5]),
    )?;
    check(
        "expand_rank_up",
        &[2, 3, 4, 5],
        StridedCopyRuntimeOp::<f32>::expand(BLOCK, &[3, 1, 5], &[2, 3, 4, 5]),
    )
}
