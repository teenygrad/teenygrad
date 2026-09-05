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

//! teenygrad-3w0.10 Step 1: `transpose_2d_forward` compiles and produces
//! numerically correct output on real CUDA hardware.

use std::path::PathBuf;

use dotenv::dotenv;
use insta::assert_debug_snapshot;
use teeny_core::device::program::Kernel;
use teeny_cuda::compiler::{compile_kernel, target::Target};

#[cfg(feature = "cuda")]
use teeny_core::device::Device;
#[cfg(feature = "cuda")]
use teeny_core::device::buffer::Buffer;
#[cfg(feature = "cuda")]
use teeny_cuda::{errors::Result, testing};

use teeny_kernels::nn::tensor::transpose::Transpose2dForward;

const BLOCK_M: i32 = 32;
const BLOCK_N: i32 = 32;

#[test]
fn test_transpose_2d_forward_source() -> anyhow::Result<()> {
    dotenv().ok();
    let kernel = Transpose2dForward::<f32>::new(BLOCK_M, BLOCK_N);
    let target = Target::new(teeny_cuda::compiler::target::Capability::Sm89);
    let ptx_path = PathBuf::from(compile_kernel(&kernel, &target, true, false)?);
    let mlir = std::fs::read_to_string(ptx_path.with_extension("mlir"))?;
    assert_debug_snapshot!("transpose_2d_forward_source", kernel.source());
    assert_debug_snapshot!("transpose_2d_forward_mlir", mlir.trim());
    Ok(())
}

#[cfg(feature = "cuda")]
#[test]
fn test_transpose_2d_forward_gpu() -> Result<()> {
    dotenv().ok();
    let env = testing::setup_cuda_env()?;
    let target = Target::new(env.capability);

    // Must be exact multiples of BLOCK_M/BLOCK_N -- see transpose.rs's module
    // doc for why (a real, narrow gap in T::trans + non-block-aligned
    // tensor-descriptor stores, found by direct experimentation: M=65/N=96,
    // M=64/N=65, and M=65/N=130 all silently produced wrong values at
    // in-bounds, non-edge-tile positions, while every block-aligned size
    // tried -- 32x32, 64x96, 128x256 -- was exactly correct).
    const M: usize = 128;
    const N: usize = 256;

    let kernel = Transpose2dForward::<f32>::new(BLOCK_M, BLOCK_N);
    let ptx = std::fs::read(compile_kernel(&kernel, &target, true, false)?)?;
    let program = testing::load_program_from_ptx::<Transpose2dForward<f32>>(&ptx)?;

    let mut x_data = vec![0.0f32; M * N];
    for (i, v) in x_data.iter_mut().enumerate() {
        *v = i as f32;
    }
    let mut x_buf = env.device.buffer::<f32>(M * N)?;
    let y_buf = env.device.buffer::<f32>(M * N)?;
    x_buf.to_device(&x_data)?;

    let pm = (M as u32).div_ceil(BLOCK_M as u32);
    let pn = (N as u32).div_ceil(BLOCK_N as u32);
    let cfg = teeny_cuda::device::CudaLaunchConfig {
        grid: [pm * pn, 1, 1],
        block: [program.threads_per_block(), 1, 1],
        cluster: [program.num_ctas().max(1), 1, 1],
    };

    env.device.launch(
        &program,
        &cfg,
        (
            x_buf.as_device_ptr() as *mut f32,
            y_buf.as_device_ptr() as *mut f32,
            M as i32,
            N as i32,
        ),
    )?;

    let mut y_out = vec![0.0f32; M * N];
    y_buf.to_host(&mut y_out)?;

    for m in 0..M {
        for n in 0..N {
            let expected = x_data[m * N + n];
            let got = y_out[n * M + m];
            assert!(
                (expected - got).abs() < 1e-5,
                "mismatch at (m={m}, n={n}): expected {expected}, got {got}"
            );
        }
    }
    Ok(())
}
