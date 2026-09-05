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

// ChannelBiasAdd forward and backward tests.
//
// Layout: NC flat (N=B*H*W=64, C=16).
// Forward: y[n,c] = x[n,c] + bias[c].
// Backward: dx = dy; dbias[c] = sum_n dy[n,c].

use dotenv::dotenv;
use insta::assert_debug_snapshot;
use std::path::PathBuf;
use teeny_core::device::Device;
use teeny_core::device::buffer::Buffer;
use teeny_core::device::program::Kernel;
use teeny_cuda::compiler::{compile_kernel, target::Target};

#[cfg(feature = "cuda")]
use teeny_cuda::{compiler::target::Capability, device::CudaLaunchConfig, errors::Result};
#[cfg(feature = "cuda")]
use teeny_test::cuda as testing;

const N_SPATIAL: usize = 64; // B*H*W
const C: usize = 16;
const BLOCK_N: i32 = 128;

const N_ELEM: usize = N_SPATIAL * C; // 1024

fn load(rel: &str) -> Vec<f32> {
    let path = format!("{}/tests/fixtures/{}", env!("CARGO_MANIFEST_DIR"), rel);
    let bytes = std::fs::read(&path).unwrap_or_else(|e| panic!("missing {path}: {e}"));
    bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|b| f32::from_le_bytes(*b))
        .collect()
}

#[test]
fn test_channel_bias_add_forward_snapshot() -> std::result::Result<(), Box<dyn std::error::Error>> {
    dotenv().ok();
    let kernel =
        teeny_kernels::nn::tensor::channel_bias_add::ChannelBiasAddForward::<f32>::new(BLOCK_N);
    let target = Target::new(Capability::Sm89);
    let ptx_path = PathBuf::from(compile_kernel(&kernel, &target, true, false)?);
    let mlir = std::fs::read_to_string(ptx_path.with_extension("mlir"))?;
    assert_debug_snapshot!("channel_bias_add_forward_source", kernel.source());
    assert_debug_snapshot!("channel_bias_add_forward_mlir", mlir.trim());
    Ok(())
}

#[test]
fn test_channel_bias_add_backward_snapshot() -> std::result::Result<(), Box<dyn std::error::Error>>
{
    dotenv().ok();
    let kernel =
        teeny_kernels::nn::tensor::channel_bias_add::ChannelBiasAddBackward::<f32>::new(BLOCK_N);
    let target = Target::new(Capability::Sm89);
    let ptx_path = PathBuf::from(compile_kernel(&kernel, &target, true, false)?);
    let mlir = std::fs::read_to_string(ptx_path.with_extension("mlir"))?;
    assert_debug_snapshot!("channel_bias_add_backward_source", kernel.source());
    assert_debug_snapshot!("channel_bias_add_backward_mlir", mlir.trim());
    Ok(())
}

#[test]
#[cfg(feature = "cuda")]
fn test_channel_bias_add_forward_cuda() -> Result<()> {
    dotenv().ok();
    let env = testing::setup_cuda_env()?;
    let device = env.device;

    let x_host = load("channel_bias_add/x.bin");
    let bias_host = load("channel_bias_add/bias.bin");
    let expected = load("channel_bias_add/expected_y.bin");

    assert_eq!(x_host.len(), N_ELEM);
    assert_eq!(bias_host.len(), C);
    assert_eq!(expected.len(), N_ELEM);

    let mut x_buf = device.buffer::<f32>(N_ELEM)?;
    let mut bias_buf = device.buffer::<f32>(C)?;
    let y_buf = device.buffer::<f32>(N_ELEM)?;

    x_buf.to_device(&x_host)?;
    bias_buf.to_device(&bias_host)?;

    let kernel =
        teeny_kernels::nn::tensor::channel_bias_add::ChannelBiasAddForward::<f32>::new(BLOCK_N);
    let target = Target::new(env.capability);
    let ptx = std::fs::read(compile_kernel(&kernel, &target, true, false)?)?;
    let program = testing::load_program_from_ptx::<
        teeny_kernels::nn::tensor::channel_bias_add::ChannelBiasAddForward<f32>,
    >(&ptx)?;

    let cfg = CudaLaunchConfig {
        grid: [C as u32, 1, 1],
        block: [BLOCK_N as u32, 1, 1],
        cluster: [1, 1, 1],
    };
    device.launch(
        &program,
        &cfg,
        (
            x_buf.as_device_ptr() as *mut f32,
            bias_buf.as_device_ptr() as *mut f32,
            y_buf.as_device_ptr() as *mut f32,
            N_SPATIAL as i32,
            C as i32,
        ),
    )?;

    let mut y_host = vec![0.0f32; N_ELEM];
    y_buf.to_host(&mut y_host)?;

    for i in 0..N_ELEM {
        assert!(
            (y_host[i] - expected[i]).abs() < 1e-5,
            "forward mismatch at {i}: gpu={} expected={}",
            y_host[i],
            expected[i]
        );
    }
    Ok(())
}

#[test]
#[cfg(feature = "cuda")]
fn test_channel_bias_add_backward_cuda() -> Result<()> {
    dotenv().ok();
    let env = testing::setup_cuda_env()?;
    let device = env.device;

    let dy_host = load("channel_bias_add/dy.bin");
    let expected_dx = load("channel_bias_add/expected_dx.bin");
    let expected_dbias = load("channel_bias_add/expected_dbias.bin");

    let mut dy_buf = device.buffer::<f32>(N_ELEM)?;
    let dx_buf = device.buffer::<f32>(N_ELEM)?;
    let dbias_buf = device.buffer::<f32>(C)?;
    dy_buf.to_device(&dy_host)?;

    let kernel =
        teeny_kernels::nn::tensor::channel_bias_add::ChannelBiasAddBackward::<f32>::new(BLOCK_N);
    let target = Target::new(env.capability);
    let ptx = std::fs::read(compile_kernel(&kernel, &target, true, false)?)?;
    let program = testing::load_program_from_ptx::<
        teeny_kernels::nn::tensor::channel_bias_add::ChannelBiasAddBackward<f32>,
    >(&ptx)?;

    let cfg = CudaLaunchConfig {
        grid: [C as u32, 1, 1],
        block: [BLOCK_N as u32, 1, 1],
        cluster: [1, 1, 1],
    };
    device.launch(
        &program,
        &cfg,
        (
            dy_buf.as_device_ptr() as *mut f32,
            dx_buf.as_device_ptr() as *mut f32,
            dbias_buf.as_device_ptr() as *mut f32,
            N_SPATIAL as i32,
            C as i32,
        ),
    )?;

    let mut dx_host = vec![0.0f32; N_ELEM];
    let mut dbias_host = vec![0.0f32; C];
    dx_buf.to_host(&mut dx_host)?;
    dbias_buf.to_host(&mut dbias_host)?;

    for i in 0..N_ELEM {
        assert!(
            (dx_host[i] - expected_dx[i]).abs() < 1e-5,
            "dx mismatch at {i}: gpu={} expected={}",
            dx_host[i],
            expected_dx[i]
        );
    }
    for c in 0..C {
        assert!(
            (dbias_host[c] - expected_dbias[c]).abs() < 1e-3,
            "dbias mismatch at {c}: gpu={} expected={}",
            dbias_host[c],
            expected_dbias[c]
        );
    }
    Ok(())
}
