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
#[cfg(feature = "hardware")]
use teeny_core::device::Device;
#[cfg(feature = "hardware")]
use teeny_core::device::buffer::Buffer;
use teeny_core::device::program::Kernel;

#[cfg(feature = "hardware")]
const N_SPATIAL: usize = 64; // B*H*W
#[cfg(feature = "hardware")]
const C: usize = 16;
const BLOCK_N: i32 = 128;

#[cfg(feature = "hardware")]
const N_ELEM: usize = N_SPATIAL * C; // 1024

#[cfg(feature = "hardware")]
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
    let target = teeny_runtime::reference_target();
    let ptx_path = PathBuf::from(teeny_runtime::compile_kernel(
        &kernel, &target, true, false,
    )?);
    let mlir = std::fs::read_to_string(ptx_path.with_extension("mlir"))?;
    assert_debug_snapshot!(
        format!(
            "channel_bias_add_forward_source_{}",
            teeny_runtime::BACKEND_NAME
        ),
        kernel.source()
    );
    assert_debug_snapshot!(
        format!(
            "channel_bias_add_forward_mlir_{}",
            teeny_runtime::BACKEND_NAME
        ),
        mlir.trim()
    );
    Ok(())
}

#[test]
fn test_channel_bias_add_backward_snapshot() -> std::result::Result<(), Box<dyn std::error::Error>>
{
    dotenv().ok();
    let kernel =
        teeny_kernels::nn::tensor::channel_bias_add::ChannelBiasAddBackward::<f32>::new(BLOCK_N);
    let target = teeny_runtime::reference_target();
    let ptx_path = PathBuf::from(teeny_runtime::compile_kernel(
        &kernel, &target, true, false,
    )?);
    let mlir = std::fs::read_to_string(ptx_path.with_extension("mlir"))?;
    assert_debug_snapshot!(
        format!(
            "channel_bias_add_backward_source_{}",
            teeny_runtime::BACKEND_NAME
        ),
        kernel.source()
    );
    assert_debug_snapshot!(
        format!(
            "channel_bias_add_backward_mlir_{}",
            teeny_runtime::BACKEND_NAME
        ),
        mlir.trim()
    );
    Ok(())
}

#[test]
#[cfg(feature = "hardware")]
fn test_channel_bias_add_forward_cuda() -> anyhow::Result<()> {
    dotenv().ok();
    let device = teeny_runtime::open()?;

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
    let target = teeny_runtime::default_target(&device)?;
    let ptx_path = teeny_runtime::compile_kernel(&kernel, &target, true, false)?;
    let program = teeny_runtime::load_program::<
        teeny_kernels::nn::tensor::channel_bias_add::ChannelBiasAddForward<f32>,
    >(&ptx_path)?;

    let cfg =
        teeny_runtime::launch_config_custom([C as u32, 1, 1], [BLOCK_N as u32, 1, 1], [1, 1, 1]);
    device.launch(
        &program,
        &cfg,
        (
            x_buf.as_device_ptr(),
            bias_buf.as_device_ptr(),
            y_buf.as_device_ptr(),
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
#[cfg(feature = "hardware")]
fn test_channel_bias_add_backward_cuda() -> anyhow::Result<()> {
    dotenv().ok();
    let device = teeny_runtime::open()?;

    let dy_host = load("channel_bias_add/dy.bin");
    let expected_dx = load("channel_bias_add/expected_dx.bin");
    let expected_dbias = load("channel_bias_add/expected_dbias.bin");

    let mut dy_buf = device.buffer::<f32>(N_ELEM)?;
    let dx_buf = device.buffer::<f32>(N_ELEM)?;
    let dbias_buf = device.buffer::<f32>(C)?;
    dy_buf.to_device(&dy_host)?;

    let kernel =
        teeny_kernels::nn::tensor::channel_bias_add::ChannelBiasAddBackward::<f32>::new(BLOCK_N);
    let target = teeny_runtime::default_target(&device)?;
    let ptx_path = teeny_runtime::compile_kernel(&kernel, &target, true, false)?;
    let program = teeny_runtime::load_program::<
        teeny_kernels::nn::tensor::channel_bias_add::ChannelBiasAddBackward<f32>,
    >(&ptx_path)?;

    let cfg =
        teeny_runtime::launch_config_custom([C as u32, 1, 1], [BLOCK_N as u32, 1, 1], [1, 1, 1]);
    device.launch(
        &program,
        &cfg,
        (
            dy_buf.as_device_ptr(),
            dx_buf.as_device_ptr(),
            dbias_buf.as_device_ptr(),
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
