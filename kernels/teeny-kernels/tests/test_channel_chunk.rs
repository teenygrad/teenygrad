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

// ChannelChunk forward and backward tests.
//
// Kernel: channel_chunk_forward / channel_chunk_backward
// Layout: flat NC (channels-last), index = n_spatial * C + c
//
// Dimensions (matching the first C3k2 layer in YOLO11n with n=2):
//   N_SPATIAL = 64   (B=2, H=4, W=8)
//   C_TOTAL   = 32
//   CHUNK_C   = 16   (split into 2 equal halves)
//
// Grid: N_SPATIAL * cdiv(CHUNK_C, BLOCK_SIZE) CTAs.
// With CHUNK_C=16 < BLOCK_SIZE=128, num_c_tiles=1, so grid = N_SPATIAL CTAs.
// Each CTA processes one spatial position; 16 of 128 lanes are active.

use std::path::PathBuf;

use dotenv::dotenv;
use insta::assert_debug_snapshot;
#[cfg(feature = "hardware")]
use teeny_core::device::Device;
#[cfg(feature = "hardware")]
use teeny_core::device::buffer::Buffer;
use teeny_core::device::program::Kernel;

#[cfg(feature = "hardware")]
use teeny_test::load_fixture;

// ── Dimensions ───────────────────────────────────────────────────────────────
#[cfg(feature = "hardware")]
const N_SPATIAL: usize = 64;
#[cfg(feature = "hardware")]
const C_TOTAL: usize = 32;
#[cfg(feature = "hardware")]
const CHUNK_C: usize = 16;
const BLOCK_SIZE: i32 = 128;

#[cfg(feature = "hardware")]
const N_ELEM_IN: usize = N_SPATIAL * C_TOTAL; // 2048
#[cfg(feature = "hardware")]
const N_ELEM_CHUNK: usize = N_SPATIAL * CHUNK_C; // 1024

// ── Fixture loader ────────────────────────────────────────────────────────────

// ── MLIR snapshot tests ───────────────────────────────────────────────────────

#[test]
fn test_channel_chunk_forward_snapshot() -> anyhow::Result<()> {
    dotenv().ok();

    let kernel =
        teeny_kernels::nn::tensor::channel_chunk::ChannelChunkForward::<f32>::new(BLOCK_SIZE);
    let target = teeny_runtime::reference_target();
    let ptx_path = PathBuf::from(teeny_runtime::compile_kernel(
        &kernel, &target, true, false,
    )?);
    let mlir = std::fs::read_to_string(ptx_path.with_extension("mlir"))?;

    assert_debug_snapshot!(
        format!(
            "channel_chunk_forward_source_{}",
            teeny_runtime::BACKEND_NAME
        ),
        kernel.source()
    );
    assert_debug_snapshot!(
        format!("channel_chunk_forward_mlir_{}", teeny_runtime::BACKEND_NAME),
        mlir.trim()
    );

    Ok(())
}

#[test]
fn test_channel_chunk_backward_snapshot() -> anyhow::Result<()> {
    dotenv().ok();

    let kernel =
        teeny_kernels::nn::tensor::channel_chunk::ChannelChunkBackward::<f32>::new(BLOCK_SIZE);
    let target = teeny_runtime::reference_target();
    let ptx_path = PathBuf::from(teeny_runtime::compile_kernel(
        &kernel, &target, true, false,
    )?);
    let mlir = std::fs::read_to_string(ptx_path.with_extension("mlir"))?;

    assert_debug_snapshot!(
        format!(
            "channel_chunk_backward_source_{}",
            teeny_runtime::BACKEND_NAME
        ),
        kernel.source()
    );
    assert_debug_snapshot!(
        format!(
            "channel_chunk_backward_mlir_{}",
            teeny_runtime::BACKEND_NAME
        ),
        mlir.trim()
    );

    Ok(())
}

// ── CUDA forward test ─────────────────────────────────────────────────────────

#[test]
#[cfg(feature = "hardware")]
fn test_channel_chunk_forward_cuda() -> anyhow::Result<()> {
    dotenv().ok();
    let device = teeny_runtime::open()?;

    let x_host = load_fixture(env!("CARGO_MANIFEST_DIR"), "channel_chunk/x.bin");
    let expected_chunk0 = load_fixture(
        env!("CARGO_MANIFEST_DIR"),
        "channel_chunk/expected_chunk0.bin",
    );
    let expected_chunk1 = load_fixture(
        env!("CARGO_MANIFEST_DIR"),
        "channel_chunk/expected_chunk1.bin",
    );

    assert_eq!(x_host.len(), N_ELEM_IN);
    assert_eq!(expected_chunk0.len(), N_ELEM_CHUNK);

    let mut x_buf = device.buffer::<f32>(N_ELEM_IN)?;
    let y0_buf = device.buffer::<f32>(N_ELEM_CHUNK)?;
    let y1_buf = device.buffer::<f32>(N_ELEM_CHUNK)?;
    x_buf.to_device(&x_host)?;

    let kernel =
        teeny_kernels::nn::tensor::channel_chunk::ChannelChunkForward::<f32>::new(BLOCK_SIZE);
    let target = teeny_runtime::default_target(&device)?;
    let ptx_path = teeny_runtime::compile_kernel(&kernel, &target, true, false)?;
    let program = teeny_runtime::load_program::<
        teeny_kernels::nn::tensor::channel_chunk::ChannelChunkForward<f32>,
    >(&ptx_path)?;

    // Grid: N_SPATIAL * cdiv(CHUNK_C, BLOCK_SIZE) CTAs.
    let num_c_tiles = CHUNK_C.div_ceil(BLOCK_SIZE as usize);
    let cfg = teeny_runtime::launch_config_custom(
        [(N_SPATIAL * num_c_tiles) as u32, 1, 1],
        [BLOCK_SIZE as u32, 1, 1],
        [1, 1, 1],
    );

    // Chunk 0: channels [0, CHUNK_C)
    device.launch(
        &program,
        &cfg,
        (
            x_buf.as_device_ptr(),
            y0_buf.as_device_ptr(),
            C_TOTAL as i32,
            CHUNK_C as i32,
            0i32, // chunk_offset = 0
        ),
    )?;

    // Chunk 1: channels [CHUNK_C, C_TOTAL)
    device.launch(
        &program,
        &cfg,
        (
            x_buf.as_device_ptr(),
            y1_buf.as_device_ptr(),
            C_TOTAL as i32,
            CHUNK_C as i32,
            CHUNK_C as i32, // chunk_offset = 16
        ),
    )?;

    let mut y0_host = vec![0.0f32; N_ELEM_CHUNK];
    let mut y1_host = vec![0.0f32; N_ELEM_CHUNK];
    y0_buf.to_host(&mut y0_host)?;
    y1_buf.to_host(&mut y1_host)?;

    for i in 0..N_ELEM_CHUNK {
        assert_eq!(
            y0_host[i], expected_chunk0[i],
            "chunk0 mismatch at {i}: gpu={} expected={}",
            y0_host[i], expected_chunk0[i],
        );
        assert_eq!(
            y1_host[i], expected_chunk1[i],
            "chunk1 mismatch at {i}: gpu={} expected={}",
            y1_host[i], expected_chunk1[i],
        );
    }

    Ok(())
}

// ── CUDA backward test ────────────────────────────────────────────────────────

#[test]
#[cfg(feature = "hardware")]
fn test_channel_chunk_backward_cuda() -> anyhow::Result<()> {
    dotenv().ok();
    let device = teeny_runtime::open()?;

    let dy_host = load_fixture(env!("CARGO_MANIFEST_DIR"), "channel_chunk/dy.bin");
    let expected = load_fixture(env!("CARGO_MANIFEST_DIR"), "channel_chunk/expected_dx.bin");

    assert_eq!(dy_host.len(), N_ELEM_CHUNK);
    assert_eq!(expected.len(), N_ELEM_IN);

    let mut dy_buf = device.buffer::<f32>(N_ELEM_CHUNK)?;
    // dx must be zero-initialised; the backward only writes to its chunk range.
    let dx_buf = device.buffer::<f32>(N_ELEM_IN)?;
    dy_buf.to_device(&dy_host)?;

    let kernel =
        teeny_kernels::nn::tensor::channel_chunk::ChannelChunkBackward::<f32>::new(BLOCK_SIZE);
    let target = teeny_runtime::default_target(&device)?;
    let ptx_path = teeny_runtime::compile_kernel(&kernel, &target, true, false)?;
    let program = teeny_runtime::load_program::<
        teeny_kernels::nn::tensor::channel_chunk::ChannelChunkBackward<f32>,
    >(&ptx_path)?;

    let num_c_tiles = CHUNK_C.div_ceil(BLOCK_SIZE as usize);
    let cfg = teeny_runtime::launch_config_custom(
        [(N_SPATIAL * num_c_tiles) as u32, 1, 1],
        [BLOCK_SIZE as u32, 1, 1],
        [1, 1, 1],
    );

    // Backward for chunk 0 (chunk_offset = 0).
    device.launch(
        &program,
        &cfg,
        (
            dy_buf.as_device_ptr(),
            dx_buf.as_device_ptr(),
            C_TOTAL as i32,
            CHUNK_C as i32,
            0i32,
        ),
    )?;

    let mut dx_host = vec![0.0f32; N_ELEM_IN];
    dx_buf.to_host(&mut dx_host)?;

    for i in 0..N_ELEM_IN {
        assert_eq!(
            dx_host[i], expected[i],
            "dx mismatch at {i}: gpu={} expected={}",
            dx_host[i], expected[i],
        );
    }

    Ok(())
}
