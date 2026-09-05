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

// ChannelCat forward and backward tests.
//
// Kernel: channel_cat_forward / channel_cat_backward
// Layout: flat NC (channels-last), index = n_spatial * C + c
//
// Dimensions:
//   N_SPATIAL = 64   (B=2, H=4, W=8)
//   C_TOTAL   = 32
//   CHUNK_C   = 16   (two equal chunks concatenated)
//
// Forward: two calls — one per input chunk — scatter each into y.
// Backward: two calls — one per input chunk — gather each from dy.
//
// Grid: N_SPATIAL * cdiv(CHUNK_C, BLOCK_SIZE) CTAs per call.

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
const N_ELEM_CHUNK: usize = N_SPATIAL * CHUNK_C; // 1024
#[cfg(feature = "hardware")]
const N_ELEM_OUT: usize = N_SPATIAL * C_TOTAL; // 2048

// ── Fixture loader ────────────────────────────────────────────────────────────

// ── MLIR snapshot tests ───────────────────────────────────────────────────────

#[test]
fn test_channel_cat_forward_snapshot() -> anyhow::Result<()> {
    dotenv().ok();

    let kernel = teeny_kernels::nn::tensor::channel_cat::ChannelCatForward::<f32>::new(BLOCK_SIZE);
    let target = teeny_runtime::reference_target();
    let ptx_path = PathBuf::from(teeny_runtime::compile_kernel(
        &kernel, &target, true, false,
    )?);
    let mlir = std::fs::read_to_string(ptx_path.with_extension("mlir"))?;

    assert_debug_snapshot!(
        format!("channel_cat_forward_source_{}", teeny_runtime::BACKEND_NAME),
        kernel.source()
    );
    assert_debug_snapshot!(
        format!("channel_cat_forward_mlir_{}", teeny_runtime::BACKEND_NAME),
        mlir.trim()
    );

    Ok(())
}

#[test]
fn test_channel_cat_backward_snapshot() -> anyhow::Result<()> {
    dotenv().ok();

    let kernel = teeny_kernels::nn::tensor::channel_cat::ChannelCatBackward::<f32>::new(BLOCK_SIZE);
    let target = teeny_runtime::reference_target();
    let ptx_path = PathBuf::from(teeny_runtime::compile_kernel(
        &kernel, &target, true, false,
    )?);
    let mlir = std::fs::read_to_string(ptx_path.with_extension("mlir"))?;

    assert_debug_snapshot!(
        format!(
            "channel_cat_backward_source_{}",
            teeny_runtime::BACKEND_NAME
        ),
        kernel.source()
    );
    assert_debug_snapshot!(
        format!("channel_cat_backward_mlir_{}", teeny_runtime::BACKEND_NAME),
        mlir.trim()
    );

    Ok(())
}

// ── CUDA forward test ─────────────────────────────────────────────────────────

#[test]
#[cfg(feature = "hardware")]
fn test_channel_cat_forward_cuda() -> anyhow::Result<()> {
    dotenv().ok();
    let device = teeny_runtime::open()?;

    let x0_host = load_fixture(env!("CARGO_MANIFEST_DIR"), "channel_cat/x0.bin");
    let x1_host = load_fixture(env!("CARGO_MANIFEST_DIR"), "channel_cat/x1.bin");
    let expected_cat = load_fixture(env!("CARGO_MANIFEST_DIR"), "channel_cat/expected_cat.bin");

    assert_eq!(x0_host.len(), N_ELEM_CHUNK);
    assert_eq!(x1_host.len(), N_ELEM_CHUNK);
    assert_eq!(expected_cat.len(), N_ELEM_OUT);

    let mut x0_buf = device.buffer::<f32>(N_ELEM_CHUNK)?;
    let mut x1_buf = device.buffer::<f32>(N_ELEM_CHUNK)?;
    // y is zero-initialised; each kernel call fills its channel slice.
    let y_buf = device.buffer::<f32>(N_ELEM_OUT)?;

    x0_buf.to_device(&x0_host)?;
    x1_buf.to_device(&x1_host)?;

    let kernel = teeny_kernels::nn::tensor::channel_cat::ChannelCatForward::<f32>::new(BLOCK_SIZE);
    let target = teeny_runtime::default_target(&device)?;
    let ptx_path = teeny_runtime::compile_kernel(&kernel, &target, true, false)?;
    let program = teeny_runtime::load_program::<
        teeny_kernels::nn::tensor::channel_cat::ChannelCatForward<f32>,
    >(&ptx_path)?;

    let num_c_tiles = CHUNK_C.div_ceil(BLOCK_SIZE as usize);
    let cfg = teeny_runtime::launch_config_custom(
        [(N_SPATIAL * num_c_tiles) as u32, 1, 1],
        [BLOCK_SIZE as u32, 1, 1],
        [1, 1, 1],
    );

    // Chunk 0 → output channels [0, CHUNK_C)
    device.launch(
        &program,
        &cfg,
        (
            x0_buf.as_device_ptr(),
            y_buf.as_device_ptr(),
            CHUNK_C as i32,
            C_TOTAL as i32,
            0i32, // chunk_offset = 0
        ),
    )?;

    // Chunk 1 → output channels [CHUNK_C, C_TOTAL)
    device.launch(
        &program,
        &cfg,
        (
            x1_buf.as_device_ptr(),
            y_buf.as_device_ptr(),
            CHUNK_C as i32,
            C_TOTAL as i32,
            CHUNK_C as i32, // chunk_offset = 16
        ),
    )?;

    let mut y_host = vec![0.0f32; N_ELEM_OUT];
    y_buf.to_host(&mut y_host)?;

    for i in 0..N_ELEM_OUT {
        assert_eq!(
            y_host[i], expected_cat[i],
            "cat mismatch at {i}: gpu={} expected={}",
            y_host[i], expected_cat[i],
        );
    }

    Ok(())
}

// ── CUDA backward test ────────────────────────────────────────────────────────

#[test]
#[cfg(feature = "hardware")]
fn test_channel_cat_backward_cuda() -> anyhow::Result<()> {
    dotenv().ok();
    let device = teeny_runtime::open()?;

    let dy_host = load_fixture(env!("CARGO_MANIFEST_DIR"), "channel_cat/dy.bin");
    let expected_dx0 = load_fixture(env!("CARGO_MANIFEST_DIR"), "channel_cat/expected_dx0.bin");
    let expected_dx1 = load_fixture(env!("CARGO_MANIFEST_DIR"), "channel_cat/expected_dx1.bin");

    assert_eq!(dy_host.len(), N_ELEM_OUT);
    assert_eq!(expected_dx0.len(), N_ELEM_CHUNK);

    let mut dy_buf = device.buffer::<f32>(N_ELEM_OUT)?;
    let dx0_buf = device.buffer::<f32>(N_ELEM_CHUNK)?;
    let dx1_buf = device.buffer::<f32>(N_ELEM_CHUNK)?;
    dy_buf.to_device(&dy_host)?;

    let kernel = teeny_kernels::nn::tensor::channel_cat::ChannelCatBackward::<f32>::new(BLOCK_SIZE);
    let target = teeny_runtime::default_target(&device)?;
    let ptx_path = teeny_runtime::compile_kernel(&kernel, &target, true, false)?;
    let program = teeny_runtime::load_program::<
        teeny_kernels::nn::tensor::channel_cat::ChannelCatBackward<f32>,
    >(&ptx_path)?;

    let num_c_tiles = CHUNK_C.div_ceil(BLOCK_SIZE as usize);
    let cfg = teeny_runtime::launch_config_custom(
        [(N_SPATIAL * num_c_tiles) as u32, 1, 1],
        [BLOCK_SIZE as u32, 1, 1],
        [1, 1, 1],
    );

    // Gradient for chunk 0 (reads dy channels [0, CHUNK_C))
    device.launch(
        &program,
        &cfg,
        (
            dy_buf.as_device_ptr(),
            dx0_buf.as_device_ptr(),
            CHUNK_C as i32,
            C_TOTAL as i32,
            0i32,
        ),
    )?;

    // Gradient for chunk 1 (reads dy channels [CHUNK_C, C_TOTAL))
    device.launch(
        &program,
        &cfg,
        (
            dy_buf.as_device_ptr(),
            dx1_buf.as_device_ptr(),
            CHUNK_C as i32,
            C_TOTAL as i32,
            CHUNK_C as i32,
        ),
    )?;

    let mut dx0_host = vec![0.0f32; N_ELEM_CHUNK];
    let mut dx1_host = vec![0.0f32; N_ELEM_CHUNK];
    dx0_buf.to_host(&mut dx0_host)?;
    dx1_buf.to_host(&mut dx1_host)?;

    for i in 0..N_ELEM_CHUNK {
        assert_eq!(
            dx0_host[i], expected_dx0[i],
            "dx0 mismatch at {i}: gpu={} expected={}",
            dx0_host[i], expected_dx0[i],
        );
        assert_eq!(
            dx1_host[i], expected_dx1[i],
            "dx1 mismatch at {i}: gpu={} expected={}",
            dx1_host[i], expected_dx1[i],
        );
    }

    Ok(())
}
