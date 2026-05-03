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

// detect_decode_forward test.
//
// Converts raw LTRB box predictions to XYWH world coordinates.
// B=2, A=16, BLOCK_A=16.

use std::path::PathBuf;
use dotenv::dotenv;
use insta::assert_debug_snapshot;
use teeny_compiler::compiler::{driver::cuda::compile_kernel, target::cuda::Target};
use teeny_core::device::Device;
use teeny_core::device::buffer::Buffer;
use teeny_core::device::program::Kernel;

#[cfg(feature = "cuda")]
use teeny_cuda::{compiler::target::Capability, device::CudaLaunchConfig, errors::Result, testing};

const B:       usize = 2;
const A:       usize = 16;
const BLOCK_A: i32   = 16;

fn load(rel: &str) -> Vec<f32> {
    let path = format!("{}/tests/fixtures/{}", env!("CARGO_MANIFEST_DIR"), rel);
    let bytes = std::fs::read(&path).unwrap_or_else(|e| panic!("missing {path}: {e}"));
    bytes.chunks_exact(4).map(|b| f32::from_le_bytes([b[0],b[1],b[2],b[3]])).collect()
}

#[test]
fn test_detect_decode_forward_snapshot() -> std::result::Result<(), Box<dyn std::error::Error>> {
    dotenv().ok();
    let kernel = teeny_kernels::nn::tensor::detect_decode::DetectDecodeForward::new(BLOCK_A);
    let target = Target::new(Capability::Sm90);
    let ptx_path = PathBuf::from(compile_kernel(&kernel, &target, true)?);
    let mlir = std::fs::read_to_string(ptx_path.with_extension("mlir"))?;
    assert_debug_snapshot!("detect_decode_forward_source", kernel.source());
    assert_debug_snapshot!("detect_decode_forward_mlir", mlir.trim());
    Ok(())
}

#[test]
#[cfg(feature = "cuda")]
fn test_detect_decode_forward_cuda() -> Result<()> {
    dotenv().ok();
    let env = testing::setup_cuda_env()?;
    let device = env.device;

    let raw_boxes = load("detect_decode/raw_boxes.bin");
    let anchor_x  = load("detect_decode/anchor_x.bin");
    let anchor_y  = load("detect_decode/anchor_y.bin");
    let strides   = load("detect_decode/strides.bin");
    let expected  = load("detect_decode/expected.bin");

    assert_eq!(raw_boxes.len(), B * 4 * A);
    assert_eq!(anchor_x.len(), A);
    assert_eq!(expected.len(), B * 4 * A);

    let mut boxes_buf    = device.buffer::<f32>(B * 4 * A)?;
    let mut anchor_x_buf = device.buffer::<f32>(A)?;
    let mut anchor_y_buf = device.buffer::<f32>(A)?;
    let mut strides_buf  = device.buffer::<f32>(A)?;
    let out_buf          = device.buffer::<f32>(B * 4 * A)?;

    boxes_buf.to_device(&raw_boxes)?;
    anchor_x_buf.to_device(&anchor_x)?;
    anchor_y_buf.to_device(&anchor_y)?;
    strides_buf.to_device(&strides)?;

    let kernel = teeny_kernels::nn::tensor::detect_decode::DetectDecodeForward::new(BLOCK_A);
    let target = Target::new(env.capability);
    let ptx = std::fs::read(compile_kernel(&kernel, &target, true)?)?;
    let program = testing::load_program_from_ptx::<
        teeny_kernels::nn::tensor::detect_decode::DetectDecodeForward
    >(&ptx)?;

    let a_tiles = A.div_ceil(BLOCK_A as usize);
    let cfg = testing::launch_config_with_grid(B * a_tiles, &program);
    device.launch(&program, &cfg, (
        boxes_buf.as_device_ptr() as *mut f32,
        anchor_x_buf.as_device_ptr() as *mut f32,
        anchor_y_buf.as_device_ptr() as *mut f32,
        strides_buf.as_device_ptr() as *mut f32,
        out_buf.as_device_ptr() as *mut f32,
        B as i32,
        A as i32,
    ))?;

    let mut out_host = vec![0.0f32; B * 4 * A];
    out_buf.to_host(&mut out_host)?;

    for i in 0..B * 4 * A {
        assert!(
            (out_host[i] - expected[i]).abs() < 1e-4,
            "mismatch at {i}: gpu={} expected={}", out_host[i], expected[i]
        );
    }
    Ok(())
}
