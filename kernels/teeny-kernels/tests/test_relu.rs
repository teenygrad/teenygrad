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

use std::path::PathBuf;

use dotenv::dotenv;
use insta::assert_debug_snapshot;
use ndarray::Array1;
use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::Uniform;
use teeny_compiler::compiler::{driver::cuda::compile_kernel, target::cuda::Target};
use teeny_core::device::Device;
use teeny_core::device::buffer::Buffer;
use teeny_core::device::program::Kernel;

#[cfg(feature = "cuda")]
use teeny_cuda::{compiler::target::Capability, errors::Result, testing};

const N: usize = 1024;
const BLOCK_SIZE: i32 = 128;

#[test]
fn test_relu() -> Result<()> {
    dotenv()?;

    let kernel = teeny_kernels::activation::relu::ReluForward::<f32, 1024>::new();
    let target = Target::new(Capability::Sm90);
    let ptx_path = PathBuf::from(compile_kernel(&kernel, &target, true)?);
    let mlir = std::fs::read_to_string(ptx_path.with_extension("mlir"))?;

    assert_debug_snapshot!("relu_source", kernel.source());
    assert_debug_snapshot!("relu_mlir", mlir.trim());

    Ok(())
}

#[test]
#[cfg(feature = "cuda")]
fn test_relu_forward_gpu() -> Result<()> {
    dotenv()?;
    let env = testing::setup_cuda_env()?;
    let device = env.device;

    // ── Host data: random values in [-5, 5] to exercise both negative and positive inputs ──
    let input_arr = Array1::<f32>::random(N, Uniform::new(-5.0f32, 5.0f32).unwrap());
    let input_host: Vec<f32> = input_arr.to_vec();
    let mut output_host = vec![0.0f32; N];

    // ── Reference: ndarray relu ────────────────────────────────────────────
    let expected: Vec<f32> = input_arr.mapv(|x| x.max(0.0)).to_vec();

    // ── Device buffers ─────────────────────────────────────────────────────
    let mut in_buf = device.buffer::<f32>(N)?;
    let out_buf = device.buffer::<f32>(N)?;
    println!(
        "[4/9] allocated device buffers: in={:#x} out={:#x}",
        in_buf.as_device_ptr(),
        out_buf.as_device_ptr(),
    );

    in_buf.to_device(&input_host)?;
    println!("[5/9] copied input data to device");

    // ── Compile PTX ────────────────────────────────────────────────────────
    let kernel = teeny_kernels::activation::relu::ReluForward::<f32, BLOCK_SIZE>::new();
    let target = Target::new(env.capability);
    let ptx_path = compile_kernel(&kernel, &target, true)?;
    println!("[6/9] compiled PTX: {ptx_path}");
    let ptx = std::fs::read(&ptx_path)?;

    let program = testing::load_program_from_ptx::<
        teeny_kernels::activation::relu::ReluForward<f32, BLOCK_SIZE>,
    >(&ptx)?;

    // ── Launch ─────────────────────────────────────────────────────────────
    let cfg = testing::launch_config(N, BLOCK_SIZE);
    println!(
        "[8/9] launching: grid={:?} block={:?} n_elements={N}",
        cfg.grid, cfg.block,
    );

    let args = (
        in_buf.as_device_ptr() as *mut f32,
        out_buf.as_device_ptr() as *mut f32,
        N as i32,
    );

    device.launch(&program, &cfg, args)?;
    println!("      kernel completed (synchronized)");

    // ── Copy results back and verify ───────────────────────────────────────
    out_buf.to_host(&mut output_host)?;
    println!(
        "[9/9] copied results back: output[0]={} output[{}]={}",
        output_host[0],
        N - 1,
        output_host[N - 1]
    );

    for i in 0..N {
        assert_eq!(
            output_host[i], expected[i],
            "relu mismatch at index {i}: input={}, gpu={}, expected={}",
            input_host[i], output_host[i], expected[i]
        );
    }

    Ok(())
}

#[test]
#[cfg(feature = "cuda")]
fn test_relu_backward_gpu() -> Result<()> {
    dotenv()?;
    let env = testing::setup_cuda_env()?;
    let device = env.device;

    // ── Host data ─────────────────────────────────────────────────────────────
    // y is relu output (non-negative); mix of zeros and positives to exercise both branches.
    let y_arr = Array1::<f32>::random(N, Uniform::new(0.0f32, 5.0f32).unwrap())
        .mapv(|v| if v < 1.0 { 0.0 } else { v });
    let dy_arr = Array1::<f32>::random(N, Uniform::new(-5.0f32, 5.0f32).unwrap());

    let y_host: Vec<f32> = y_arr.to_vec();
    let dy_host: Vec<f32> = dy_arr.to_vec();
    let mut dx_host = vec![0.0f32; N];

    // ── Reference: pass gradient through where y > 0 ──────────────────────────
    let expected: Vec<f32> = y_arr
        .iter()
        .zip(dy_arr.iter())
        .map(|(&y, &dy)| if y > 0.0 { dy } else { 0.0 })
        .collect();

    // ── Device buffers ─────────────────────────────────────────────────────────
    let mut dy_buf = device.buffer::<f32>(N)?;
    let mut y_buf = device.buffer::<f32>(N)?;
    let dx_buf = device.buffer::<f32>(N)?;

    dy_buf.to_device(&dy_host)?;
    y_buf.to_device(&y_host)?;

    // ── Compile PTX ────────────────────────────────────────────────────────────
    let kernel = teeny_kernels::activation::relu::ReluBackward::<f32, BLOCK_SIZE>::new();
    let target = Target::new(env.capability);
    let ptx_path = compile_kernel(&kernel, &target, true)?;
    let ptx = std::fs::read(&ptx_path)?;

    let program = testing::load_program_from_ptx::<
        teeny_kernels::activation::relu::ReluBackward<f32, BLOCK_SIZE>,
    >(&ptx)?;

    // ── Launch ─────────────────────────────────────────────────────────────────
    let cfg = testing::launch_config(N, BLOCK_SIZE);
    let args = (
        dy_buf.as_device_ptr() as *mut f32,
        y_buf.as_device_ptr() as *mut f32,
        dx_buf.as_device_ptr() as *mut f32,
        N as i32,
    );

    device.launch(&program, &cfg, args)?;

    // ── Verify ─────────────────────────────────────────────────────────────────
    dx_buf.to_host(&mut dx_host)?;

    for i in 0..N {
        assert_eq!(
            dx_host[i], expected[i],
            "relu_backward mismatch at index {i}: y={}, dy={}, gpu={}, expected={}",
            y_host[i], dy_host[i], dx_host[i], expected[i]
        );
    }

    Ok(())
}
