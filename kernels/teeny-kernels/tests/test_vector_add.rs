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
use teeny_core::context::buffer::Buffer;
use teeny_core::context::device::Device;
use teeny_core::context::program::Kernel;
use teeny_cuda::errors::Result;
use teeny_cuda::testing;
use teeny_cuda::target::Capability;

const N: usize = 1024;
const BLOCK_SIZE: i32 = 128;

#[test]
fn test_tensor_add() -> Result<()> {
    dotenv()?;

    let kernel = teeny_kernels::math::add::VectorAdd::<f32, 1024>::new();
    let target = Target::new(Capability::Sm90);
    let ptx_path = PathBuf::from(compile_kernel(&kernel, &target, true)?);
    let mlir = std::fs::read_to_string(ptx_path.with_extension("mlir"))?;

    assert_debug_snapshot!("vector_add_source", kernel.source());
    assert_debug_snapshot!("vector_add_mlir", mlir.trim());

    Ok(())
}

#[test]
#[cfg(feature = "cuda")]
fn test_tensor_add_gpu_execution() -> Result<()> {
    dotenv()?;

    let env = testing::setup_cuda_env()?;
    let device = env.device;

    let x_arr = Array1::<f32>::random(N, Uniform::new(-10.0f32, 10.0f32).unwrap());
    let y_arr = Array1::<f32>::random(N, Uniform::new(-10.0f32, 10.0f32).unwrap());
    let x_host = x_arr.to_vec();
    let y_host = y_arr.to_vec();
    let mut output_host = vec![0.0f32; N];

    let expected: Vec<f32> = x_arr.iter().zip(y_arr.iter()).map(|(x, y)| x + y).collect();

    let mut x_buf = device.buffer::<f32>(N)?;
    let mut y_buf = device.buffer::<f32>(N)?;
    let out_buf = device.buffer::<f32>(N)?;
    println!(
        "[4/9] allocated device buffers: x={:#x} y={:#x} out={:#x}",
        x_buf.as_device_ptr(),
        y_buf.as_device_ptr(),
        out_buf.as_device_ptr(),
    );

    x_buf.to_device(&x_host)?;
    y_buf.to_device(&y_host)?;
    println!("[5/9] copied input data to device");

    let kernel = teeny_kernels::math::add::VectorAdd::<f32, BLOCK_SIZE>::new();
    let target = Target::new(env.capability);
    let ptx_path = compile_kernel(&kernel, &target, true)?;
    println!("[6/9] compiled PTX: {ptx_path}");
    let ptx = std::fs::read(&ptx_path)?;

    let program =
        testing::load_program_from_ptx::<teeny_kernels::math::add::VectorAdd<f32, BLOCK_SIZE>>(
            &ptx,
        )?;

    let cfg = testing::launch_config(N, BLOCK_SIZE);
    println!(
        "[8/9] launching: grid={:?} block={:?} n_elements={N}",
        cfg.grid, cfg.block,
    );

    let args = (
        x_buf.as_device_ptr() as *mut f32,
        y_buf.as_device_ptr() as *mut f32,
        out_buf.as_device_ptr() as *mut f32,
        N as i32,
    );

    device.launch(&program, &cfg, args)?;
    println!("      kernel completed (synchronized)");

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
            "vector add mismatch at index {i}: x={}, y={}, gpu={}, expected={}",
            x_host[i], y_host[i], output_host[i], expected[i]
        );
    }

    Ok(())
}
