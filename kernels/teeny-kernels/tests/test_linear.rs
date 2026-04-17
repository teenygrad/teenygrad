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

use dotenv::dotenv;
use ndarray::Array1;
use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::Uniform;
use teeny_compiler::compiler::{driver::cuda::compile_kernel, target::cuda::Target};
use teeny_core::context::buffer::Buffer;
use teeny_core::context::device::Device;
use teeny_cuda::errors::Result;
use teeny_cuda::testing;

const N: usize = 1024;
const BLOCK_SIZE: i32 = 128;

#[test]
#[cfg(feature = "cuda")]
fn test_linear_no_bias_gpu_execution() -> Result<()> {
    dotenv()?;
    let env = testing::setup_cuda_env()?;
    let device = env.device;

    let input_arr = Array1::<f32>::random(N, Uniform::new(-5.0f32, 5.0f32).unwrap());
    let weight_arr = Array1::<f32>::random(N, Uniform::new(-5.0f32, 5.0f32).unwrap());
    let bias_arr = Array1::<f32>::zeros(N);

    let input_host = input_arr.to_vec();
    let weight_host = weight_arr.to_vec();
    let bias_host = bias_arr.to_vec();
    let mut output_host = vec![0.0f32; N];

    // Reference: element-wise multiply (no bias)
    let expected: Vec<f32> = input_arr
        .iter()
        .zip(weight_arr.iter())
        .map(|(x, w)| x * w)
        .collect();

    let mut in_buf = device.buffer::<f32>(N)?;
    let mut w_buf = device.buffer::<f32>(N)?;
    let mut bias_buf = device.buffer::<f32>(N)?;
    let out_buf = device.buffer::<f32>(N)?;

    in_buf.to_device(&input_host)?;
    w_buf.to_device(&weight_host)?;
    bias_buf.to_device(&bias_host)?;

    let kernel = teeny_kernels::mlp::linear::Linear::<f32, false, BLOCK_SIZE>::new();
    let target = Target::new(env.capability);
    let ptx_path = compile_kernel(&kernel, &target, true)?;
    println!("[6/9] compiled PTX: {ptx_path}");
    let ptx = std::fs::read(&ptx_path)?;

    let program = testing::load_program_from_ptx::<
        teeny_kernels::mlp::linear::Linear<f32, false, BLOCK_SIZE>,
    >(&ptx)?;

    let cfg = testing::launch_config(N, BLOCK_SIZE);
    println!(
        "[8/9] launching: grid={:?} block={:?} n_elements={N}",
        cfg.grid, cfg.block,
    );

    let args = (
        in_buf.as_device_ptr() as *mut f32,
        w_buf.as_device_ptr() as *mut f32,
        bias_buf.as_device_ptr() as *mut f32,
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
        assert!(
            (output_host[i] - expected[i]).abs() < 1e-5,
            "linear (no bias) mismatch at index {i}: input={}, weight={}, gpu={}, expected={}",
            input_host[i],
            weight_host[i],
            output_host[i],
            expected[i]
        );
    }

    Ok(())
}

#[test]
#[cfg(feature = "cuda")]
fn test_linear_with_bias_gpu_execution() -> Result<()> {
    dotenv()?;
    let env = testing::setup_cuda_env()?;
    let device = env.device;

    let input_arr = Array1::<f32>::random(N, Uniform::new(-5.0f32, 5.0f32).unwrap());
    let weight_arr = Array1::<f32>::random(N, Uniform::new(-5.0f32, 5.0f32).unwrap());
    let bias_arr = Array1::<f32>::random(N, Uniform::new(-2.0f32, 2.0f32).unwrap());

    let input_host = input_arr.to_vec();
    let weight_host = weight_arr.to_vec();
    let bias_host = bias_arr.to_vec();
    let mut output_host = vec![0.0f32; N];

    // Reference: element-wise multiply + bias
    let expected: Vec<f32> = input_arr
        .iter()
        .zip(weight_arr.iter())
        .zip(bias_arr.iter())
        .map(|((x, w), b)| x * w + b)
        .collect();

    let mut in_buf = device.buffer::<f32>(N)?;
    let mut w_buf = device.buffer::<f32>(N)?;
    let mut bias_buf = device.buffer::<f32>(N)?;
    let out_buf = device.buffer::<f32>(N)?;

    in_buf.to_device(&input_host)?;
    w_buf.to_device(&weight_host)?;
    bias_buf.to_device(&bias_host)?;

    let kernel = teeny_kernels::mlp::linear::Linear::<f32, true, BLOCK_SIZE>::new();
    let target = Target::new(env.capability);
    let ptx_path = compile_kernel(&kernel, &target, true)?;
    println!("[6/9] compiled PTX: {ptx_path}");
    let ptx = std::fs::read(&ptx_path)?;

    let program = testing::load_program_from_ptx::<
        teeny_kernels::mlp::linear::Linear<f32, true, BLOCK_SIZE>,
    >(&ptx)?;

    let cfg = testing::launch_config(N, BLOCK_SIZE);
    println!(
        "[8/9] launching: grid={:?} block={:?} n_elements={N}",
        cfg.grid, cfg.block,
    );

    let args = (
        in_buf.as_device_ptr() as *mut f32,
        w_buf.as_device_ptr() as *mut f32,
        bias_buf.as_device_ptr() as *mut f32,
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
        assert!(
            (output_host[i] - expected[i]).abs() < 1e-5,
            "linear (with bias) mismatch at index {i}: input={}, weight={}, bias={}, gpu={}, expected={}",
            input_host[i],
            weight_host[i],
            bias_host[i],
            output_host[i],
            expected[i]
        );
    }

    Ok(())
}
