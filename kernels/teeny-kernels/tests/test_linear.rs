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
use insta::assert_debug_snapshot;
use ndarray::Array2;
use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::Uniform;
use std::path::PathBuf;
use teeny_compiler::compiler::{driver::cuda::compile_kernel, target::cuda::Target};
use teeny_core::device::Device;
use teeny_core::device::buffer::Buffer;
use teeny_core::device::program::Kernel;

#[cfg(feature = "cuda")]
use teeny_cuda::{compiler::target::Capability, device::CudaLaunchConfig, errors::Result, testing};

const M: usize = 64;
const N: usize = 48;
const K: usize = 64;
const BLOCK_M: i32 = 32;
const BLOCK_N: i32 = 32;
const BLOCK_K: i32 = 32;
const GROUP_M: i32 = 8;

/// Must match `.reqntid` in the generated PTX (see e.g. `/tmp/teenygrad_rustc/linear_*.o`).
const PTX_LAUNCH_THREADS_X: u32 = 128;

#[test]
fn test_linear_mlir_without_bias_output() -> Result<()> {
    dotenv()?;

    let kernel =
        teeny_kernels::mlp::linear::Linear::<f32, false, BLOCK_M, BLOCK_N, BLOCK_K, GROUP_M>::new();
    let target = Target::new(Capability::Sm90);
    let ptx_path = PathBuf::from(compile_kernel(&kernel, &target, true)?);
    let mlir = std::fs::read_to_string(ptx_path.with_extension("mlir"))?;

    assert_debug_snapshot!("linear_source", kernel.source());
    assert_debug_snapshot!("linear_mlir", mlir.trim());

    Ok(())
}

#[test]
fn test_linear_mlir_with_bias_output() -> Result<()> {
    dotenv()?;

    let kernel =
        teeny_kernels::mlp::linear::Linear::<f32, true, BLOCK_M, BLOCK_N, BLOCK_K, GROUP_M>::new();
    let target = Target::new(Capability::Sm90);
    let ptx_path = PathBuf::from(compile_kernel(&kernel, &target, true)?);
    let mlir = std::fs::read_to_string(ptx_path.with_extension("mlir"))?;

    assert_debug_snapshot!("linear_with_bias_source", kernel.source());
    assert_debug_snapshot!("linear_with_bias_mlir", mlir.trim());

    Ok(())
}

#[test]
fn test_linear_backward_mlir_without_bias_output() -> Result<()> {
    dotenv()?;

    let kernel = teeny_kernels::mlp::linear::LinearBackward::<
        f32,
        false,
        BLOCK_M,
        BLOCK_N,
        BLOCK_K,
        GROUP_M,
    >::new();
    let target = Target::new(Capability::Sm90);
    let ptx_path = PathBuf::from(compile_kernel(&kernel, &target, true)?);
    let mlir = std::fs::read_to_string(ptx_path.with_extension("mlir"))?;

    assert_debug_snapshot!("linear_backward_source", kernel.source());
    assert_debug_snapshot!("linear_backward_mlir", mlir.trim());

    Ok(())
}

#[test]
fn test_linear_backward_mlir_with_bias_output() -> Result<()> {
    dotenv()?;

    let kernel = teeny_kernels::mlp::linear::LinearBackward::<
        f32,
        true,
        BLOCK_M,
        BLOCK_N,
        BLOCK_K,
        GROUP_M,
    >::new();
    let target = Target::new(Capability::Sm90);
    let ptx_path = PathBuf::from(compile_kernel(&kernel, &target, true)?);
    let mlir = std::fs::read_to_string(ptx_path.with_extension("mlir"))?;

    assert_debug_snapshot!("linear_backward_with_bias_source", kernel.source());
    assert_debug_snapshot!("linear_backward_with_bias_mlir", mlir.trim());

    Ok(())
}

#[test]
#[cfg(feature = "cuda")]
fn test_linear_forward_no_bias_cuda() -> Result<()> {
    dotenv()?;
    let env = testing::setup_cuda_env()?;
    let device = env.device;

    let input_arr = Array2::<f32>::random((M, K), Uniform::new(-5.0f32, 5.0f32).unwrap());
    let weight_arr = Array2::<f32>::random((N, K), Uniform::new(-5.0f32, 5.0f32).unwrap());
    let bias_arr = vec![0.0f32; N];

    let input_host = input_arr.iter().copied().collect::<Vec<_>>();
    let weight_host = weight_arr.iter().copied().collect::<Vec<_>>();
    let bias_host = bias_arr;
    let mut output_host = vec![0.0f32; M * N];

    let mut expected = vec![0.0f32; M * N];
    for m in 0..M {
        for n in 0..N {
            let mut acc = 0.0f32;
            for k in 0..K {
                acc += input_arr[[m, k]] * weight_arr[[n, k]];
            }
            expected[m * N + n] = acc;
        }
    }

    let mut in_buf = device.buffer::<f32>(M * K)?;
    let mut w_buf = device.buffer::<f32>(N * K)?;
    let mut bias_buf = device.buffer::<f32>(N)?;
    let out_buf = device.buffer::<f32>(M * N)?;

    in_buf.to_device(&input_host)?;
    w_buf.to_device(&weight_host)?;
    bias_buf.to_device(&bias_host)?;

    let kernel =
        teeny_kernels::mlp::linear::Linear::<f32, false, BLOCK_M, BLOCK_N, BLOCK_K, GROUP_M>::new();
    let target = Target::new(env.capability);
    let ptx_path = compile_kernel(&kernel, &target, true)?;
    println!("[6/9] compiled PTX: {ptx_path}");
    let ptx = std::fs::read(&ptx_path)?;

    let program = testing::load_program_from_ptx::<
        teeny_kernels::mlp::linear::Linear<f32, false, BLOCK_M, BLOCK_N, BLOCK_K, GROUP_M>,
    >(&ptx)?;

    let grid_x = (M as u32).div_ceil(BLOCK_M as u32) * (N as u32).div_ceil(BLOCK_N as u32);
    let cfg = CudaLaunchConfig {
        grid: [grid_x, 1, 1],
        block: [PTX_LAUNCH_THREADS_X, 1, 1],
        cluster: [1, 1, 1],
    };
    println!(
        "[8/9] launching: grid={:?} block={:?} M={M} N={N} K={K}",
        cfg.grid, cfg.block,
    );

    let args = (
        in_buf.as_device_ptr() as *mut f32,
        w_buf.as_device_ptr() as *mut f32,
        bias_buf.as_device_ptr() as *mut f32,
        out_buf.as_device_ptr() as *mut f32,
        M as i32,
        N as i32,
        K as i32,
        K as i32,
        1i32,
        K as i32,
        1i32,
        N as i32,
        1i32,
    );

    device.launch(&program, &cfg, args)?;
    println!("      kernel completed (synchronized)");

    out_buf.to_host(&mut output_host)?;
    println!(
        "[9/9] copied results back: output[0]={} output[{}]={}",
        output_host[0],
        (M * N) - 1,
        output_host[(M * N) - 1]
    );

    for i in 0..(M * N) {
        assert!(
            (output_host[i] - expected[i]).abs() < 5e-3,
            "linear (no bias) mismatch at index {i}: gpu={}, expected={}",
            output_host[i],
            expected[i]
        );
    }

    Ok(())
}

#[test]
#[cfg(feature = "cuda")]
fn test_linear_forward_with_bias_cuda() -> Result<()> {
    dotenv()?;
    let env = testing::setup_cuda_env()?;
    let device = env.device;

    let input_arr = Array2::<f32>::random((M, K), Uniform::new(-5.0f32, 5.0f32).unwrap());
    let weight_arr = Array2::<f32>::random((N, K), Uniform::new(-5.0f32, 5.0f32).unwrap());
    let bias_arr = Array2::<f32>::random((1, N), Uniform::new(-2.0f32, 2.0f32).unwrap());

    let input_host = input_arr.iter().copied().collect::<Vec<_>>();
    let weight_host = weight_arr.iter().copied().collect::<Vec<_>>();
    let bias_host = bias_arr.iter().copied().collect::<Vec<_>>();
    let mut output_host = vec![0.0f32; M * N];

    let mut expected = vec![0.0f32; M * N];
    for m in 0..M {
        for n in 0..N {
            let mut acc = 0.0f32;
            for k in 0..K {
                acc += input_arr[[m, k]] * weight_arr[[n, k]];
            }
            expected[m * N + n] = acc + bias_host[n];
        }
    }

    let mut in_buf = device.buffer::<f32>(M * K)?;
    let mut w_buf = device.buffer::<f32>(N * K)?;
    let mut bias_buf = device.buffer::<f32>(N)?;
    let out_buf = device.buffer::<f32>(M * N)?;

    in_buf.to_device(&input_host)?;
    w_buf.to_device(&weight_host)?;
    bias_buf.to_device(&bias_host)?;

    let kernel =
        teeny_kernels::mlp::linear::Linear::<f32, true, BLOCK_M, BLOCK_N, BLOCK_K, GROUP_M>::new();
    let target = Target::new(env.capability);
    let ptx_path = compile_kernel(&kernel, &target, true)?;
    println!("[6/9] compiled PTX: {ptx_path}");
    let ptx = std::fs::read(&ptx_path)?;

    let program = testing::load_program_from_ptx::<
        teeny_kernels::mlp::linear::Linear<f32, true, BLOCK_M, BLOCK_N, BLOCK_K, GROUP_M>,
    >(&ptx)?;

    let grid_x = (M as u32).div_ceil(BLOCK_M as u32) * (N as u32).div_ceil(BLOCK_N as u32);
    let cfg = CudaLaunchConfig {
        grid: [grid_x, 1, 1],
        block: [PTX_LAUNCH_THREADS_X, 1, 1],
        cluster: [1, 1, 1],
    };
    println!(
        "[8/9] launching: grid={:?} block={:?} M={M} N={N} K={K}",
        cfg.grid, cfg.block,
    );

    let args = (
        in_buf.as_device_ptr() as *mut f32,
        w_buf.as_device_ptr() as *mut f32,
        bias_buf.as_device_ptr() as *mut f32,
        out_buf.as_device_ptr() as *mut f32,
        M as i32,
        N as i32,
        K as i32,
        K as i32,
        1i32,
        K as i32,
        1i32,
        N as i32,
        1i32,
    );

    device.launch(&program, &cfg, args)?;
    println!("      kernel completed (synchronized)");

    out_buf.to_host(&mut output_host)?;
    println!(
        "[9/9] copied results back: output[0]={} output[{}]={}",
        output_host[0],
        (M * N) - 1,
        output_host[(M * N) - 1]
    );

    for i in 0..(M * N) {
        assert!(
            (output_host[i] - expected[i]).abs() < 5e-3,
            "linear (with bias) mismatch at index {i}: gpu={}, expected={}",
            output_host[i],
            expected[i]
        );
    }

    Ok(())
}

#[test]
#[cfg(feature = "cuda")]
fn test_linear_backward_without_bias_cuda() -> Result<()> {
    dotenv()?;
    let env = testing::setup_cuda_env()?;
    let device = env.device;

    let input_arr = Array2::<f32>::random((M, K), Uniform::new(-5.0f32, 5.0f32).unwrap());
    let weight_arr = Array2::<f32>::random((N, K), Uniform::new(-5.0f32, 5.0f32).unwrap());
    let dy_arr = Array2::<f32>::random((M, N), Uniform::new(-1.0f32, 1.0f32).unwrap());

    let input_host = input_arr.iter().copied().collect::<Vec<_>>();
    let weight_host = weight_arr.iter().copied().collect::<Vec<_>>();
    let dy_host: Vec<f32> = dy_arr.iter().copied().collect();

    let mut dx_host = vec![0.0f32; M * K];
    let mut dw_host = vec![0.0f32; N * K];

    let mut x_buf = device.buffer::<f32>(M * K)?;
    let mut w_buf = device.buffer::<f32>(N * K)?;
    let mut dy_buf = device.buffer::<f32>(M * N)?;
    let dx_buf = device.buffer::<f32>(M * K)?;
    let dw_buf = device.buffer::<f32>(N * K)?;
    let db_buf = device.buffer::<f32>(N)?;

    x_buf.to_device(&input_host)?;
    w_buf.to_device(&weight_host)?;
    dy_buf.to_device(&dy_host)?;

    let kernel = teeny_kernels::mlp::linear::LinearBackward::<
        f32,
        false,
        BLOCK_M,
        BLOCK_N,
        BLOCK_K,
        GROUP_M,
    >::new();
    let target = Target::new(env.capability);
    let ptx_path = compile_kernel(&kernel, &target, true)?;
    println!("[linear_backward no bias] compiled PTX: {ptx_path}");
    let ptx = std::fs::read(&ptx_path)?;

    let program = testing::load_program_from_ptx::<
        teeny_kernels::mlp::linear::LinearBackward<f32, false, BLOCK_M, BLOCK_N, BLOCK_K, GROUP_M>,
    >(&ptx)?;

    // 3D grid: ceil(M/BLOCK_M) * ceil(N/BLOCK_N) * ceil(K/BLOCK_K)
    let grid_x = (M as u32).div_ceil(BLOCK_M as u32)
        * (N as u32).div_ceil(BLOCK_N as u32)
        * (K as u32).div_ceil(BLOCK_K as u32);
    let cfg = CudaLaunchConfig {
        grid: [grid_x, 1, 1],
        block: [PTX_LAUNCH_THREADS_X, 1, 1],
        cluster: [1, 1, 1],
    };

    // Strides match row-major [M,K], [N,K], [M,N], etc.; see `linear_backward` parameter order.
    let args = (
        x_buf.as_device_ptr() as *mut f32,
        w_buf.as_device_ptr() as *mut f32,
        dy_buf.as_device_ptr() as *mut f32,
        dx_buf.as_device_ptr() as *mut f32,
        dw_buf.as_device_ptr() as *mut f32,
        db_buf.as_device_ptr() as *mut f32,
        M as i32,
        N as i32,
        K as i32,
        K as i32,
        1i32,
        1i32,
        K as i32,
        N as i32,
        1i32,
        K as i32,
        1i32,
        1i32,
        K as i32,
        1i32,
    );

    device.launch(&program, &cfg, args)?;
    dx_buf.to_host(&mut dx_host)?;
    dw_buf.to_host(&mut dw_host)?;

    let mut dx_expected = vec![0f32; M * K];
    let mut dw_expected = vec![0f32; N * K];

    for m in 0..M {
        for k in 0..K {
            let mut acc = 0f32;
            for n in 0..N {
                acc += dy_host[m * N + n] * weight_host[n * K + k];
            }
            dx_expected[m * K + k] = acc;
        }
    }
    for n in 0..N {
        for k in 0..K {
            let mut acc = 0f32;
            for m in 0..M {
                acc += dy_host[m * N + n] * input_host[m * K + k];
            }
            dw_expected[n * K + k] = acc;
        }
    }

    for i in 0..(M * K) {
        assert!(
            (dx_host[i] - dx_expected[i]).abs() < 5e-3,
            "linear_backward (dx, no bias) mismatch at index {i}: gpu={}, expected={}",
            dx_host[i],
            dx_expected[i]
        );
    }

    for i in 0..(N * K) {
        assert!(
            (dw_host[i] - dw_expected[i]).abs() < 5e-3,
            "linear_backward (dw, no bias) mismatch at index {i}: gpu={}, expected={}",
            dw_host[i],
            dw_expected[i]
        );
    }

    Ok(())
}

#[test]
#[cfg(feature = "cuda")]
fn test_linear_backward_with_bias_cuda() -> Result<()> {
    dotenv()?;
    let env = testing::setup_cuda_env()?;
    let device = env.device;

    let input_arr = Array2::<f32>::random((M, K), Uniform::new(-5.0f32, 5.0f32).unwrap());
    let weight_arr = Array2::<f32>::random((N, K), Uniform::new(-5.0f32, 5.0f32).unwrap());
    let dy_arr = Array2::<f32>::random((M, N), Uniform::new(-1.0f32, 1.0f32).unwrap());

    let input_host = input_arr.iter().copied().collect::<Vec<_>>();
    let weight_host = weight_arr.iter().copied().collect::<Vec<_>>();
    let dy_host: Vec<f32> = dy_arr.iter().copied().collect();

    let mut dx_host = vec![0.0f32; M * K];
    let mut dw_host = vec![0.0f32; N * K];
    let mut db_host = vec![0.0f32; N];

    let mut x_buf = device.buffer::<f32>(M * K)?;
    let mut w_buf = device.buffer::<f32>(N * K)?;
    let mut dy_buf = device.buffer::<f32>(M * N)?;
    let dx_buf = device.buffer::<f32>(M * K)?;
    let dw_buf = device.buffer::<f32>(N * K)?;
    let db_buf = device.buffer::<f32>(N)?;

    x_buf.to_device(&input_host)?;
    w_buf.to_device(&weight_host)?;
    dy_buf.to_device(&dy_host)?;

    let kernel = teeny_kernels::mlp::linear::LinearBackward::<
        f32,
        true,
        BLOCK_M,
        BLOCK_N,
        BLOCK_K,
        GROUP_M,
    >::new();
    let target = Target::new(env.capability);
    let ptx_path = compile_kernel(&kernel, &target, true)?;
    println!("[linear_backward with bias] compiled PTX: {ptx_path}");
    let ptx = std::fs::read(&ptx_path)?;

    let program = testing::load_program_from_ptx::<
        teeny_kernels::mlp::linear::LinearBackward<f32, true, BLOCK_M, BLOCK_N, BLOCK_K, GROUP_M>,
    >(&ptx)?;

    let grid_x = (M as u32).div_ceil(BLOCK_M as u32)
        * (N as u32).div_ceil(BLOCK_N as u32)
        * (K as u32).div_ceil(BLOCK_K as u32);
    let cfg = CudaLaunchConfig {
        grid: [grid_x, 1, 1],
        block: [PTX_LAUNCH_THREADS_X, 1, 1],
        cluster: [1, 1, 1],
    };

    let args = (
        x_buf.as_device_ptr() as *mut f32,
        w_buf.as_device_ptr() as *mut f32,
        dy_buf.as_device_ptr() as *mut f32,
        dx_buf.as_device_ptr() as *mut f32,
        dw_buf.as_device_ptr() as *mut f32,
        db_buf.as_device_ptr() as *mut f32,
        M as i32,
        N as i32,
        K as i32,
        K as i32,
        1i32,
        1i32,
        K as i32,
        N as i32,
        1i32,
        K as i32,
        1i32,
        1i32,
        K as i32,
        1i32,
    );

    device.launch(&program, &cfg, args)?;
    dx_buf.to_host(&mut dx_host)?;
    dw_buf.to_host(&mut dw_host)?;
    db_buf.to_host(&mut db_host)?;

    let mut dx_expected = vec![0f32; M * K];
    let mut dw_expected = vec![0f32; N * K];
    let mut db_expected = vec![0f32; N];

    for m in 0..M {
        for k in 0..K {
            let mut acc = 0f32;
            for n in 0..N {
                acc += dy_host[m * N + n] * weight_host[n * K + k];
            }
            dx_expected[m * K + k] = acc;
        }
    }
    for n in 0..N {
        for k in 0..K {
            let mut acc = 0f32;
            for m in 0..M {
                acc += dy_host[m * N + n] * input_host[m * K + k];
            }
            dw_expected[n * K + k] = acc;
        }
    }
    for n in 0..N {
        let mut acc = 0f32;
        for m in 0..M {
            acc += dy_host[m * N + n];
        }
        db_expected[n] = acc;
    }

    for i in 0..(M * K) {
        assert!(
            (dx_host[i] - dx_expected[i]).abs() < 5e-3,
            "linear_backward (dx, with bias) mismatch at index {i}: gpu={}, expected={}",
            dx_host[i],
            dx_expected[i]
        );
    }

    for i in 0..(N * K) {
        assert!(
            (dw_host[i] - dw_expected[i]).abs() < 5e-3,
            "linear_backward (dw, with bias) mismatch at index {i}: gpu={}, expected={}",
            dw_host[i],
            dw_expected[i]
        );
    }

    for i in 0..N {
        assert!(
            (db_host[i] - db_expected[i]).abs() < 5e-3,
            "linear_backward (db) mismatch at index {i}: gpu={}, expected={}",
            db_host[i],
            db_expected[i]
        );
    }

    Ok(())
}
