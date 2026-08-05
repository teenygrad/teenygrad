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

//! Chapter 16 of the kernels book: sweeping the block size.
//!
//! The same kernel, compiled once per block size, timed on real data. There is
//! no autotuner in this SDK, so this is the loop you run by hand.
//!
//! ```bash
//! cargo run --release -p teeny-triton --features cuda --example block_size
//! ```
//!
//! `--release` matters: the host-side loop is doing the timing.

#![allow(non_snake_case)]

use std::time::Instant;

use anyhow::Result;
use teeny_compiler::compiler::{driver::cuda::compile_kernel, target::cuda::Target};
use teeny_core::device::Device;
use teeny_core::device::buffer::Buffer;
use teeny_core::dtype::Num;
use teeny_macros::kernel;
use teeny_triton::triton::{
    types::{AddOffsets, Comparison},
    *,
};

// ANCHOR: kernel
/// The Chapter 5 kernel, unchanged. Only `BLOCK_SIZE` varies below.
#[kernel]
pub fn sweep_add<T: Triton, D: Num, const BLOCK_SIZE: i32>(
    a_ptr: T::Pointer<D>,
    b_ptr: T::Pointer<D>,
    out_ptr: T::Pointer<D>,
    n_elements: i32,
) where
    T::I32Tensor: types::Tensor<i32, 1>,
    T::I32Tensor: Comparison<i32, BoolTensor = T::BoolTensor>,
    T::Pointer<D>: AddOffsets<i32, 1, T::I32Tensor, Output = T::Tensor<T::Pointer<D>>>,
{
    let pid = T::program_id(Axis::X);
    let offsets = T::arange(0, BLOCK_SIZE) + pid * BLOCK_SIZE;
    let in_bounds = offsets.lt(n_elements);

    let a = T::load(
        a_ptr.add_offsets(offsets),
        Some(in_bounds),
        None,
        &[],
        None,
        None,
        None,
        false,
    );
    let b = T::load(
        b_ptr.add_offsets(offsets),
        Some(in_bounds),
        None,
        &[],
        None,
        None,
        None,
        false,
    );

    T::store(
        out_ptr.add_offsets(offsets),
        a + b,
        Some(in_bounds),
        &[],
        None,
        None,
    );
}
// ANCHOR_END: kernel

// ANCHOR: sweep
/// Block sizes to try. Powers of two, all multiples of a warp's 32 lanes.
const BLOCK_SIZES: [i32; 6] = [32, 64, 128, 256, 512, 1024];

/// 32M elements — three buffers of 128 MB, far past any cache, so this is
/// squarely memory-bound and the block size is the only thing changing.
const N: usize = 32 * 1024 * 1024;

const WARMUP: usize = 5;
const ITERS: usize = 50;

fn main() -> Result<()> {
    let env = teeny_cuda::testing::setup_cuda_env()?;
    let device = env.device;
    let target = Target::new(env.capability);

    let a_host: Vec<f32> = (0..N).map(|i| (i % 1000) as f32).collect();
    let b_host: Vec<f32> = (0..N).map(|i| (i % 777) as f32).collect();

    let mut a_buf = device.buffer::<f32>(N)?;
    let mut b_buf = device.buffer::<f32>(N)?;
    let out_buf = device.buffer::<f32>(N)?;
    a_buf.to_device(&a_host)?;
    b_buf.to_device(&b_host)?;

    // Read two arrays, write one.
    let bytes = 3.0 * N as f64 * std::mem::size_of::<f32>() as f64;

    println!("\n{N} elements, {:.0} MB moved per launch\n", bytes / 1e6);
    println!(
        "{:>6}  {:>7}  {:>10}  {:>12}  {:>9}",
        "BLOCK", "threads", "time", "bandwidth", "grid"
    );
    println!(
        "{:->6}  {:->7}  {:->10}  {:->12}  {:->9}",
        "", "", "", "", ""
    );

    for block_size in BLOCK_SIZES {
        let kernel = SweepAdd::<f32>::new(block_size);
        let ptx = std::fs::read(compile_kernel(&kernel, &target, false)?)?;
        let program = teeny_cuda::testing::load_program_from_ptx::<SweepAdd<f32>>(&ptx)?;

        // One program per BLOCK_SIZE-wide slice of the data. The *thread* count
        // is not ours to choose — teenyc records it in the PTX and the driver
        // rejects any other block dimension with "invalid argument" — so the
        // grid comes from BLOCK_SIZE and the block from the compiled metadata.
        let grid = N.div_ceil(block_size as usize);
        let cfg = teeny_cuda::testing::launch_config_with_grid(grid, &program);

        let args = (
            a_buf.as_device_ptr() as *mut f32,
            b_buf.as_device_ptr() as *mut f32,
            out_buf.as_device_ptr() as *mut f32,
            N as i32,
        );

        // `Device::launch` calls cuCtxSynchronize internally, so each iteration
        // already waits for the kernel to finish — no extra barrier needed.
        for _ in 0..WARMUP {
            device.launch(&program, &cfg, args)?;
        }

        let start = Instant::now();
        for _ in 0..ITERS {
            device.launch(&program, &cfg, args)?;
        }
        let per_launch = start.elapsed().as_secs_f64() / ITERS as f64;

        println!(
            "{:>6}  {:>7}  {:>7.1} µs  {:>9.1} GB/s  {:>9}",
            block_size,
            cfg.block[0],
            per_launch * 1e6,
            bytes / per_launch / 1e9,
            cfg.grid[0],
        );
    }

    // Correctness, once — a fast wrong kernel is not interesting.
    let mut out_host = vec![0.0f32; N];
    out_buf.to_host(&mut out_host)?;
    for i in [0, 1, N / 2, N - 1] {
        anyhow::ensure!(
            (out_host[i] - (a_host[i] + b_host[i])).abs() < 1e-5,
            "mismatch at {i}"
        );
    }
    println!("\nresults verified");
    Ok(())
}
// ANCHOR_END: sweep
