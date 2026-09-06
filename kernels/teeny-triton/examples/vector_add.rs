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

//! Chapter 5 of the kernels book: a vector add, end to end.
//!
//! Needs the `cuda` feature and a GPU of compute capability sm_75 or newer:
//!
//! ```bash
//! cargo run -p teeny-triton --features cuda --example vector_add
//! ```
//!
//! The book includes the `kernel` anchor below, so the lines a reader sees are
//! the lines that ran.

#![allow(non_snake_case)]

use anyhow::Result;
use teeny_core::device::buffer::Buffer;
use teeny_core::device::{Device, program::Kernel};
use teeny_core::dtype::Num;
use teeny_cuda::compiler::{compile_kernel, target::Target};
use teeny_macros::kernel;
use teeny_triton::triton::{
    types::{AddOffsets, Comparison},
    *,
};

// ANCHOR: kernel
/// Adds two vectors: `out[i] = a[i] + b[i]`.
#[kernel]
pub fn vector_add<T: Triton, D: Num, const BLOCK_SIZE: i32>(
    a_ptr: InPtr<T::Pointer<D>>,
    b_ptr: InPtr<T::Pointer<D>>,
    out_ptr: OutPtr<T::Pointer<D>>,
    n_elements: i32,
) where
    T::I32Tensor: types::Tensor<i32, 1>,
    T::I32Tensor: Comparison<i32, BoolTensor = T::BoolTensor>,
    T::Pointer<D>: AddOffsets<i32, 1, T::I32Tensor, Output = T::Tensor<T::Pointer<D>>>,
{
    // ANCHOR: indices
    // Which slice of the vector is this program responsible for?
    let pid = T::program_id(Axis::X);
    let block_start = pid * BLOCK_SIZE;

    // The indices it will touch: block_start, block_start + 1, and so on.
    let offsets = T::arange(0, BLOCK_SIZE) + block_start;
    // ANCHOR_END: indices

    // ANCHOR: mask
    // The last program runs off the end of the vector. This says which of its
    // lanes are real.
    let in_bounds = offsets.lt(n_elements);
    // ANCHOR_END: mask

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

// ANCHOR: constants
/// Threads per program. A power of two, and a multiple of a warp's 32 lanes.
const BLOCK_SIZE: i32 = 128;

/// Deliberately not a multiple of `BLOCK_SIZE`, so the mask has something to
/// do: 1000 elements is seven full programs and a ragged eighth.
const N: usize = 1000;
// ANCHOR_END: constants

// ANCHOR: run
fn main() -> Result<()> {
    // a[i] = i, b[i] = 2i — inputs whose correct sum is obvious by eye.
    let a_host: Vec<f32> = (0..N).map(|i| i as f32).collect();
    let b_host: Vec<f32> = (0..N).map(|i| (i * 2) as f32).collect();

    // Open the first GPU and ask what it is, so the kernel is compiled for the
    // card that will run it rather than for a guess.
    let env = teeny_test::cuda::setup_cuda_env()?;
    let device = env.device;

    // Build the kernel for this block size, then compile it to PTX.
    let kernel = VectorAdd::<f32>::new(BLOCK_SIZE);
    let ptx_path = compile_kernel(&kernel, &Target::new(env.capability), false, false)?;
    println!("compiled {} → {ptx_path}", kernel.name());

    // Device memory: two inputs to fill, one output to read back.
    let mut a_buf = device.buffer::<f32>(N)?;
    let mut b_buf = device.buffer::<f32>(N)?;
    let out_buf = device.buffer::<f32>(N)?;
    a_buf.to_device(&a_host)?;
    b_buf.to_device(&b_host)?;

    let ptx = std::fs::read(&ptx_path)?;
    let program = teeny_test::cuda::load_program_from_ptx::<VectorAdd<f32>>(&ptx)?;

    // One program per BLOCK_SIZE-wide slice, rounding up so the ragged tail
    // still gets one. The threads-per-program is not ours to pick: teenyc
    // records it in the PTX, and the driver rejects any other block dimension.
    let grid = N.div_ceil(BLOCK_SIZE as usize);
    let cfg = teeny_test::cuda::launch_config_with_grid(grid, &program);
    println!(
        "launching {} programs of {} threads",
        cfg.grid[0], cfg.block[0]
    );

    device.launch(
        &program,
        &cfg,
        (
            a_buf.as_device_ptr() as *mut f32,
            b_buf.as_device_ptr() as *mut f32,
            out_buf.as_device_ptr() as *mut f32,
            N as i32,
        ),
    )?;

    let mut out_host = vec![0.0f32; N];
    out_buf.to_host(&mut out_host)?;

    for i in 0..N {
        let expected = a_host[i] + b_host[i];
        anyhow::ensure!(
            (out_host[i] - expected).abs() < 1e-5,
            "mismatch at {i}: gpu={} expected={expected}",
            out_host[i],
        );
    }

    println!("out[0]   = {}", out_host[0]);
    println!("out[1]   = {}", out_host[1]);
    println!("out[999] = {}", out_host[N - 1]);
    println!("all {N} elements match the CPU result");
    Ok(())
}
// ANCHOR_END: run
