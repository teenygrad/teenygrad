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

//! Chapter 17 of the kernels book: what access order costs.
//!
//! Two kernels that read exactly the same 64 MB and do exactly the same
//! arithmetic. One reads along rows, the other down columns. Only the order
//! differs.
//!
//! ```bash
//! cargo run --release -p teeny-triton --features cuda --example coalescing
//! ```

#![allow(non_snake_case)]

use std::time::Instant;

use anyhow::Result;
use teeny_core::device::Device;
use teeny_core::device::buffer::Buffer;
use teeny_core::dtype::Float;
use teeny_cuda::compiler::{compile_kernel, target::Target};
use teeny_macros::kernel;
use teeny_triton::triton::{
    types::{AddOffsets, Comparison},
    *,
};

// ANCHOR: contiguous
/// Sums a `BLOCK`-wide **contiguous** run. Consecutive lanes read consecutive
/// addresses, which is the pattern the hardware is built for.
#[kernel]
pub fn sum_rows<T: Triton, D: Float, const BLOCK: i32>(
    in_ptr: In<T::Pointer<D>>,
    out_ptr: Out<T::Pointer<D>>,
    _n: i32,
) where
    T::I32Tensor: types::Tensor<i32, 1>,
    T::I32Tensor: Comparison<i32, BoolTensor = T::BoolTensor>,
    T::Pointer<D>: AddOffsets<i32, 1, T::I32Tensor, Output = T::Tensor<T::Pointer<D>>>,
{
    let pid = T::program_id(Axis::X);
    let offsets = T::arange(0, BLOCK) + pid * BLOCK;

    let x = T::load(
        in_ptr.add_offsets(offsets),
        None,
        None,
        &[],
        None,
        None,
        None,
        false,
    );
    let s = T::sum(x, Some(0), true);

    let o = T::arange(0, 1) + pid;
    T::store(out_ptr.add_offsets(o), s, None, &[], None, None);
}
// ANCHOR_END: contiguous

// ANCHOR: strided
/// Sums `BLOCK` elements of one **column** of a row-major `[_, COLS]` matrix.
/// Consecutive lanes are `COLS` elements apart, so each one lands in a
/// different cache line.
#[kernel]
pub fn sum_cols<T: Triton, D: Float, const BLOCK: i32, const COLS: i32>(
    in_ptr: In<T::Pointer<D>>,
    out_ptr: Out<T::Pointer<D>>,
    _n: i32,
) where
    T::I32Tensor: types::Tensor<i32, 1>,
    T::I32Tensor: Comparison<i32, BoolTensor = T::BoolTensor>,
    T::Pointer<D>: AddOffsets<i32, 1, T::I32Tensor, Output = T::Tensor<T::Pointer<D>>>,
{
    let pid = T::program_id(Axis::X);
    let col = pid % COLS;
    let row_block = pid / COLS;

    // The only line that differs from sum_rows: a stride of COLS, not 1.
    let rows = T::arange(0, BLOCK) + row_block * BLOCK;
    let offsets = rows * COLS + col;

    let x = T::load(
        in_ptr.add_offsets(offsets),
        None,
        None,
        &[],
        None,
        None,
        None,
        false,
    );
    let s = T::sum(x, Some(0), true);

    let o = T::arange(0, 1) + pid;
    T::store(out_ptr.add_offsets(o), s, None, &[], None, None);
}
// ANCHOR_END: strided

// ANCHOR: harness
const ROWS: i32 = 4096;
const COLS: i32 = 4096;
const BLOCK: i32 = 256;

/// 16M elements = 64 MB, comfortably past any cache, so this measures memory.
const N: usize = (ROWS * COLS) as usize;
/// Both kernels launch the same number of programs and read the same elements.
const GRID: usize = N / BLOCK as usize;

const WARMUP: usize = 5;
const ITERS: usize = 30;

fn main() -> Result<()> {
    let env = teeny_cuda::testing::setup_cuda_env()?;
    let device = env.device;
    let target = Target::new(env.capability);

    let host: Vec<f32> = (0..N).map(|i| (i % 251) as f32).collect();
    let mut in_buf = device.buffer::<f32>(N)?;
    let out_buf = device.buffer::<f32>(GRID)?;
    in_buf.to_device(&host)?;

    let bytes = N as f64 * std::mem::size_of::<f32>() as f64;
    println!(
        "\n{ROWS}x{COLS} f32 = {:.0} MB read per launch",
        bytes / 1e6
    );
    println!("{GRID} programs, {BLOCK} elements each, both kernels\n");
    println!("{:>12}  {:>10}  {:>12}", "access", "time", "bandwidth");
    println!("{:->12}  {:->10}  {:->12}", "", "", "");

    let args = (
        in_buf.as_device_ptr() as *mut f32,
        out_buf.as_device_ptr() as *mut f32,
        N as i32,
    );

    // Warm up, then time. A macro rather than a function because the two
    // kernels are different types and `Kernel::Args` is a generic associated
    // type tied to the program's lifetime — a shared `fn` needs higher-ranked
    // bounds that obscure what is really a six-line loop.
    macro_rules! timed {
        ($kernel:expr, $ty:ty, $label:literal) => {{
            let ptx = std::fs::read(compile_kernel(&$kernel, &target, false, false)?)?;
            let prog = teeny_cuda::testing::load_program_from_ptx::<$ty>(&ptx)?;
            let cfg = teeny_cuda::testing::launch_config_with_grid(GRID, &prog);

            for _ in 0..WARMUP {
                device.launch(&prog, &cfg, args)?;
            }
            let start = Instant::now();
            for _ in 0..ITERS {
                device.launch(&prog, &cfg, args)?;
            }
            let secs = start.elapsed().as_secs_f64() / ITERS as f64;

            println!(
                "{:>12}  {:>7.1} µs  {:>9.1} GB/s",
                $label,
                secs * 1e6,
                bytes / secs / 1e9
            );
            secs
        }};
    }

    let contiguous = timed!(SumRows::<f32>::new(BLOCK), SumRows<f32>, "row-major");
    let strided = timed!(SumCols::<f32>::new(BLOCK, COLS), SumCols<f32>, "column");

    println!("\ncolumn access is {:.1}x slower", strided / contiguous);
    Ok(())
}
// ANCHOR_END: harness
