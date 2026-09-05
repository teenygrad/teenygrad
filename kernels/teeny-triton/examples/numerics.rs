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

//! Chapter 19 of the kernels book: what dtype choice costs, and what it loses.
//!
//! Three questions, measured:
//!   1. What does `f64` cost against `f32` on a memory-bound kernel?
//!   2. Is a GPU reduction reproducible when the block size changes?
//!   3. How far does an `f32` accumulator drift from an exact answer?
//!
//! ```bash
//! cargo run --release -p teeny-triton --features cuda --example numerics
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

// ANCHOR: kernels
/// Elementwise add, generic over the float type. `D` is both the storage and
/// the arithmetic dtype, so instantiating it twice measures what the width
/// costs end to end.
#[kernel]
pub fn add_typed<T: Triton, D: Float, const BLOCK: i32>(
    a_ptr: In<T::Pointer<D>>,
    b_ptr: In<T::Pointer<D>>,
    out_ptr: Out<T::Pointer<D>>,
    n: i32,
) where
    T::I32Tensor: types::Tensor<i32, 1>,
    T::I32Tensor: Comparison<i32, BoolTensor = T::BoolTensor>,
    T::Pointer<D>: AddOffsets<i32, 1, T::I32Tensor, Output = T::Tensor<T::Pointer<D>>>,
{
    let pid = T::program_id(Axis::X);
    let offsets = T::arange(0, BLOCK) + pid * BLOCK;
    let m = offsets.lt(n);

    let a = T::load(
        a_ptr.add_offsets(offsets),
        Some(m),
        None,
        &[],
        None,
        None,
        None,
        false,
    );
    let b = T::load(
        b_ptr.add_offsets(offsets),
        Some(m),
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
        Some(m),
        &[],
        None,
        None,
    );
}

/// Partial sum of one `BLOCK`-wide slice. The reduction tree's shape depends on
/// `BLOCK`, which is what makes the result order-dependent.
#[kernel]
pub fn partial_sum<T: Triton, D: Float, const BLOCK: i32>(
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
// ANCHOR_END: kernels

const N: usize = 16 * 1024 * 1024;
const WARMUP: usize = 5;
const ITERS: usize = 30;

fn main() -> Result<()> {
    let env = teeny_cuda::testing::setup_cuda_env()?;
    let device = env.device;
    let target = Target::new(env.capability);

    throughput(&device, &target)?;
    reproducibility(&device, &target)?;
    accumulator(&device, &target)?;
    Ok(())
}

// ANCHOR: throughput
/// f32 against f64 on a kernel that is entirely memory-bound. `f64` moves twice
/// the bytes per element, so on bandwidth alone it should cost about 2x.
fn throughput(device: &teeny_cuda::device::CudaDevice<'_>, target: &Target) -> Result<()> {
    println!("\n── 1. what f64 costs ──────────────────────────────────\n");
    println!(
        "{:>6}  {:>10}  {:>12}  {:>10}",
        "dtype", "time", "bandwidth", "moved"
    );
    println!("{:->6}  {:->10}  {:->12}  {:->10}", "", "", "", "");

    macro_rules! bench {
        ($ty:ty, $ktype:ty, $label:literal) => {{
            let host: Vec<$ty> = (0..N).map(|i| (i % 1000) as $ty).collect();
            let mut a = device.buffer::<$ty>(N)?;
            let mut b = device.buffer::<$ty>(N)?;
            let out = device.buffer::<$ty>(N)?;
            a.to_device(&host)?;
            b.to_device(&host)?;

            let k = <$ktype>::new(256);
            let ptx = std::fs::read(compile_kernel(&k, target, false, false)?)?;
            let prog = teeny_cuda::testing::load_program_from_ptx::<$ktype>(&ptx)?;
            let cfg = teeny_cuda::testing::launch_config_with_grid(N.div_ceil(256), &prog);
            let args = (
                a.as_device_ptr() as *mut $ty,
                b.as_device_ptr() as *mut $ty,
                out.as_device_ptr() as *mut $ty,
                N as i32,
            );

            for _ in 0..WARMUP {
                device.launch(&prog, &cfg, args)?;
            }
            let start = Instant::now();
            for _ in 0..ITERS {
                device.launch(&prog, &cfg, args)?;
            }
            let secs = start.elapsed().as_secs_f64() / ITERS as f64;
            let bytes = 3.0 * N as f64 * std::mem::size_of::<$ty>() as f64;
            println!(
                "{:>6}  {:>7.1} µs  {:>9.1} GB/s  {:>7.0} MB",
                $label,
                secs * 1e6,
                bytes / secs / 1e9,
                bytes / 1e6
            );
            secs
        }};
    }

    let t32 = bench!(f32, AddTyped<f32>, "f32");
    let t64 = bench!(f64, AddTyped<f64>, "f64");
    println!("\nf64 costs {:.2}x the time of f32", t64 / t32);
    Ok(())
}
// ANCHOR_END: throughput

// ANCHOR: reproducibility
/// The same data, reduced with two different block sizes. Floating-point
/// addition is not associative, so a different tree shape can give a different
/// answer — this checks whether it actually does.
fn reproducibility(device: &teeny_cuda::device::CudaDevice<'_>, target: &Target) -> Result<()> {
    println!("\n── 2. is a reduction reproducible? ────────────────────\n");

    // 0.1 has no exact binary representation, so every addition rounds.
    let host: Vec<f32> = (0..N).map(|i| 0.1 + (i % 7) as f32 * 0.01).collect();
    let mut input = device.buffer::<f32>(N)?;
    input.to_device(&host)?;

    let mut totals = Vec::new();
    for block in [256i32, 1024i32] {
        let grid = N / block as usize;
        let out = device.buffer::<f32>(grid)?;

        let k = PartialSum::<f32>::new(block);
        let ptx = std::fs::read(compile_kernel(&k, target, false, false)?)?;
        let prog = teeny_cuda::testing::load_program_from_ptx::<PartialSum<f32>>(&ptx)?;
        let cfg = teeny_cuda::testing::launch_config_with_grid(grid, &prog);
        device.launch(
            &prog,
            &cfg,
            (
                input.as_device_ptr() as *mut f32,
                out.as_device_ptr() as *mut f32,
                N as i32,
            ),
        )?;

        // Finish the reduction on the host, in f64, so the only f32 arithmetic
        // being compared is what happened inside the blocks.
        let mut partials = vec![0.0f32; grid];
        out.to_host(&mut partials)?;
        let total: f64 = partials.iter().map(|&v| v as f64).sum();
        println!("  BLOCK={block:<5} → {total:.6}");
        totals.push(total);
    }

    if totals[0] == totals[1] {
        println!("\n  identical");
    } else {
        let rel = (totals[0] - totals[1]).abs() / totals[1].abs();
        println!(
            "\n  differ by {:.3e} absolute, {rel:.3e} relative",
            (totals[0] - totals[1]).abs()
        );
        println!("  same data, same kernel, different tree shape");
    }
    Ok(())
}
// ANCHOR_END: reproducibility

// ANCHOR: accumulator
/// How far the f32 result drifts from an exact answer, and whether accumulating
/// the partials in f64 recovers it.
fn accumulator(device: &teeny_cuda::device::CudaDevice<'_>, target: &Target) -> Result<()> {
    println!("\n── 3. accumulator width ───────────────────────────────\n");

    let host: Vec<f32> = (0..N).map(|i| 0.1 + (i % 7) as f32 * 0.01).collect();
    // Reference: sum the exact f32 values in f64. Each element is exactly
    // representable as f64, so this is the right answer to within f64's error.
    let exact: f64 = host.iter().map(|&v| v as f64).sum();

    let mut input = device.buffer::<f32>(N)?;
    input.to_device(&host)?;

    let block = 1024i32;
    let grid = N / block as usize;
    let out = device.buffer::<f32>(grid)?;

    let k = PartialSum::<f32>::new(block);
    let ptx = std::fs::read(compile_kernel(&k, target, false, false)?)?;
    let prog = teeny_cuda::testing::load_program_from_ptx::<PartialSum<f32>>(&ptx)?;
    let cfg = teeny_cuda::testing::launch_config_with_grid(grid, &prog);
    device.launch(
        &prog,
        &cfg,
        (
            input.as_device_ptr() as *mut f32,
            out.as_device_ptr() as *mut f32,
            N as i32,
        ),
    )?;

    let mut partials = vec![0.0f32; grid];
    out.to_host(&mut partials)?;

    // Same partials, two accumulator widths.
    let in_f32: f32 = partials.iter().sum();
    let in_f64: f64 = partials.iter().map(|&v| v as f64).sum();

    // And the naive thing: accumulate the raw data sequentially in f32.
    let naive: f32 = host.iter().sum();

    println!("  exact (f64 reference)      {exact:.4}");
    println!(
        "  GPU blocks + f64 accum     {in_f64:.4}   rel err {:.2e}",
        (in_f64 - exact).abs() / exact
    );
    println!(
        "  GPU blocks + f32 accum     {in_f32:.4}   rel err {:.2e}",
        (in_f32 as f64 - exact).abs() / exact
    );
    println!(
        "  sequential f32 on the CPU  {naive:.4}   rel err {:.2e}",
        (naive as f64 - exact).abs() / exact
    );
    Ok(())
}
// ANCHOR_END: accumulator
