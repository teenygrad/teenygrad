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

#![cfg(all(feature = "cuda", feature = "hardware"))]
//! Regression tests for #133: teenyc declared scalar kernel arguments `tt.divisibility = 16`,
//! and Triton then computed anything derived from them (divisions, bounds masks, strided
//! indices) once per group of 4 contiguous lanes on CUDA.

#![allow(non_snake_case)]

use teeny_core::device::Device as _;
use teeny_core::device::buffer::Buffer;
use teeny_core::dtype::{AddOffsets, Comparison, Num, Tensor};
use teeny_cuda::compiler::{compile_kernel, target::Target};
use teeny_cuda::errors::Result;
use teeny_macros::kernel;
use teeny_test::cuda as testing;
use teeny_triton::triton::*;

/// `y[i] = i / d`, with `d` a scalar kernel argument.
#[kernel]
pub fn div_probe<T, D: Num, const BLOCK_SIZE: i32>(
    y_ptr: OutPtr<T::Pointer<D>>,
    n_elements: i32,
    d: i32,
) where
    T: Triton<I32Tensor = <T as Triton>::Tensor<i32>>,
    T::I32Tensor: Tensor<i32, 1>,
    T::I32Tensor: core::ops::Add<i32, Output = T::I32Tensor>,
    T::I32Tensor: core::ops::Div<i32, Output = T::I32Tensor>,
    T::I32Tensor: Comparison<i32, BoolTensor = T::BoolTensor>,
    T::Pointer<D>: AddOffsets<i32, 1, T::I32Tensor, Output = T::Tensor<T::Pointer<D>>>,
{
    let pid = T::program_id(Axis::X);
    let p = T::arange(0, BLOCK_SIZE) + pid * BLOCK_SIZE;
    let in_bounds = p.lt(n_elements);
    let q = p / d;
    let qf = T::cast::<i32, D>(q, None, false);
    T::store(y_ptr.add_offsets(p), qf, Some(in_bounds), &[], None, None);
}

/// Regression test for #133: teenyc declared every kernel argument divisible by 16, so Triton
/// computed `p / d` once per group of 4 lanes and the quotient was wrong unless `d` was a
/// multiple of 4. Fixed in teenygrad/teeny#39.
#[test]
fn integer_division_by_a_scalar_argument() -> Result<()> {
    dotenv::dotenv().ok();
    let env = testing::setup_cuda_env()?;
    let device = env.device;
    let kernel = DivProbe::<f32>::new(1024);
    let ptx = std::fs::read(compile_kernel(
        &kernel,
        &Target::new(env.capability),
        true,
        false,
    )?)?;
    let program = testing::load_program_from_ptx::<DivProbe<f32>>(&ptx)?;

    const N: usize = 32;
    println!(" d   first wrong lane (want i/d)");
    let mut bad_divisors = Vec::new();
    for d in 2..=13i32 {
        let out_buf = device.buffer::<f32>(N)?;
        let cfg = testing::launch_config_with_grid(1, &program);
        device.launch(&program, &cfg, (out_buf.as_device_ptr(), N as i32, d))?;
        let mut got = vec![0.0f32; N];
        out_buf.to_host(&mut got)?;

        let wrong: Vec<(usize, i32, i32)> = (0..N)
            .filter_map(|i| {
                let want = i as i32 / d;
                (got[i] as i32 != want).then_some((i, got[i] as i32, want))
            })
            .collect();
        if wrong.is_empty() {
            println!("{d:>2}   ok");
        } else {
            let (i, g, w) = wrong[0];
            println!(
                "{d:>2}   lane {i}: got {g}, want {w}   ({} of {N} wrong)",
                wrong.len()
            );
            bad_divisors.push(d);
        }
    }
    assert!(
        bad_divisors.is_empty(),
        "integer division is wrong for divisors {bad_divisors:?}"
    );
    Ok(())
}

/// `y[p] = x[p]` for `p < n_elements`; nothing at or past `n_elements` is read or written.
#[kernel]
pub fn masked_copy<T, D: Num, const BLOCK_SIZE: i32>(
    x_ptr: InPtr<T::Pointer<D>>,
    y_ptr: OutPtr<T::Pointer<D>>,
    n_elements: i32,
) where
    T: Triton,
    T::I32Tensor: Tensor<i32, 1>,
    T::I32Tensor: Comparison<i32, BoolTensor = T::BoolTensor>,
    T::Pointer<D>: AddOffsets<i32, 1, T::I32Tensor, Output = T::Tensor<T::Pointer<D>>>,
{
    let pid = T::program_id(Axis::X);
    let p = T::arange(0, BLOCK_SIZE) + pid * BLOCK_SIZE;
    let in_bounds = p.lt(n_elements);
    let x = T::load(
        x_ptr.add_offsets(p),
        Some(in_bounds),
        None,
        &[],
        None,
        None,
        None,
        false,
    );
    T::store(y_ptr.add_offsets(p), x, Some(in_bounds), &[], None, None);
}

/// The bounds mask of #133: with `n_elements` hinted divisible by 16, Triton evaluated
/// `p < n_elements` once per group of 4 lanes, so a store into the last partial group wrote up to
/// 3 elements past the end. The block must be large enough that each thread holds several
/// contiguous elements (1024 over 128 threads), or there are no groups to merge. The lengths
/// cover every remainder mod 4, in the first block and in a partial second one.
#[test]
fn masked_store_stops_at_n_elements() -> Result<()> {
    const BLOCK_SIZE: usize = 1024;
    const LEN: usize = 2 * BLOCK_SIZE;
    const GUARD: f32 = -1.0;

    dotenv::dotenv().ok();
    let env = testing::setup_cuda_env()?;
    let device = env.device;
    let kernel = MaskedCopy::<f32>::new(BLOCK_SIZE as i32);
    let ptx = std::fs::read(compile_kernel(
        &kernel,
        &Target::new(env.capability),
        true,
        false,
    )?)?;
    let program = testing::load_program_from_ptx::<MaskedCopy<f32>>(&ptx)?;

    let x_host: Vec<f32> = (0..LEN).map(|i| i as f32).collect();
    let mut x_buf = device.buffer::<f32>(LEN)?;
    x_buf.to_device(&x_host)?;

    let mut failures = Vec::new();
    for n in (1..=33usize).chain(BLOCK_SIZE - 3..=BLOCK_SIZE + 9) {
        let mut y_buf = device.buffer::<f32>(LEN)?;
        y_buf.to_device(&[GUARD; LEN])?;
        let cfg = testing::launch_config_with_grid(n.div_ceil(BLOCK_SIZE), &program);
        device.launch(
            &program,
            &cfg,
            (x_buf.as_device_ptr(), y_buf.as_device_ptr(), n as i32),
        )?;
        let mut y = vec![0.0f32; LEN];
        y_buf.to_host(&mut y)?;

        let unwritten: Vec<usize> = (0..n).filter(|&i| y[i] != x_host[i]).collect();
        let overrun: Vec<usize> = (n..LEN).filter(|&i| y[i] != GUARD).collect();
        if !unwritten.is_empty() || !overrun.is_empty() {
            failures.push(format!(
                "n_elements={n}: unwritten {unwritten:?}, written past the end {overrun:?}"
            ));
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
    Ok(())
}

/// Copies a row-major `[rows, cols]` matrix into its `[cols, rows]` transpose, recovering each
/// element's row and column from its flat index by dividing by the scalar `cols`.
#[kernel]
pub fn transpose_copy<T, D: Num, const BLOCK_SIZE: i32>(
    x_ptr: InPtr<T::Pointer<D>>,
    y_ptr: OutPtr<T::Pointer<D>>,
    rows: i32,
    cols: i32,
) where
    T: Triton<I32Tensor = <T as Triton>::Tensor<i32>>,
    T::I32Tensor: Tensor<i32, 1>,
    T::I32Tensor: core::ops::Add<i32, Output = T::I32Tensor>,
    T::I32Tensor: core::ops::Mul<i32, Output = T::I32Tensor>,
    T::I32Tensor: core::ops::Div<i32, Output = T::I32Tensor>,
    T::I32Tensor: core::ops::Add<T::I32Tensor, Output = T::I32Tensor>,
    T::I32Tensor: core::ops::Sub<T::I32Tensor, Output = T::I32Tensor>,
    T::I32Tensor: Comparison<i32, BoolTensor = T::BoolTensor>,
    T::Pointer<D>: AddOffsets<i32, 1, T::I32Tensor, Output = T::Tensor<T::Pointer<D>>>,
{
    let pid = T::program_id(Axis::X);
    let p = T::arange(0, BLOCK_SIZE) + pid * BLOCK_SIZE;
    let in_bounds = p.lt(rows * cols);
    let r = p / cols;
    let c = p - r * cols;
    let x = T::load(
        x_ptr.add_offsets(p),
        Some(in_bounds),
        None,
        &[],
        None,
        None,
        None,
        false,
    );
    T::store(
        y_ptr.add_offsets(c * rows + r),
        x,
        Some(in_bounds),
        &[],
        None,
        None,
    );
}

/// A strided copy whose innermost extent isn't a multiple of 4: before #133 was fixed,
/// `p / cols` was computed once per group of 4 lanes, so the transpose scattered elements to the
/// wrong place unless `cols` was a multiple of 4. As in `masked_store_stops_at_n_elements`, the
/// block gives each thread several contiguous elements.
#[test]
fn transpose_with_any_innermost_extent() -> Result<()> {
    const BLOCK_SIZE: usize = 1024;

    dotenv::dotenv().ok();
    let env = testing::setup_cuda_env()?;
    let device = env.device;
    let kernel = TransposeCopy::<f32>::new(BLOCK_SIZE as i32);
    let ptx = std::fs::read(compile_kernel(
        &kernel,
        &Target::new(env.capability),
        true,
        false,
    )?)?;
    let program = testing::load_program_from_ptx::<TransposeCopy<f32>>(&ptx)?;

    let mut failures = Vec::new();
    for rows in [3usize, 5] {
        for cols in 2..=13usize {
            let n = rows * cols;
            let x_host: Vec<f32> = (0..n).map(|i| i as f32).collect();
            let mut x_buf = device.buffer::<f32>(n)?;
            x_buf.to_device(&x_host)?;
            let mut y_buf = device.buffer::<f32>(n)?;
            y_buf.to_device(&vec![-1.0f32; n])?;
            let cfg = testing::launch_config_with_grid(n.div_ceil(BLOCK_SIZE), &program);
            device.launch(
                &program,
                &cfg,
                (
                    x_buf.as_device_ptr(),
                    y_buf.as_device_ptr(),
                    rows as i32,
                    cols as i32,
                ),
            )?;
            let mut y = vec![0.0f32; n];
            y_buf.to_host(&mut y)?;

            let want: Vec<f32> = (0..cols)
                .flat_map(|c| (0..rows).map(move |r| (r * cols + c) as f32))
                .collect();
            if y != want {
                failures.push(format!("[{rows}, {cols}]: got {y:?}, want {want:?}"));
            }
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
    Ok(())
}
