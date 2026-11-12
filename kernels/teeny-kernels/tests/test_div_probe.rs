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
//! Minimal repro: integer division of a tensor by a kernel scalar on CUDA.

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
