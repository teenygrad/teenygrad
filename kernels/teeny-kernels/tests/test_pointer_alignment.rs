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
//! teenyc declares every kernel pointer argument `tt.divisibility = 16` (teenyc-24b), so Triton
//! may use 16-byte vector loads and stores through it. A pointer offset into its buffer, as a
//! sub-buffer view would be, breaks that promise: launched, the kernel faults with "misaligned
//! address" and poisons the CUDA context. Launches must refuse such pointers up front instead.

#![allow(non_snake_case)]

use teeny_core::device::Device as _;
use teeny_core::device::buffer::Buffer;
use teeny_core::dtype::{AddOffsets, Num, Tensor};
use teeny_cuda::compiler::{compile_kernel, target::Target};
use teeny_cuda::errors::Result;
use teeny_macros::kernel;
use teeny_test::cuda as testing;
use teeny_triton::triton::*;

/// `y[p] = x[p]` over whole blocks, unmasked (the grid covers the buffers exactly), so Triton
/// vectorises both accesses on the strength of the pointers' alignment hint.
#[kernel]
pub fn unmasked_copy<T, D: Num, const BLOCK_SIZE: i32>(
    x_ptr: InPtr<T::Pointer<D>>,
    y_ptr: OutPtr<T::Pointer<D>>,
) where
    T: Triton,
    T::I32Tensor: Tensor<i32, 1>,
    T::Pointer<D>: AddOffsets<i32, 1, T::I32Tensor, Output = T::Tensor<T::Pointer<D>>>,
{
    let pid = T::program_id(Axis::X);
    let p = T::arange(0, BLOCK_SIZE) + pid * BLOCK_SIZE;
    let x = T::load(
        x_ptr.add_offsets(p),
        None,
        None,
        &[],
        None,
        None,
        None,
        false,
    );
    T::store(y_ptr.add_offsets(p), x, None, &[], None, None);
}

#[test]
fn launch_refuses_pointers_not_16_byte_aligned() -> Result<()> {
    const BLOCK_SIZE: usize = 1024;
    const N: usize = 2 * BLOCK_SIZE;
    const OFFSET: usize = 1;

    dotenv::dotenv().ok();
    let env = testing::setup_cuda_env()?;
    let device = env.device;
    let kernel = UnmaskedCopy::<f32>::new(BLOCK_SIZE as i32);
    let ptx = std::fs::read(compile_kernel(
        &kernel,
        &Target::new(env.capability),
        true,
        false,
    )?)?;
    let program = testing::load_program_from_ptx::<UnmaskedCopy<f32>>(&ptx)?;
    let cfg = testing::launch_config_with_grid(N / BLOCK_SIZE, &program);

    let x_host: Vec<f32> = (0..N + OFFSET).map(|i| i as f32).collect();
    let mut x_buf = device.buffer::<f32>(N + OFFSET)?;
    x_buf.to_device(&x_host)?;
    let mut y_buf = device.buffer::<f32>(N + OFFSET)?;
    y_buf.to_device(&vec![-1.0f32; N + OFFSET])?;

    // One f32 into each buffer: 4-byte aligned, not 16.
    // SAFETY: the copy would cover [OFFSET, OFFSET + N), inside both allocations.
    let x = unsafe { x_buf.as_device_ptr().add(OFFSET) };
    let y = unsafe { y_buf.as_device_ptr().add(OFFSET) };
    let err = device
        .launch(&program, &cfg, (x, y))
        .expect_err("a launch through pointers that aren't 16-byte aligned must be refused");
    let msg = err.to_string();
    assert!(
        msg.contains("16-byte aligned") && msg.contains("argument 0"),
        "error should name the misaligned argument: {msg}"
    );

    // Refused before launching, so the context is intact: the same kernel still runs through the
    // buffers' own (aligned) pointers.
    device.launch(
        &program,
        &cfg,
        (x_buf.as_device_ptr(), y_buf.as_device_ptr()),
    )?;
    let mut y_host = vec![0.0f32; N + OFFSET];
    y_buf.to_host(&mut y_host)?;
    let mut want = x_host[..N].to_vec();
    want.push(-1.0);
    assert_eq!(y_host, want, "aligned launch after the refused one");
    Ok(())
}
