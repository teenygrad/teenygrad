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

//! Compiles `ReluForward` through `teenyc`'s RISC-V path (same as `test_compile_riscv.rs`), runs
//! the resulting `.so` under `qemu-riscv64` via `teeny-test`'s `riscv::qemu` module, and checks
//! its output against relu computed on the host. Gated behind the `qemu` feature (needs a RISC-V
//! cross toolchain and QEMU's user-mode emulator on the host; see `teeny-test`'s `riscv` module
//! for how those are resolved).

#![cfg(feature = "qemu")]

use std::path::Path;

use dotenv::dotenv;
use teeny_core::device::Device;
use teeny_core::device::buffer::Buffer;
use teeny_core::device::program::Kernel;
use teeny_kernels::nn::activation::relu::ReluForward;
use teeny_riscv::compiler::compile_kernel;
use teeny_riscv::compiler::target::{Capability, Target};
use teeny_riscv::device::context::RiscvDeviceInfo;
use teeny_riscv::device::program::RiscvProgram;
use teeny_riscv::device::{RiscvDevice, RiscvLaunchConfig};
use teeny_test::riscv::qemu::setup_qemu_env;

const BLOCK_SIZE: i32 = 1024;

/// Two full blocks and part of a third, so the run exercises both the program id's block offset
/// and the mask on the last block.
const N_ELEMENTS: usize = 2 * BLOCK_SIZE as usize + 500;

/// Pre-fills the output past `N_ELEMENTS`; the kernel must leave it untouched.
const SENTINEL: f32 = 12345.0;

#[test]
fn compiled_relu_kernel_runs_correctly_under_qemu() -> anyhow::Result<()> {
    dotenv().ok();

    let kernel = ReluForward::<f32>::new(BLOCK_SIZE);
    let target = Target::new(Capability::GenericRvv1_0);
    let so_path = compile_kernel(&kernel, &target, true, false)?;

    let block_size = BLOCK_SIZE as usize;
    let len = N_ELEMENTS.div_ceil(block_size) * block_size;
    // Negative and positive inputs; the padding is negative, so a store that ignored the mask
    // would replace the sentinel with 0.
    let x: Vec<f32> = (0..len)
        .map(|i| {
            if i < N_ELEMENTS {
                (i % 17) as f32 - 8.25
            } else {
                -1.0
            }
        })
        .collect();
    let mut y = vec![SENTINEL; len];

    let qemu = setup_qemu_env()?;
    qemu.run_pointwise_kernel(
        Path::new(&so_path),
        &kernel.entry_point_name(),
        block_size,
        N_ELEMENTS,
        &x,
        &mut y,
    )?;

    for (i, (&xi, &yi)) in x.iter().zip(&y).take(N_ELEMENTS).enumerate() {
        assert_eq!(yi, xi.max(0.0), "y[{i}] for x[{i}] = {xi}");
    }
    assert!(
        y[N_ELEMENTS..].iter().all(|&v| v == SENTINEL),
        "the kernel wrote past n_elements"
    );

    Ok(())
}

#[test]
fn device_launch_runs_relu_under_qemu() -> anyhow::Result<()> {
    dotenv().ok();

    let kernel = ReluForward::<f32>::new(BLOCK_SIZE);
    let target = Target::new(Capability::GenericRvv1_0);
    let so_path = compile_kernel(&kernel, &target, true, false)?;

    let device = RiscvDevice::new(RiscvDeviceInfo::default());
    let x: Vec<f32> = (0..N_ELEMENTS).map(|i| (i % 17) as f32 - 8.25).collect();
    let mut x_buf = device.buffer::<f32>(N_ELEMENTS)?;
    let y_buf = device.buffer::<f32>(N_ELEMENTS)?;
    x_buf.to_device(&x)?;

    let program = RiscvProgram::<ReluForward<f32>>::try_new(&so_path)?;
    let blocks = N_ELEMENTS.div_ceil(BLOCK_SIZE as usize) as u32;
    device.launch(
        &program,
        &RiscvLaunchConfig::new([blocks, 1, 1]),
        (
            x_buf.as_device_ptr(),
            y_buf.as_device_ptr(),
            N_ELEMENTS as i32,
        ),
    )?;

    let mut y = vec![0.0f32; N_ELEMENTS];
    y_buf.to_host(&mut y)?;
    for (i, (&xi, &yi)) in x.iter().zip(&y).enumerate() {
        assert_eq!(yi, xi.max(0.0), "y[{i}] for x[{i}] = {xi}");
    }

    Ok(())
}
