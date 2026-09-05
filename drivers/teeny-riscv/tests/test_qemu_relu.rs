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

//! Automates the manual verification recorded on `teenygrad-1zd`: compiles a real kernel through
//! `teenyc`'s RISC-V path (same as `test_compile_riscv.rs`), then actually loads and calls the
//! resulting `.so` under `qemu-riscv64`, via `teeny-test`'s `riscv::qemu` module. Gated behind
//! the `qemu` feature (needs a RISC-V cross toolchain and QEMU's user-mode emulator on the host;
//! see `teeny-test`'s `riscv` module for how those are resolved).
//!
//! `RiscvBackend` is still a stub (see `test_compile_riscv.rs`'s doc comment) -- every kernel
//! compiles to the same placeholder no-argument `void @riscv_kernel()` function regardless of
//! `kernel`'s actual body (the exported symbol is `"riscv_kernel"`, `RiscvBackend`'s fallback
//! name for a module with no name of its own -- see `kernelNameFor` in the `teeny` compiler
//! fork's `RiscvBackend.cpp`). So this only proves the compiled `.so` is loadable and callable
//! under emulation, not that it runs `kernel`'s actual logic.

#![cfg(feature = "qemu")]

use dotenv::dotenv;
use teeny_kernels::nn::activation::relu::ReluForward;
use teeny_riscv::compiler::compile_kernel;
use teeny_riscv::compiler::target::{Capability, Target};
use teeny_test::riscv::qemu::setup_qemu_env;

const BLOCK_SIZE: i32 = 1024;

#[test]
fn compiled_kernel_loads_and_runs_under_qemu() -> anyhow::Result<()> {
    dotenv().ok();

    let kernel = ReluForward::<f32>::new(BLOCK_SIZE);
    let target = Target::new(Capability::GenericRvv1_0);
    let so_path = compile_kernel(&kernel, &target, true)?;

    let qemu = setup_qemu_env()?;
    qemu.run_kernel(std::path::Path::new(&so_path), "riscv_kernel")?;

    Ok(())
}
