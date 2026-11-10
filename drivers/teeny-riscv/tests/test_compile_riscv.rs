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

//! Compiles a real kernel through `teenyc`'s RISC-V path and checks that the output is a
//! well-formed RISC-V ELF shared library exporting the kernel's entry point.
//!
//! Running the kernel needs RISC-V (native, or under `qemu-riscv64`) -- see `test_qemu_relu.rs`
//! (feature `qemu`).

use dotenv::dotenv;
use teeny_core::device::program::Kernel;
use teeny_kernels::nn::activation::relu::ReluForward;
use teeny_riscv::compiler::compile_kernel;
use teeny_riscv::compiler::target::{Capability, Target};
use teeny_riscv::device::program::RiscvProgram;

const BLOCK_SIZE: i32 = 1024;

#[test]
fn compiles_to_a_riscv_elf_shared_library() -> anyhow::Result<()> {
    dotenv().ok();

    let kernel = ReluForward::<f32>::new(BLOCK_SIZE);
    let target = Target::new(Capability::GenericRvv1_0);

    let output_path = compile_kernel(&kernel, &target, true, false)?;
    let bytes = std::fs::read(&output_path)?;

    assert_eq!(
        &bytes[..4],
        b"\x7fELF",
        "expected a real ELF file, not assembly/PTX text"
    );
    // e_type at offset 16 (u16 LE): ET_DYN (3) for a shared object.
    assert_eq!(
        u16::from_le_bytes([bytes[16], bytes[17]]),
        3,
        "expected ET_DYN (shared object)"
    );
    // e_machine at offset 18 (u16 LE): EM_RISCV (243).
    assert_eq!(
        u16::from_le_bytes([bytes[18], bytes[19]]),
        243,
        "expected EM_RISCV"
    );

    // `launch` finds the kernel in the library by this entry point.
    let program = RiscvProgram::<ReluForward<f32>>::try_new(&output_path)?;
    assert_eq!(program.entry_point(), kernel.entry_point_name());

    Ok(())
}
