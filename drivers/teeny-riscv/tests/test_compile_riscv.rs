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

//! Compiles a real kernel through `teenyc`'s RISC-V path and checks that the output is a
//! well-formed RISC-V ELF shared library.
//!
//! `RiscvBackend` (in the `teeny` compiler fork) is still a stub: every kernel compiles to the
//! same placeholder no-argument `void @<name>()` function regardless of source (see the crate
//! README), so this deliberately doesn't check the kernel's *behavior* -- there isn't one yet --
//! only that the compile pipeline (real LLVM RISC-V codegen, linked via `ld.lld`) produces a
//! genuine RISC-V shared object.
//!
//! Actually loading and calling it (via [`teeny_riscv::runtime::KernelLibrary`]) requires running
//! on RISC-V (native, or under `qemu-riscv64`) -- see `test_qemu_relu.rs` (feature `qemu`) for
//! that, via `teeny-test`'s `riscv::qemu` module.

use dotenv::dotenv;
use teeny_kernels::nn::activation::relu::ReluForward;
use teeny_riscv::compiler::compile_kernel;
use teeny_riscv::compiler::target::{Capability, Target};

const BLOCK_SIZE: i32 = 1024;

#[test]
fn compiles_to_a_riscv_elf_shared_library() -> anyhow::Result<()> {
    dotenv().ok();

    let kernel = ReluForward::<f32>::new(BLOCK_SIZE);
    let target = Target::new(Capability::GenericRvv1_0);

    let output_path = compile_kernel(&kernel, &target, true)?;
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

    Ok(())
}
