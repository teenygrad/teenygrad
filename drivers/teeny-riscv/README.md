# teeny-riscv

The RISC-V device backend for [teenygrad](https://teenygrad.org) — Target/Capability types for
the `mlir`/Triton compiler backend's `riscv64-generic` path, and a `libloading`-based runtime for
loading a compiled kernel's shared library and calling its exported symbol.

## Status

Early scaffold. The underlying `teeny` compiler fork's RISC-V codegen (`RiscvBackend`) is still a
stub: every kernel currently compiles to the same placeholder no-argument `void @<name>()`
function regardless of source, through a real LLVM RISC-V `TargetMachine` and linked into a real
ELF shared library via `ld.lld`. There is no chip-name-to-LLVM-cpu/feature mapping yet (this
crate's `Capability` values are accepted by the compiler invocation but not yet honored by
codegen), and no real MIR/Triton-to-LLVM-IR lowering, so real per-kernel argument passing isn't
possible yet.

This crate's own tests only exercise that pipeline against the current placeholder kernel:
compiling it via `LlvmCompiler` with the `riscv64-generic` target, and verifying the output is a
well-formed RISC-V ELF shared object. Actually `dlopen`/calling the compiled kernel (via
[`runtime::KernelLibrary`]) requires running on real RISC-V hardware or under RISC-V user-mode
emulation (e.g. `qemu-riscv64`) — this was verified manually during development but isn't part of
the automated test suite here, since that requires cross-compiling and running the test binary
itself for `riscv64gc-unknown-linux-gnu`, not just the kernel.

## Prerequisites

Like `teeny-cuda`, kernel compilation shells out to the `teenyc` compiler at runtime — see
`teeny-compiler`'s docs for `TEENYC_PATH`/`cargo-teeny` setup. No additional system dependency is
required to build this crate itself (unlike `teeny-cuda`, there's no CUDA toolkit equivalent).
