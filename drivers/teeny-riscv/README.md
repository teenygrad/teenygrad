# teeny-riscv

The RISC-V device backend for [teenygrad](https://teenygrad.org) — Target/Capability types for
the `mlir`/Triton compiler backend's `riscv64-generic` path, and a `libloading`-based runtime for
loading a compiled kernel's shared library and calling its exported symbol.

## Status

Kernels compile through the `teeny` compiler fork's RISC-V backend: their Triton bodies are
lowered to LLVM IR, compiled with LLVM's RISC-V `TargetMachine` and linked into an ELF shared
library with `ld.lld`. Built for a `riscv64` host, `RiscvDevice` loads that library and runs the
kernel natively; on any other host it runs it under `qemu-riscv64` with the `qemu` feature.

There is no chip-name-to-LLVM-cpu mapping yet: `Capability` values are passed to the compiler as
the target cpu, but every one currently gets generic RV64 codegen with the vector extension, not
chip-specific tuning.

This crate's tests compile a real kernel and check the output is a well-formed RISC-V ELF shared
object exporting its entry point (`test_compile_riscv.rs`), and, with the `qemu` feature, run it
under `qemu-riscv64` and check its results (`test_qemu_relu.rs`). `teeny-kernels`' snapshot and
execution tests (features `riscv`, plus `qemu,hardware` to execute) cover the full kernel set.

## Ahead-of-time compilation

`compiler::aot::compile_graph` lowers a traced `teeny-core` graph and compiles each op's forward
kernel into a cache directory, keyed by kernel id and capability. Options are parsed by
`compiler::options::Options::parse` from strings like `"capability=spacemit-k3,log-level=debug"`.
This is what `teeny-cli`'s `--device riscv` backend (feature `riscv`) calls.

## Prerequisites

Like `teeny-cuda`, kernel compilation shells out to the `teenyc` compiler at runtime — see
`teeny-compiler`'s docs for `TEENYC_PATH`/`cargo-teeny` setup. No additional system dependency is
required to build this crate itself (unlike `teeny-cuda`, there's no CUDA toolkit equivalent).
