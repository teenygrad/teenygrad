# teeny-test

Test support for crates built on [teenygrad](https://teenygrad.org): helpers to load test
fixtures, read compiled kernel output for snapshot tests, and set up a CUDA device or a RISC-V
QEMU environment to run kernels against. The same helpers work for every driver backend, so a
kernel crate's tests aren't tied to one device.

Add it as a dev-dependency, enabling the backends your tests use:

```toml
[dev-dependencies]
teeny-test = { version = "0.2", features = ["cuda"] }
```

## What's included

Always available:

- `load_fixture` / `load_fixture_i32`: read a little-endian `f32`/`i32` fixture from your crate's
  `tests/fixtures/` (pass `env!("CARGO_MANIFEST_DIR")`).
- `read_compiled_asm`: read a compiled kernel's PTX or assembly as text, for `insta` snapshots.
- `teenyc_cache_dir`: the kernel cache directory (`$TEENYC_CACHE_DIR`, default
  `/tmp/teenyc_cache`).
- `ExecKernel`: adapts a lowered `ExecutableOp` to `Kernel`, so any backend's `compile_kernel`
  can compile it.

With features:

| Feature | Module | Provides |
|---|---|---|
| `cuda` | `cuda` | `setup_cuda_env` (opens the first device; `TEENYC_CAPABILITY` overrides its compute capability), launch-config helpers, loading a program from PTX |
| `riscv` + `qemu` | `riscv`, `riscv::qemu` | `setup_qemu_env`, which finds `qemu-riscv64` and a `riscv64-linux-gnu` cross C compiler (`TEENYC_QEMU_RISCV64` / `TEENYC_RISCV_CC` override them), and runs a compiled kernel under user-mode emulation |

`riscv` on its own only brings in `teeny-riscv`. Compiling a kernel for RISC-V works on any host,
and only running one needs `qemu`.

## Prerequisites

Compiling kernels shells out to the `teenyc` compiler at runtime. See `teeny-compiler`'s docs for
`TEENYC_PATH` / `cargo-teeny` setup. The `cuda` feature builds `teeny-cuda`, which needs the CUDA
toolkit. The `qemu` helpers need `qemu-riscv64` and a `riscv64-linux-gnu` cross C compiler on the
host when a test runs a kernel, but nothing extra to build.

## License

Apache-2.0
