# teeny-compiler

Compiles a [teenygrad](https://teenygrad.org) computational graph (FXGraph, traced from
`teeny-core`) down to a target backend — LLVM/MLIR object code today, with `ndarray`/CPU support
behind a feature flag.

## Prerequisites

- **Rust**: any stable or nightly toolchain to `cargo build`/`cargo doc` this crate itself — no
  system-library dependency at build time.
- **Compiling kernels at runtime** (the LLVM backend, `compiler::backend::llvm`) shells out to a
  custom compiler binary, `teenyc` — a fork of `rustc` with an MLIR codegen backend
  (`-Zcodegen-backend=mlir`), distributed on the "stable" channel. It is **not** needed to build
  `teeny-compiler` itself, only to actually compile/run kernels through it:
  ```bash
  export TEENYC_PATH=/path/to/teenyc   # falls back to `teenyc` on $PATH if unset
  ```
  The supported way to obtain `teenyc` is via
  [`cargo-teeny`](https://github.com/spinorml/cargo-teeny), which installs a prebuilt release:
  ```bash
  cargo install --git https://github.com/spinorml/cargo-teeny
  cargo teeny install-toolchain
  ```
  This mirrors the toolchain setup used by downstream projects such as
  [`vision-rs`](https://github.com/spinorml/vision-rs) — see its README for the full
  `cargo-teeny`/`TEENYC_PATH` walkthrough, including GPU compute-capability options
  (`-Ctarget-cpu`) and PTX ISA version overrides (`TEENYC_PTX_VERSION`).

## Features

| Feature   | Default | Description                          |
|-----------|---------|---------------------------------------|
| `ndarray` | ✅      | Enables the `ndarray`-backed CPU path. |

## Getting started

```toml
[dependencies]
teeny-compiler = "0"
```

```rust
teeny_compiler::init_logging();
```

## License

Apache-2.0. See [LICENSE-APACHE](https://github.com/teenygrad/teenygrad/blob/main/LICENSE-APACHE).
