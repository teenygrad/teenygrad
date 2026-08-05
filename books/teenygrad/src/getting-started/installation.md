# Installation & Toolchain

## Rust

Most `teeny-*` crates build with a plain stable (or nightly) Rust toolchain and no system
dependencies — see each crate's own README (linked from its
[API docs](https://docs.teenygrad.org/api/teenygrad/)) for specifics. Two crates are exceptions:

- **`teeny-cuda`** requires the CUDA toolkit headers/libraries on the host at *build* time (its
  `build.rs` runs `bindgen` against them unconditionally). See its README for
  `CUDA_INCLUDE_DIR`/`CUDA_LIB_DIR`.
- **`teeny-onnx`** vendors the upstream ONNX proto schema as a git submodule — run
  `git submodule update --init support/teeny-onnx/onnx` in a workspace checkout (not needed when
  depending on the published crate).

## The `teenyc` compiler (runtime only)

Compiling/running kernels — through `teeny-compiler`'s LLVM/MLIR backend, `teeny-triton`'s DSL, or
`teeny-cuda`'s AOT/JIT path — shells out **at runtime** to a custom compiler binary, `teenyc` (a
fork of `rustc` with an MLIR codegen backend). This is *not* needed to `cargo build`/`cargo doc`
any `teeny-*` crate, only to actually compile and run kernels.

```bash
export TEENYC_PATH=/path/to/teenyc   # optional — see detection below
```

The supported way to obtain it is via [`cargo-teeny`](https://github.com/spinorml/cargo-teeny),
which downloads a prebuilt release and links it as a named `rustup` toolchain:

```bash
cargo install --git https://github.com/spinorml/cargo-teeny
cargo teeny install-toolchain
```

This mirrors the toolchain setup used by downstream projects such as
[`vision-rs`](https://github.com/spinorml/vision-rs).

### How `teenyc` is located

`teeny-compiler::compiler::find_teenyc` resolves the binary in two steps:

1. `$TEENYC_PATH`, if set — used as-is.
2. Otherwise, the sole `rustup`-linked toolchain whose name contains `teenyc` (the naming
   convention `cargo teeny install-toolchain` uses — by default `<channel>-<host-triple>` with
   `channel` defaulting to `stable-teenyc`), resolved to a binary path via
   `rustup which --toolchain <name> teenyc`.

If neither step finds a binary — no env var, and no matching `rustup` toolchain (or more than one,
which is ambiguous) — this returns an error rather than silently guessing at a bare `teenyc` on
`$PATH`.

## System setup (Ubuntu)

For working on the teenygrad workspace itself (not just depending on its published crates):

```bash
sudo apt-get install build-essential z3 libz3-dev lld
```

or run `./setup_ubuntu.sh` from the repository root.

> **TODO**: expand this chapter with GPU compute-capability selection, PTX ISA version overrides
> (`TEENYC_PTX_VERSION`), and cross-compilation notes once they're documented per-crate.
