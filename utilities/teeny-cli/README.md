# teeny-cli

Ahead-of-time (AOT) kernel compilation helpers for [teenygrad](https://teenygrad.org). Provides a
reusable `AotArgs` (a `clap`-derived struct, meant to be `#[command(flatten)]`ed into a downstream
binary's own CLI) and `aot_compile`, plus a `teeny-cli` binary that's a runnable smoke test /
reference implementation — it AOT-compiles `teeny-vision`'s LeNet-5 against the CUDA backend, and
is the pattern a downstream project (e.g. [`vision-rs`](https://github.com/spinorml/vision-rs))
copies into its own binary.

## Prerequisites

- **Rust**: any stable or nightly toolchain to `cargo build`/`cargo doc` the library — no
  system-library dependency at build time unless the `cuda` feature is enabled.
- **`cuda` feature** (required for the `teeny-cli` binary — `required-features = ["cuda"]`) pulls
  in `teeny-cuda`, `teeny-kernels`, and `teeny-vision`, and therefore needs the CUDA toolkit to
  build — see [`teeny-cuda`](https://docs.teenygrad.org/api/teenygrad/teeny_cuda/)'s README.
- **`riscv` feature** enables the `riscv` `--device` backend (`teeny-riscv`, which compiles
  kernels for RISC-V RVV via `teenyc`'s `riscv64-generic` target). It needs no CUDA toolkit. The
  reference binary is CUDA-only, so for RISC-V enable `teeny-cli/riscv` in your own binary and
  pass `--device riscv --options "capability=<chip>"`, where `<chip>` is `generic-rvv1.0`,
  `spacemit-k1` or `spacemit-k3`. An optional `log-level=<level>` controls `teenyc` logging.
- **Running** the binary (or `aot_compile` in your own binary) needs the custom `teenyc` compiler
  on your machine, installed via `cargo teeny install-toolchain` — it's auto-detected from there
  (no env var needed). See [`teeny-compiler`](https://docs.teenygrad.org/api/teenygrad/teeny_compiler/)'s
  README for the `cargo-teeny` setup and the `TEENYC_PATH` override.

## Getting started

As a library, flattened into your own CLI:

```toml
[dependencies]
teeny-cli = "0"
```

```rust
#[derive(clap::Parser)]
struct Cli {
    #[command(flatten)]
    aot: teeny_cli::AotArgs,
}
```

Or run the reference binary directly (after `cargo teeny install-toolchain`):

```bash
cargo run -p teeny-cli --features cuda --bin teeny-cli -- --device cuda --options "capability=sm_90"
```

Typically driven via `cargo teeny aot --bin teeny-cli --device cuda --options "capability=sm_90"`
(from `cargo-teeny`) rather than run directly. For a RISC-V board, `cargo teeny package --target
riscv-rvv` (or `spacemit-k3`) runs your binary's AOT step with `--device riscv` and the board's
capability by default.

## License

Apache-2.0. See [LICENSE](https://github.com/teenygrad/teenygrad/blob/main/LICENSE).
