# `teeny-cli` and Ahead-of-Time Compilation

[`teeny-cli`](https://docs.teenygrad.org/api/teeny-cli/) provides ahead-of-time (AOT) kernel
compilation as both a reusable library and a runnable reference binary.

## As a library

Flatten `teeny_cli::AotArgs` (a `clap`-derived struct) into your own binary's CLI, and call
`teeny_cli::aot_compile`:

```rust,ignore
#[derive(clap::Parser)]
struct Cli {
    #[command(flatten)]
    aot: teeny_cli::AotArgs,
}
```

This is the pattern [`vision-rs`](https://github.com/spinorml/vision-rs) uses in its own binaries
— see `teeny-cli`'s own `main.rs` for the full reference implementation (AOT-compiling
`teeny-vision`'s LeNet-5).

`aot_compile` traces your model with a symbolic input (see
[Tensors & the Computational Graph](../core-concepts/tensors-and-graph.md#tracing)), then compiles
every kernel the traced graph references for the backend/config selected by `--device`. Currently
only `--device cuda` is implemented; adding a new backend means adding a match arm in
`aot_compile`, not changing `AotArgs`.

## Requires `teenyc` at runtime

Like the rest of the compilation pipeline, this needs `TEENYC_PATH` set — see
[Installation & Toolchain](../getting-started/installation.md). It's typically driven via
`cargo teeny aot` (from `cargo-teeny`) rather than run directly.

```bash
cargo teeny aot --bin teeny-cli --device cuda --options "capability=sm_90"
```
