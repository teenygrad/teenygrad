# teeny-triton

The Triton-like kernel DSL for [teenygrad](https://teenygrad.org). Kernel functions are written
against this crate's `triton`/`triton_lang` types and `#[kernel]`-annotated
(`teeny-macros::kernel`); at build time, `build.rs` embeds `src/triton/`'s DSL source (plus
`teeny-core`'s dtype definitions) as a string constant (`teeny_triton::triton_lang::TRITON`),
which the custom `teenyc` compiler consumes at kernel-compile time to generate device code.

## Prerequisites

- **Rust**: any stable or nightly toolchain to `cargo build`/`cargo doc` this crate. `build.rs`
  only does text processing (reading source files via `cargo_metadata`) — no system-library or
  custom-compiler dependency to build the crate itself.
- **Compiling/running kernels** defined with this crate's DSL requires the custom `teenyc`
  compiler at runtime (via `teeny-compiler`'s LLVM backend), not at `teeny-triton`'s own build
  time. See [`teeny-compiler`](https://docs.teenygrad.org/api/teenygrad/teeny_compiler/)'s README for the
  `TEENYC_PATH`/`cargo-teeny` setup.

## Getting started

```toml
[dependencies]
teeny-triton = "0"
```

## License

Apache-2.0. See [LICENSE](https://github.com/teenygrad/teenygrad/blob/main/LICENSE).
