# teeny-kernels

CPU/GPU kernel implementations of [teenygrad](https://teenygrad.org)'s `nn` layers (attention —
including a Flash Attention 2 forward/backward implementation — math ops, and graph lowering),
written against the `teeny-triton` DSL and compiled via `teeny-compiler`.

## Prerequisites

- **Rust**: any stable or nightly toolchain to `cargo build`/`cargo doc` this crate — no
  system-library dependency at build time (the `cuda` feature pulls in `teeny-cuda`, which does
  have a CUDA toolkit build requirement — see its README).
- **Compiling/running kernels** at runtime needs the custom `teenyc` compiler — see
  [`teeny-compiler`](https://docs.teenygrad.org/api/teeny-compiler/)'s README for the
  `cargo-teeny`/`TEENYC_PATH` setup.

## Features

| Feature    | Default | Description                                         |
|------------|---------|-------------------------------------------------------|
| `cuda`     | ✅      | Enables the `teeny-cuda` backend (requires the CUDA toolkit to build — see `teeny-cuda`'s README). |
| `training` | ✅      | Enables `teeny-core`'s `training` feature passthrough. |

## Getting started

```toml
[dependencies]
teeny-kernels = "0"
```

## License

Apache-2.0. See [LICENSE-APACHE](https://github.com/teenygrad/teenygrad/blob/main/LICENSE-APACHE).
