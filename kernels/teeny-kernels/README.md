# teeny-kernels

CPU/GPU kernel implementations of [teenygrad](https://teenygrad.org)'s `nn` layers (attention —
including a Flash Attention 2 forward/backward implementation — math ops, and graph lowering),
written against the `teeny-triton` DSL and compiled via `teeny-compiler`.

## Prerequisites

- **Rust**: any stable or nightly toolchain to `cargo build`/`cargo doc` this crate — no
  system-library dependency at build time (the `cuda` feature pulls in `teeny-cuda`, which does
  have a CUDA toolkit build requirement — see its README).
- **Compiling/running kernels** at runtime needs the custom `teenyc` compiler — see
  [`teeny-compiler`](https://docs.teenygrad.org/api/teenygrad/teeny_compiler/)'s README for the
  `cargo-teeny`/`TEENYC_PATH` setup.
- **On Blackwell (sm_120) GPUs**, `teenyc`'s default PTX version for `sm_120a` may be rejected by
  the installed driver's JIT compiler (`PTX .version 8.6 does not support .target sm_120a`); if so,
  set `TEENYC_PTX_VERSION=87` (see `teeny-compiler`'s `TEENYC_PTX_VERSION` env var).

## Benchmarks

`benches/conv2d_bn_silu.rs` compares the three fused Conv2d+BatchNorm+SiLU kernel variants
(scalar, channel-tiled, GEMM/tensor-core) across shapes that straddle the shape-based dispatch
thresholds in `graph/mod.rs`:

```bash
cargo bench -p teeny-kernels --features cuda,training --bench conv2d_bn_silu
```

Needs a real CUDA device (same runtime `teenyc` requirement as above); results are written to
`target/criterion/report/index.html`.

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

Apache-2.0. See [LICENSE](https://github.com/teenygrad/teenygrad/blob/main/LICENSE).
