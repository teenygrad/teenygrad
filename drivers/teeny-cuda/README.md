# teeny-cuda

The CUDA device backend for [teenygrad](https://teenygrad.org) — driver bindings (via `bindgen`
against the CUDA headers), device/runtime abstraction, and the AOT/JIT kernel-compilation path
used by `teeny-kernels`' `cuda` feature.

## Prerequisites

Unlike most `teeny-*` crates, **this one has a real build-time system dependency**: `build.rs`
unconditionally generates bindings against the CUDA headers and links `cuda`/`cudart`/`nvrtc`/
`nvptxcompiler_static` — it will not compile without the CUDA toolkit installed.

- **OS**: Linux (developed/tested on Ubuntu).
- **CUDA Toolkit** on the host. Verify with:
  ```bash
  nvcc --version
  ```
  By default `build.rs` looks in `/usr/local/cuda/{include,lib64}`; override with:
  ```bash
  export CUDA_INCLUDE_DIR=/path/to/cuda/include
  export CUDA_LIB_DIR=/path/to/cuda/lib64
  ```
- **Compiling kernels at runtime** (AOT/JIT, `compiler::aot`) additionally needs the custom
  `teenyc` compiler on your machine — see
  [`teeny-compiler`](https://docs.teenygrad.org/api/teenygrad/teeny_compiler/)'s README for the
  `cargo-teeny`/`TEENYC_PATH` setup. This is a *runtime* requirement, separate from the CUDA
  toolkit needed to build this crate.

Because of the CUDA header/lib requirement, this crate does not build on docs.rs — see
[`package.metadata.docs.rs`] in `Cargo.toml`. API docs are published at
<https://docs.teenygrad.org/api/teenygrad/teeny_cuda/> instead.

## Features

| Feature    | Default | Description                                    |
|------------|---------|--------------------------------------------------|
| `training` | ✅      | Enables `teeny-core`'s `training` feature passthrough. |

## Getting started

```toml
[dependencies]
teeny-cuda = "0"
```

## License

Apache-2.0. See [LICENSE-APACHE](https://github.com/teenygrad/teenygrad/blob/main/LICENSE-APACHE).
