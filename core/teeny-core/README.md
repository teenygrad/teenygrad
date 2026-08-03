# teeny-core

Foundation crate of [teenygrad](https://teenygrad.org): tensor/graph types, the computational
graph (`Graph`/`Op`/`Shape`), neural network layers (`nn`), the dtype system, device abstraction,
and name-scoping used by every other `teeny-*` crate.

## Prerequisites

- **Rust**: a recent stable or nightly toolchain. This crate has no system-library dependencies
  and no custom compiler requirements — a plain `cargo build` is all that's needed.
- **`no_std` by default.** The `std` feature is *not* in the default feature set, so
  `#![cfg_attr(not(feature = "std"), no_std)]` applies unless you opt in:
  ```toml
  teeny-core = { version = "...", features = ["std"] }
  ```

## Features

| Feature    | Default | Description                                              |
|------------|---------|------------------------------------------------------------|
| `training` | ✅      | Enables training-time graph/autograd machinery.            |
| `std`      |         | Opts out of `no_std`; needed if you rely on `std`-only deps downstream. |

## Getting started

```toml
[dependencies]
teeny-core = "0"
```

Downstream crates typically depend on `teeny-core` for `Graph`, `Op`, `Shape`, `DtypeRepr`, and
the `nn::Layer` trait — see `teeny-kernels` and `teeny-compiler` for how the computational graph
is lowered and compiled.

## Related crates

- [`teeny-compiler`](https://docs.teenygrad.org/api/teenygrad/teeny_compiler/) — compiles a `teeny-core`
  `Graph` (via FXGraph) down to a target backend.
- [`teeny-kernels`](https://docs.teenygrad.org/api/teenygrad/teeny_kernels/) — CPU/GPU kernel
  implementations of `teeny-core`'s `nn` layers.

## License

Apache-2.0. See [LICENSE-APACHE](https://github.com/teenygrad/teenygrad/blob/main/LICENSE-APACHE).
