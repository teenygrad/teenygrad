# teeny-data

Dataset loading utilities for [teenygrad](https://teenygrad.org): downloading (via `teeny-http`
style `reqwest`/`tokio`), CSV parsing, `safetensors` loading, and memory-mapped access
(`memmap2`).

## Prerequisites

- **Rust**: any stable or nightly toolchain. No system-library or custom-compiler dependencies.

## Getting started

```toml
[dependencies]
teeny-data = "0"
```

See the `dataset` and `safetensors` modules for the available APIs.

## License

Apache-2.0. See [LICENSE](https://github.com/teenygrad/teenygrad/blob/main/LICENSE).
