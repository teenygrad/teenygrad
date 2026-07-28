# teeny-onnx

ONNX model format support for [teenygrad](https://teenygrad.org) — parses `.onnx` protobuf files
into a `teeny-core::graph::Graph`.

## Prerequisites

- **Rust**: any stable or nightly toolchain. `protoc` is vendored via `protoc-bin-vendored` and
  code-generated at build time via `protobuf-codegen` — no system `protoc` install required.

## Getting started

```toml
[dependencies]
teeny-onnx = "0"
```

```rust
use teeny_onnx::Onnx;

let graph = Onnx::from_path("model.onnx")?;
```

## License

Apache-2.0. See [LICENSE-APACHE](https://github.com/teenygrad/teenygrad/blob/main/LICENSE-APACHE).
