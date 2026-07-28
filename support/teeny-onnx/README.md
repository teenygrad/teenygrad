# teeny-onnx

ONNX model format support for [teenygrad](https://teenygrad.org) — parses `.onnx` protobuf files
into a `teeny-core::graph::Graph`.

## Prerequisites

- **Rust**: any stable or nightly toolchain. `protoc` is vendored via `protoc-bin-vendored` and
  code-generated at build time via `protobuf-codegen` — no system `protoc` install required.
- **The `onnx` git submodule** (vendoring the upstream ONNX proto schema at
  `support/teeny-onnx/onnx`, pinned to tag `v1.22.0`) must be checked out — `build.rs` parses
  `onnx/onnx/onnx.proto3` from it directly:
  ```bash
  git submodule update --init support/teeny-onnx/onnx
  ```
  (Published crates.io tarballs include the submodule contents as regular files, so this is only
  needed when building from a git checkout of this workspace.)

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
