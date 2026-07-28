# teeny-vision

Computer-vision model definitions and datasets for [teenygrad](https://teenygrad.org) — currently
MNIST (`mnist` module, including a LeNet-5 definition used elsewhere in the workspace as a smoke
test).

## Prerequisites

- **Rust**: any stable or nightly toolchain to `cargo build`/`cargo doc` this crate — no
  system-library dependencies for the library itself.
- **Running the examples** (`examples/mnist.rs`, `examples/autograd_debug.rs`) compiles kernels
  through `teeny-compiler`'s LLVM backend at runtime, which needs the custom `teenyc` compiler on
  your machine:
  ```bash
  export TEENYC_PATH=/path/to/teenyc   # or have `teenyc` on $PATH
  ```
  The supported way to obtain `teenyc` is via `cargo-teeny` (`cargo teeny install-toolchain`) —
  see [`teeny-compiler`](https://docs.teenygrad.org/api/teeny-compiler/)'s README for details.
  This is only needed to *run* the examples, not to build the `teeny-vision` library.

## Getting started

```toml
[dependencies]
teeny-vision = "0"
```

```bash
export TEENYC_PATH=/path/to/teenyc
cargo run --example mnist
```

## License

Apache-2.0. See [LICENSE-APACHE](https://github.com/teenygrad/teenygrad/blob/main/LICENSE-APACHE).
