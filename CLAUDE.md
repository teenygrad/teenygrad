# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Teenygrad is a high-performance, memory-safe Rust ML training and inference library. It targets devices from microcontrollers to data centers with statically-typed GPU kernels and full async support.

## Build Commands

```bash
# Build entire workspace
cargo build

# Build with release optimizations
cargo build --release

# Build specific package
cargo build -p teeny-core
cargo build -p teeny-compiler
cargo build -p teeny-torch

# Build with compiler features (egg + z3 optimization)
cargo build -p teeny-core --features compiler
```

## Testing

```bash
# Run all tests
cargo test

# Run tests for specific package
cargo test -p teeny-core
cargo test -p teeny-macros

# Run a single test
cargo test -p teeny-core test_name
```

## System Setup (Ubuntu)

```bash
# Install build dependencies
sudo apt-get install build-essential z3 libz3-dev lld

# Or run the setup script
./setup_ubuntu.sh
```

## Rust Toolchain

Uses nightly Rust (`nightly-2025-12-05`) with components: rust-src, rustc-dev, llvm-tools-preview. The toolchain is pinned in `rust-toolchain.toml`.

## Architecture

### Workspace Structure

The codebase is organized as a Cargo workspace with resolver "3":

- **core/teeny-core** - Foundation: tensor types, computational graph, neural network layers, dtype system
- **compiler/** - LLVM backend compilation, FXGraph to optimized code transformation
- **kernels/** - Triton-based GPU kernels (teeny-kernels, teeny-triton)
- **drivers/** - Hardware backends: teeny-cpu, teeny-cuda, teeny-vulkan
- **torch/teeny-torch** - PyO3 Python bindings for PyTorch interoperability
- **models/** - teeny-hf (Hugging Face integration), teeny-samples
- **apps/teeny-llm** - LLM inference application (binaries: tllm, tllm-agent, tllm-console)
- **utilities/** - Helper crates: teeny-cache, teeny-data, teeny-http, teeny-nlp
- **macros/teeny-macros** - Procedural macros (Edition 2021 for macro crate compatibility)
- **support/teeny-onnx** - ONNX model format support

### Compilation Flow

```
FXGraph (traced computation) → Compiler → Backends (LLVM/ndarray/Triton)
                                           ↓
                                Device Drivers (CPU/CUDA/Vulkan)
```

### Key Feature Flags

- `teeny-core` has `compiler` feature enabling egg + z3 optimizer
- `teeny-compiler` has `cpu` and `cuda` backend features
- Multiple crates have optional `ndarray` support

### Dependencies

- **Async**: tokio (full features), smol
- **Compiler**: egg (e-graph optimization), z3 (SMT solver, path-based at `../z3.rs/z3`)
- **Python**: pyo3 with extension-module
- **Serialization**: safetensors, flatbuffers, protobuf
- **NLP**: tokenizers

## Rust Best Practices

### Ownership and Borrowing

- Pass complex types by reference (`&T` or `&mut T`) to avoid unnecessary cloning
- Use `&[T]` slices instead of `&Vec<T>` for function parameters to accept both arrays and vectors
- Use `&str` instead of `&String` for string parameters
- Prefer borrowing over cloning; only clone when ownership transfer is required

### Error Handling

- Use `thiserror` for library error types, `anyhow` for application error handling
- Error messages must be specific, actionable, and include relevant values
- Avoid `.unwrap()` in library code; use `?` operator or explicit error handling
- Use early returns for validation checks to reduce nesting

### Performance

- Avoid heap allocations in hot paths; prefer stack-allocated arrays or `SmallVec` for small collections
- Use `#[inline]` judiciously for small, frequently-called functions
- Prefer iterators over indexed loops for better optimization
- Mark functions that cannot panic as `#[inline]` candidates

### Async Code

- Use `tokio` as the primary async runtime (workspace standard)
- Prefer `async`/`await` over manual `Future` implementations
- Use `rayon` for CPU-parallel data processing, `tokio` for I/O-bound async

### API Design

- Use enums instead of bool parameters when the meaning isn't clear at call sites
- Initialize all struct fields explicitly; derive `Default` when zero-initialization is appropriate
- Use the builder pattern (via `derive_builder`) for structs with many optional fields
- Prefer `impl Into<T>` or `impl AsRef<T>` for flexible parameter types

### Module Organization

- Keep implementations in the same file as type definitions (unlike C++ header/source split)
- Use `pub(crate)` for internal APIs that shouldn't be exposed publicly
- Avoid `use` statements that pull entire namespaces into scope in library code

### Concurrency

- Avoid global mutable state; if necessary, use `once_cell::sync::Lazy` or `std::sync::OnceLock`
- Prefer message passing (channels) over shared state when possible
- Use `Arc<T>` for shared ownership across threads, `Rc<T>` only in single-threaded contexts

## Code Style

- Most crates use Edition 2024; procedural macro crate uses Edition 2021
- Python files use flake8 with max-line-length = 120
- Error messages should be specific, actionable, and include relevant values (see `contributing/ErrorMessageBestPractices.md`)

## Contributing

Contributions require a signed CLA. All contributions need an issue and a pull request with appropriate reviewers.

## Git Commits

Do not add "Co-Authored-By: Claude" or any Claude attribution lines to commit messages.

## Key reference material

- This code uses a pre-installed version of the rust compiler (the source code can be found in ../rust)/
- During development you can compile and install the compiler using the rust bootstrap script `../rust/x.py install`. The installation will be into a directory that this project is aware of (via environment variables).
