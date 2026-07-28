# Introduction

> ⚠️ **Warning**: teenygrad is still under active development and not yet ready for production
> use. This book, and the crates it documents, will change frequently.

[teenygrad](https://teenygrad.org) is a high-performance, memory-safe Rust ML training and
inference library, in the spirit of PyTorch and tinygrad. It targets devices from
microcontrollers to data centers with statically-typed GPU kernels and full async support.

## Vision

We envision a world where machine learning is truly ubiquitous: from the tiniest embedded sensors
to the largest distributed clusters, every device can harness the power of AI efficiently, safely,
and without constraints.

Goals:

- **Highly concurrent, memory-safe** — designed for scalability and safety.
- **Low memory footprint** — optimized to run even on the smallest devices.
- **Statically typed** — GPU kernels and ML algorithms are statically typed.
- **No performance compromises** — hardware-accelerated wherever possible.
- **Broad hardware support** — not just NVIDIA and AMD.
- **Extensible** — build high-performance accelerators for your own hardware.
- **Embedded-friendly** — core training/inference components are `no_std` compatible
  ([`teeny-core`](https://docs.teenygrad.org/api/teeny-core/) builds `no_std` by default).
- **Full async support**, multi-threaded by default.

## How this book is organized

- **Getting Started** walks through installing the toolchain and building your first model.
- **Core Concepts** covers the tensor/graph types, the compilation pipeline, the dtype system, and
  name scoping — the ideas every other chapter builds on.
- **Neural Network Layers** covers the `nn` layer API.
- **Kernels & Backends** covers how kernels are written (the Triton-like DSL) and compiled to
  device code.
- **Compiler Internals** goes under the hood of the FXGraph compiler.
- **PyTorch Interop** covers `teeny-torch`, the PyO3-based Python bridge.
- **CLI & AOT Compilation** covers `teeny-cli` and `teeny-llm`.

This book is a guide and mental model; it deliberately doesn't duplicate API reference material —
for that, see the generated rustdoc at
[docs.teenygrad.org/api](https://docs.teenygrad.org/api/teeny-core/) (or [docs.rs](https://docs.rs)
for the crates that build there).

## Community

[Join our Discord](https://discord.gg/FG65XyPzuD) to discuss the project or get help.
