# Using `teeny-torch`

`teeny-torch` is a [PyO3](https://pyo3.rs)-based Python extension module (built as a `cdylib`,
Python module name `teenygrad`) bridging teenygrad into PyTorch-style Python workflows.

> **Not published to crates.io.** `teeny-torch` (and `teeny-fxgraph`, which it depends on) are
> excluded from the crates.io publish set (`publish = false`) — they're the PyTorch compatibility
> layer, not part of the public Rust SDK. `teeny-torch` is distributed as a Python package (PyPI),
> not a Rust crate.

It depends on `teeny-core`, `teeny-fxgraph`, and `teeny-compiler`, and uses `egg` (see
[The egg/z3 Optimizer](../compiler-internals/optimizer.md)) and `flatbuffers` for its own
serialization needs (built via `flatc-rust` in `build.rs`).

> **TODO**: document the actual Python-facing API and installation instructions once this crate
> has a stable public interface and a packaging story for PyPI.
