# FAQ & Roadmap

## Roadmap

### 2026 Q1 — Torch Inductor in Rust

- **2026 and beyond**
  - Q3: `teeny-llm` — vLLM-style inference, 2x faster
  - Q4: performance optimization
- **2027 and beyond**
  - Q2: embedded support
  - Q3: sparsity/quantization support
  - Q4: observability/metrics

(See the [repository README](https://github.com/teenygrad/teenygrad#roadmap) for the current,
authoritative roadmap — this chapter may lag behind.)

## FAQ

### Why create this project when PyTorch, TensorFlow, and tinygrad already exist?

Those frameworks excel at development and deployment on large-scale infrastructure. We believe
the future of AI lies in devices of all sizes — from edge devices to massive clusters. Existing
frameworks are relatively heavy; TensorFlow Lite comes closest to this vision, but a modern,
memory-safe language was preferred over C.

At the same time, we didn't want to sacrifice distributed training or other advanced features
offered by larger projects — hence teenygrad: a lightweight but powerful alternative.

### I'm interested in contributing — how do I get started?

See [Contributing to Teenygrad](../contributing/contributing.md).
