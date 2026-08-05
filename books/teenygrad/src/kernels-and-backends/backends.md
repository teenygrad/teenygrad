# CPU, CUDA, and Vulkan Backends

Device backends live under `drivers/` in the workspace. Today, only
[`teeny-cuda`](https://docs.teenygrad.org/api/teenygrad/teeny_cuda/) (NVIDIA, via `bindgen`-generated
driver bindings) is implemented. `teeny-cpu` and `teeny-vulkan` are on the
[roadmap](../appendix/faq-and-roadmap.md) but don't exist yet — the `ndarray`-backed CPU path in
`teeny-compiler` (the `ndarray` feature, on by default) is the current CPU story, distinct from a
dedicated `teeny-cpu` driver crate.

`teeny-kernels`' `cuda` feature (on by default) enables the `teeny-cuda` backend for its kernel
implementations. Building it requires the CUDA toolkit — see
[Installation & Toolchain](../getting-started/installation.md).

Kernel *definitions* (the actual per-op logic — attention, matmul, elementwise ops, etc.) are
backend-agnostic, written once against the `teeny-triton` DSL and compiled per-backend — see
[Writing a Triton Kernel](./writing-a-kernel.md).

> **TODO**: document the `teeny-cuda` device/runtime API directly (device selection, memory
> management, profiling via `cuda_profiler_start`/`cuda_profiler_stop`) once a driver-level
> chapter is warranted.
