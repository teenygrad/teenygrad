# The Dtype System

`teeny-core::dtype` ([API docs](https://docs.teenygrad.org/api/teeny-core/dtype/)) defines the
element-type traits used throughout the workspace — `Dtype`, `Float`, and tensor traits
(`Tensor`, `RankedTensor`, `EagerTensor`, `Comparison`) that generic kernel/layer code is written
against (e.g. `teeny-kernels`' attention kernels are generic over `D: Float`).

Host-only functionality that doesn't belong in the kernel DSL context (like converting a float to
its little-endian byte representation) lives in a separate `bytes` submodule
(`dtype::bytes::FloatBytes`) rather than on the core `Float` trait itself — this split exists
because `teeny-triton`'s `build.rs` embeds `dtype/mod.rs`'s source directly into the kernel DSL
(see [Writing a Triton Kernel](../kernels-and-backends/writing-a-kernel.md)), so anything in that
file needs to make sense in a `no_core` DSL context, not just as normal host-side Rust.

`DtypeRepr` (in `teeny-core::graph`, not `dtype`) is the runtime/graph-level enum representation
of a dtype (e.g. `DtypeRepr::F32`), distinct from the compile-time `Dtype`/`Float` traits — used
when tracing a graph (`SymTensor::input(dtype, shape)`) where the dtype isn't a type parameter.

> **TODO**: expand with a full trait reference and guidance on writing dtype-generic kernel code,
> once the trait hierarchy stabilizes.
