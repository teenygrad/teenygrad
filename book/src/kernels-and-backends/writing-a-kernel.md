# Writing a Triton Kernel

[`teeny-triton`](https://docs.teenygrad.org/api/teeny-triton/) provides a Triton-like DSL for
writing kernels in Rust. A kernel is a function generic over the DSL's `Triton`/`Dtype`/`Float`
traits, marked with the [`#[kernel]`](./kernel-macro.md) attribute macro:

```rust,ignore
use teeny_macros::kernel;
use teeny_core::dtype::Float;

#[kernel]
pub fn my_kernel<T: Triton, D: Float, const HEAD_DIM: i32>(
    // pointers, strides, block-size const-generics, etc.
) {
    // kernel body, written against T's tensor/pointer operations
}
```

See `kernels/teeny-kernels/src/nn/attention/flash_attn2.rs` for a real, well-documented example
(Flash Attention 2 forward/backward) — a good template to copy for new kernels.

## How it compiles

At `teeny-triton`'s own build time, `build.rs` reads the DSL source under `src/triton/` (plus
`teeny-core`'s dtype definitions) and embeds it as a string constant
(`teeny_triton::triton_lang::TRITON`) — pure text processing, no compiler invocation.

At *your* kernel's compile time (via `teeny-compiler`'s LLVM/MLIR backend), that DSL text plus
your kernel's source is written out and compiled by the custom `teenyc` compiler — see
[The LLVM/MLIR Backend](../compiler-internals/llvm-backend.md).

> **TODO**: document the available DSL operations (tensor loads/stores, arithmetic, reductions,
> control flow) as a proper reference once the DSL surface stabilizes.
