# Writing a Triton Kernel

Kernel authoring has its own book: **[Writing GPU Kernels in
Rust](https://docs.teenygrad.org/kernels)**.

It starts at what a GPU kernel is, assumes no CUDA experience, and goes as far
as attaching a custom kernel to a model's graph with a backward pass. This page
is the two-minute version and a map into it.

## The shape of a kernel

A kernel is a function generic over the
[`teeny-triton`](https://docs.teenygrad.org/api/teenygrad/teeny_triton/) DSL's
`Triton` trait plus a dtype, marked with [`#[kernel]`](./kernel-macro.md):

```rust,ignore
#[kernel]
pub fn vector_add<T: Triton, D: Num, const BLOCK_SIZE: i32>(
    a_ptr: T::Pointer<D>,
    b_ptr: T::Pointer<D>,
    out_ptr: T::Pointer<D>,
    n_elements: i32,
) where
    T::I32Tensor: types::Tensor<i32, 1>,
    T::I32Tensor: Comparison<i32, BoolTensor = T::BoolTensor>,
    T::Pointer<D>: AddOffsets<i32, 1, T::I32Tensor, Output = T::Tensor<T::Pointer<D>>>,
{
    let pid = T::program_id(Axis::X);
    let offsets = T::arange(0, BLOCK_SIZE) + pid * BLOCK_SIZE;
    let in_bounds = offsets.lt(n_elements);
    // ... load, compute, store
}
```

You write what **one program** does with **one block** of the data — not what
one thread does with one element. `program_id` is how a program finds its slice.

## How it compiles

At `teeny-triton`'s own build time, `build.rs` reads the DSL source under
`src/triton/` (plus `teeny-core`'s dtype definitions) and embeds it as a string
constant (`teeny_triton::triton_lang::TRITON`) — pure text processing, no
compiler invocation.

At *your* kernel's compile time (via `teeny-compiler`'s LLVM/MLIR backend), that
DSL text plus your kernel's source is written out and compiled by the custom
`teenyc` compiler — see
[The LLVM/MLIR Backend](../compiler-internals/llvm-backend.md).

The consequence worth knowing up front: **your kernel function is never called
by your program.** Its source text is the artefact.

## Where to go

| You want | Read |
|---|---|
| To run a kernel today | [Vector Add, End to End](https://docs.teenygrad.org/kernels/first-kernel/vector-add) |
| The programming model | [You Program a Block, Not a Thread](https://docs.teenygrad.org/kernels/orientation/block-not-thread) |
| Why the source is captured as text | [From Rust to PTX](https://docs.teenygrad.org/kernels/orientation/rust-to-ptx) |
| Loads, stores and masking | [Loads, Stores, and Masks](https://docs.teenygrad.org/kernels/first-kernel/loads-stores-masks) |
| Every DSL operation | [Python Triton to Rust](https://docs.teenygrad.org/kernels/reference/translation-table) |
| A kernel inside a model | [Your Kernel as a Graph Op](https://docs.teenygrad.org/kernels/in-a-model/graph-op) |

For a large, well-documented in-tree example, see
`kernels/teeny-kernels/src/nn/attention/flash_attn2.rs` (Flash Attention 2,
forward and backward).
