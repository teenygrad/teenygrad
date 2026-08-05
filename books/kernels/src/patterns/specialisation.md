# Compile-Time Parameters and Dtype Dispatch

Two facts from earlier chapters collide here.

From Chapter 6: `BLOCK_SIZE` is a compile-time constant, so a kernel with two
block sizes is two compiled kernels.

From Chapter 8: the dtype is a type parameter, filled in when the entry point is
generated, so a kernel over `f32` and one over `f64` are also two compiled
kernels.

Both are **specialisation**: one source, many compiled artefacts, each with its
constants baked in. This chapter is about what that buys, and about the machinery
for choosing between them at run time.

## What specialisation buys

Look again at the MLIR from Chapter 9. `BLOCK_SIZE` does not appear — `128`
does, in the multiply, in the range, and in every tensor type.

That is not cosmetic. A compiler that knows the trip count can unroll the loop.
One that knows the tensor shapes can allocate exactly the registers needed. One
that knows the dtype can pick the right instruction rather than a generic one.

The cost is one compilation per combination, and the `id` from Chapter 8 is what
keeps them apart — `vector_add__f32__128` and `vector_add__f32__256` are
different cache entries.

## When the dtype is only known at run time

A graph loaded from a file says its tensors are `f32`. A different file says
`f64`. You cannot pick a Rust type parameter from a value.

The `#[kernel]` attribute generates a dispatcher for this:

```rust,ignore
#[kernel(dtypes = [f32, f64])]
pub fn vector_add<T: Triton, D: Num, const BLOCK_SIZE: i32>(...)
```

which produces, alongside `VectorAdd`, a `VectorAddDispatch` with:

```rust,ignore
pub const SUPPORTED_DTYPES: &'static [DtypeRepr];

pub fn dispatch(dtype: DtypeRepr, block_size: i32) -> anyhow::Result<KernelInstance>;
```

Call it with a runtime `DtypeRepr` and you get back a `KernelInstance` — the
compiled forward kernel, its source, a runtime dispatch object, and its backward
if it has one. An unsupported dtype is an error naming what *is* supported,
rather than a panic.

The const-generic parameters become arguments to `dispatch`, in declaration
order, exactly as they are for `new`.

## The implicit set

If you opt into dispatch without listing dtypes — which happens when you use
`backward` alone — the macro infers the set from the dtype parameter's trait
bound:

| Bound | Dtypes |
|---|---|
| `Float` | `f32`, `f64` |
| `Int` | `i8`…`i64`, `u8`…`u64` |
| `Num` | all of the above |
| `Bool` | `bool` |
| `Dtype` | everything above |

So `D: Float` with no explicit list means "`f32` and `f64`". If the bound is not
one of these five, the macro cannot infer anything and says so:

```text
cannot infer supported dtypes: a `#[kernel]` that opts into dispatch without an
explicit `dtypes = [..]` must have a dtype type parameter bound by one of
Dtype/Num/Int/Float/Bool
```

## Two things the table does not say

**`f16` and `bf16` are missing.** `DtypeRepr` has variants for both, and
`#[kernel(dtypes = [f16])]` parses. But neither appears in any implicit set,
because — in the macro's own words — they are marker-only and cannot be
monomorphized. There is no concrete Rust implementation to instantiate the
kernel against.

Since half precision is a large part of why people write GPU kernels at all,
this is the biggest single gap in this book. It is recorded as item 4 in
[`KNOWN-GAPS.md`](https://github.com/teenygrad/teenygrad/blob/main/books/kernels/KNOWN-GAPS.md).
Chapter 19 covers what you *can* do about precision today.

**Nothing in this tree uses `dtypes = [...]`.** Every real kernel either takes
no attribute at all or uses `backward`. The explicit dtype list is generated
code with no in-tree user, so it is less exercised than the rest of the macro —
worth knowing before you rely on it.

What *is* used, in dozens of kernels, is the pairing attribute:

```rust
{{#include ../../../../kernels/teeny-kernels/src/nn/activation/gelu.rs:gelu_forward}}
```

`#[kernel(backward = GeluBackward)]` names the kernel that computes this one's
gradient. It also opts into dispatch, which is how these kernels get their
implicit `f32`/`f64` set. Chapter 22 covers the backward half.

## The one that is missing

Python Triton has `@triton.autotune`: give it a list of configurations, and the
first time it sees a new input shape it runs them all and caches the winner.

There is no equivalent here. Block sizes are chosen by a person, written into a
constructor call, and stay there.

That is not necessarily worse — an autotuner spends real time on its first call
and can pick differently between runs, which makes benchmarking harder. But it
does mean the numbers in your kernel are only as good as the last time somebody
measured. The thresholds in `graph/mod.rs` that Chapter 12 mentioned are
hand-picked for exactly this reason, and the bench beside them exists to check
they still hold.

Chapter 18 is how you do that measuring.

## Choosing what to specialise on

Make something a const generic when:

- it changes the generated code meaningfully — a block size, a tile shape, a
  flag that removes a branch;
- it takes few distinct values in practice.

Keep it a runtime argument when:

- it is data — a length, a stride, a pointer;
- it varies widely, since every distinct value is another compilation.

The failure mode is specialising on something with many values: a kernel
specialised on sequence length compiles afresh for every sequence length it
sees, and the compile time swamps whatever the specialisation saved.

## End of Part 3

You have the patterns: reductions, tiling with an accumulator, fusion, atomics,
and specialisation. Together they cover most of what real kernels are made of.

Part 4 is about making a kernel you already have run faster — which starts with
being able to tell whether it did.
