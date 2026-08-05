# Numerics

A kernel that is fast and slightly wrong is worse than one that is slow and
right, because the wrongness is invisible until it is expensive.

This chapter is about choosing dtypes and accumulators deliberately.

## The type hierarchy

The dtype bounds you put on a kernel are a real constraint, checked at compile
time:

```rust,ignore
pub trait Dtype: Copy + Clone {}
pub trait Num: Dtype { const BITS: u8; }
pub trait Float: Num { const ZERO: Self; const ONE: Self; /* from_f64 */ }
pub trait Int: Num {}
pub trait Bool: Dtype + Copy {}
```

Pick the tightest one that admits your operations:

| Bound | Admits | Use when |
|---|---|---|
| `Float` | `f32`, `f64` | You call `exp`, `log`, `sqrt`, `sigmoid` |
| `Int` | `i8`…`i64`, `u8`…`u64` | Bitwise operations, integer atomics |
| `Num` | both | Arithmetic only |
| `Dtype` | everything | Pure data movement |

`Float` also gives you `D::from_f64(...)`, which is how you get a constant of
the right type into a kernel — `T::full(&[BLOCK], D::from_f64(0.5))`. There is
no way to write a float literal of a generic float type without it.

Choosing the tightest bound is not pedantry. `D: Num` on a kernel that calls
`T::exp` is a compile error, and that error is the one Python Triton would have
given you on a GPU, at run time, in production.

## What is actually available

`DtypeRepr` — the runtime, type-erased tag — has thirteen variants including
`F16` and `BF16`. The macro's dtype-set inference does not:

| Bound | Dtypes you get |
|---|---|
| `Float` | `f32`, `f64` |
| `Int` | `i8`, `i16`, `i32`, `i64`, `u8`, `u16`, `u32`, `u64` |
| `Num` | all of the above |
| `Bool` | `bool` |
| `Dtype` | all of the above |

`f16` and `bf16` are absent, deliberately: the macro's own comment says they are
marker-only and cannot be monomorphized, because there is no concrete Rust
implementation to instantiate a kernel against.

**This is the largest gap in the book.** Half precision is a substantial part of
why people write GPU kernels — it halves memory traffic and doubles Tensor Core
throughput — and today you cannot write a `#[kernel]` that dispatches to it. It
is item 4 in
[`KNOWN-GAPS.md`](https://github.com/teenygrad/teenygrad/blob/main/books/kernels/KNOWN-GAPS.md).

What follows is therefore mostly about `f32`, which is what you can use.

## The formats, for when this is fixed

Worth knowing, because the trade-offs are what motivate everything below.

| Format | Bits | Exponent | Mantissa | Character |
|---|---|---|---|---|
| `f64` | 64 | 11 | 52 | Rarely worth it on a GPU |
| `f32` | 32 | 8 | 23 | The default |
| TF32 | 19 stored in 32 | 8 | 10 | `f32`'s range, less precision, Tensor Cores |
| `bf16` | 16 | 8 | 7 | `f32`'s range, much less precision |
| `f16` | 16 | 5 | 10 | More precision than `bf16`, far less range |

The exponent column is the important one. `bf16` has the same range as `f32`, so
a value that fits in `f32` fits in `bf16` — you lose precision, not magnitude.
`f16` has five exponent bits and overflows above about 65,504, which is why
training in `f16` needs loss scaling and training in `bf16` mostly does not.

## What `f64` actually costs

`examples/numerics.rs` runs an elementwise add over 16M elements, once as `f32`
and once as `f64`:

```bash
cargo run --release -p teeny-triton --features cuda --example numerics
```

On an **RTX 5070 (sm_120), CUDA 13.3, driver 610.43.02**:

| dtype | time | bandwidth | moved |
|---|---:|---:|---:|
| `f32` | 339.5 µs | 593.1 GB/s | 201 MB |
| `f64` | 676.7 µs | 595.0 GB/s | 403 MB |

**`f64` costs 1.99× the time — and the same bandwidth.**

That equality is the whole story. This kernel is memory-bound, so `f64` costs
exactly what its extra bytes cost and not one thing more. Both runs saturate the
card at ~594 GB/s, the same ceiling Chapters 16 and 17 hit with different
kernels.

Which means the folk wisdom "`f64` is catastrophically slow on a consumer GPU"
is, for this kernel, wrong. It is catastrophically slow when *arithmetic* is the
bottleneck — consumer cards run `f64` arithmetic at a small fraction of their
`f32` rate — and this kernel barely does any. Move a byte, add once, move a
byte.

So the honest rule: **on a memory-bound kernel `f64` costs 2×; on a
compute-bound one it costs far more.** Chapter 1's question — which kind is
this? — decides which number applies.

## Accumulators

Here is the rule that matters most:

> **Accumulate in `f32`, whatever you multiply in.**

`T::dot` has two type parameters precisely for this:

```rust,ignore
fn dot<D: Num, O: Num>(a: Tensor<D>, b: Tensor<D>, acc: Option<Tensor<O>>, ...) -> Tensor<O>;
```

`D` is the inputs, `O` is the accumulator. Mixing them is not a compromise; it is
the correct arrangement, and the Tensor Cores are built for it.

The reason is that error accumulates with the number of additions. Multiplying
two `f16` values gives a result `f16` can represent fine. Adding a thousand of
them in `f16` does not — each addition rounds, and after `K` additions the error
has grown roughly with `sqrt(K)` at best. In `f32`, with sixteen more mantissa
bits, the same sequence stays accurate.

The same applies to hand-written reductions. Summing a long `f16` row into an
`f16` total loses precision that summing into `f32` does not.

### Measured

The same example sums 16M `f32` values four ways. The reference is the identical
data summed in `f64`, so the only thing varying is how the sum is accumulated:

| how | result | relative error |
|---|---:|---:|
| exact (`f64` reference) | 2181038.0593 | — |
| GPU block reduction, `f64` accumulator | 2181037.9700 | **4.09e-8** |
| GPU block reduction, `f32` accumulator | 2181055.7500 | 8.11e-6 |
| sequential `f32` loop on the CPU | 2158069.0000 | **1.05e-2** |

Read the last row first. **A plain `f32` loop is wrong in the fifth
significant figure** — a 1% error, from nothing but adding numbers up in order.
Once the running total reaches ~2 million, adding 0.1 to it barely moves it, and
16 million such additions lose most of what they should have contributed.

The GPU's block reduction is **over a thousand times more accurate** than that
loop, and it is not because the GPU is careful. It is because a tree adds
numbers of *similar magnitude* to each other: 16M values pair down through 24
levels, and no partial sum ever dwarfs what is being added to it. The parallel
algorithm is more accurate than the obvious sequential one, which is the
opposite of what most people expect.

Widening only the final accumulator — the host-side sum of the per-block
partials — from `f32` to `f64` gains another **200×**. That is one cast, on
16384 values, and it is free next to the kernel.

So the practical shape of an accurate reduction is: reduce in blocks on the
device, accumulate the partials in something wider. You get the tree for free
and the wide accumulator for almost nothing.

### And it is not reproducible

The same data through the same kernel at two different block sizes:

```text
BLOCK=256   → 2181037.969997
BLOCK=1024  → 2181037.969986
```

A different block size is a different tree shape, and floating-point addition is
not associative, so the answers differ — here by 5.2e-12 relative. Tiny, real,
and enough to break an exact-equality assertion. This is the concrete version of
the warning below.

## `InputPrecision`

For `f32 × f32`, `T::dot` lets you choose what the hardware actually does:

| Value | Behaviour |
|---|---|
| `TF32` | Round inputs to 19 bits, use Tensor Cores. Fastest. Default on capable hardware |
| `TF32x3` | Three TF32 products combined to recover most of `f32`'s precision |
| `IEEE` | True `f32` arithmetic. Slowest, exact |

`TF32` keeps `f32`'s range and drops thirteen mantissa bits. For neural network
training that is almost always fine, which is why it is the default.

> **`IEEE` turns off the Tensor Cores.** It is not a slightly slower, slightly
> more accurate mode — Triton only routes a `dot` to the matrix hardware for the
> reduced-precision modes, so `IEEE` falls back to software
> fused-multiply-add. The fused conv kernel in this tree was written with
> `IEEE` to match cuDNN's accumulation and silently lost its Tensor Cores;
> it now uses `TF32`.

So "it felt safer" is an expensive instinct here. Choose `IEEE` when you need
exact `f32` more than you need the hardware, knowing that is the trade.

`TF32x3` is the middle option — three TF32 products recovering most of `f32`'s
precision while staying on the Tensor Cores — and is worth knowing about because
most people do not know it exists.

## Casting

```rust,ignore
fn cast<Src: Dtype, Dst: Dtype>(x: Tensor<Src>, rounding: Option<FpDowncastRounding>, bitcast: bool) -> Tensor<Dst>;
```

Two arguments to be careful with.

`FpDowncastRounding` applies when narrowing: `Rtne` rounds to nearest with ties
to even, `Rtz` truncates toward zero. `Rtne` is what you want; `Rtz` biases every
value toward zero, and a systematic bias through a training loop compounds in a
way that random rounding error does not.

`bitcast: true` reinterprets the bits rather than converting the value. It is
occasionally what you want, and it is never what you want by accident.

## Where precision goes

**Catastrophic cancellation.** Subtracting two nearly equal large numbers leaves
a result dominated by their rounding error. `sqrt(a² - b²)` is the classic; use
`sqrt((a-b)*(a+b))`.

**Summing many values.** Error grows with the count. A long reduction in `f32`
is fine; the same in `f16` is not.

**Exponentials.** `exp(x)` overflows `f32` above about 88, which is why softmax
subtracts the maximum first — Chapter 10.

**Division by something near zero.** Normalisation layers add an epsilon for
this reason.

## Determinism

Two things make GPU results non-reproducible, and both are expected rather than
broken:

**Floating-point addition is not associative.** A reduction combines in an
unspecified order, so the last bits can differ between a GPU and a CPU
reference, or between two block sizes — measured above at 5.2e-12 relative for
a block size change alone.

**Atomics arrive in an unspecified order.** Chapter 14. So a backward pass using
them is not bit-reproducible run to run.

Consequences for your tests: compare with a tolerance, never exact equality. The
tests in this tree use `1e-5` for a forward pass and `1e-6` for a backward one.
And when a test does fail, check whether the difference is at the tolerance
boundary — a mismatch in the last two bits is arithmetic; a mismatch in the first
digit is a bug.

## A checklist

1. **Tightest bound that admits your operations.** `Float` if you use `exp`.
2. **Accumulate wider than you store.** `f32` for `f16` inputs; `f64` for the
   host-side sum of `f32` partials, which costs nothing and bought 200× above.
3. **Reduce in a tree, not a loop.** You get this for free from `T::sum`, and it
   was 1000× more accurate than the sequential version.
4. **Leave `InputPrecision` at `TF32`** unless you can say why not — `IEEE`
   turns the Tensor Cores off.
5. **`Rtne` when narrowing**, not `Rtz`.
6. **Subtract the max before exponentiating.**
7. **Compare with a tolerance** in tests, and know which tolerance and why.
8. **`f64` costs 2× on a memory-bound kernel** and far more on a compute-bound
   one. Know which you have before ruling it out.
9. **Half precision is not available yet.** Design for it; do not depend on it.

## End of Part 4

You can choose a block size, reason about layout, measure honestly, and pick
dtypes deliberately.

Every number in this part was measured on one card, an RTX 5070, and each
chapter names it. Three independent kernels — the block-size sweep, the
coalescing comparison, and the `f32`/`f64` add — all plateau within 1% of
594 GB/s, which is the most useful single fact in these four chapters: it is
this card's ceiling, and it is what a new kernel should be judged against.

Your card's number will differ. The method will not.
