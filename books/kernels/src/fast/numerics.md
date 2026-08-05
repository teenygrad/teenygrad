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

## `InputPrecision`

For `f32 × f32`, `T::dot` lets you choose what the hardware actually does:

| Value | Behaviour |
|---|---|
| `TF32` | Round inputs to 19 bits, use Tensor Cores. Fastest. Default on capable hardware |
| `TF32x3` | Three TF32 products combined to recover most of `f32`'s precision |
| `IEEE` | True `f32` arithmetic. Slowest, exact |

`TF32` keeps `f32`'s range and drops thirteen mantissa bits. For neural network
training that is almost always fine, which is why it is the default.

When it is not fine, say why. The fused conv kernel in this tree passes `IEEE`
with a comment: it has to match cuDNN's `f32` accumulation. That is a real
reason. "It felt safer" is not — it is a large, silent slowdown.

`TF32x3` is the middle option, and is worth knowing about because most people do
not know it exists.

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
reference, or between two block sizes.

**Atomics arrive in an unspecified order.** Chapter 14. So a backward pass using
them is not bit-reproducible run to run.

Consequences for your tests: compare with a tolerance, never exact equality. The
tests in this tree use `1e-5` for a forward pass and `1e-6` for a backward one.
And when a test does fail, check whether the difference is at the tolerance
boundary — a mismatch in the last two bits is arithmetic; a mismatch in the first
digit is a bug.

## A checklist

1. **Tightest bound that admits your operations.** `Float` if you use `exp`.
2. **Accumulate in `f32`.** Always, whatever you multiply in.
3. **Leave `InputPrecision` at `TF32`** unless you can say why not.
4. **`Rtne` when narrowing**, not `Rtz`.
5. **Subtract the max before exponentiating.**
6. **Compare with a tolerance** in tests, and know which tolerance and why.
7. **Half precision is not available yet.** Design for it; do not depend on it.

## End of Part 4

You can choose a block size, reason about layout, measure honestly, and pick
dtypes deliberately.

The measurements are the part this book cannot give you. Everything here tells
you what to measure and how to avoid measuring the wrong thing — the numbers
have to come from your card.
