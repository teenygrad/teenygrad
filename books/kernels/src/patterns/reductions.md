# Reductions and Scans

Chapter 10 used two reductions to build a softmax. This chapter is the rest of
the family: what is available, how to write one the library does not have, and
the difference between a reduction and a scan.

## The two shapes

A **reduction** turns many values into one. Sum, maximum, count.

```text
[3, 1, 4, 1, 5]  --sum-->  14
```

A **scan** turns many values into the same many values, each holding the
reduction of everything up to it. Also called a prefix operation.

```text
[3, 1, 4, 1, 5]  --cumsum-->  [3, 4, 8, 9, 14]
```

Reductions are cheap and common. Scans are less common and more expensive,
because every output depends on every earlier input, but they are how you
implement anything involving running totals — offsets into a variable-length
buffer, sampling from a distribution, sorting.

## What is built in

Reductions, all taking an `axis` and `keep_dims`:

| Method | Result |
|---|---|
| `sum`, `max`, `min` | The obvious |
| `max_with_indices`, `min_with_indices` | The value *and* where it was |
| `argmax`, `argmin` | Just where it was |
| `xor_sum` | XOR-fold, integers only |

Scans and friends:

| Method | Result |
|---|---|
| `cumsum`, `cumprod` | Running total / product along an axis |
| `sort` | Sorted along a dimension |
| `histogram` | Counts into `num_bins` bins of width 1 |

Two conventions apply throughout, and both were introduced in Chapter 10:

- **`axis`** — `Some(n)` reduces dimension `n`, `None` reduces everything.
- **`keep_dims`** — `true` leaves a length-1 dimension so the result can
  broadcast back against the input. This is almost always what you want inside
  a kernel.

The `*_with_indices` and `arg*` variants also take `tie_break_left`. With `true`
the leftmost of equal values wins. It matters more than it sounds: if your
kernel and your reference implementation break ties differently, a test on data
with duplicates fails for a reason that looks like a real bug.

## A worked one

The library's sum-reduction kernel:

```rust
{{#include ../../../../kernels/teeny-kernels/src/nn/tensor/reduction.rs:reduce_sum_forward}}
```

The pattern is Chapter 10's, without the numerical-stability step: one program
per output, load the slice being reduced, mask it, reduce, store one value.

Note the masked-lane fill. It has to be the identity for the operation — zero
for a sum — or the masked lanes contribute garbage. Chapter 10 has the table of
identities; this is the kernel where getting it wrong is easiest, because the
result is a single number that looks plausible.

## Writing your own

When the operation you need is not in the list, `T::reduce` takes a combine
function:

```rust,ignore
fn combine_max_abs<T: Triton, D: Float>(a: T::Tensor<D>, b: T::Tensor<D>) -> T::Tensor<D> {
    T::maximum(T::abs(a), T::abs(b))
}

let result = T::reduce(x, 0, combine_max_abs::<T, D>, true);
```

and `T::associative_scan` is the same idea for a prefix operation, plus a
`reverse` flag.

Two requirements, and the second is a genuine gotcha.

**The function must be associative.** The compiler builds a tree and combines
pairs in an unspecified order, so `f(f(a, b), c)` and `f(a, f(b, c))` must agree.
Maximum is associative. Subtraction is not. Floating-point addition is not
*exactly* associative, which is why a GPU sum and a CPU sum can differ in the
last bits — expected, and not a bug.

**It must be a `fn` pointer, not a closure.**

```rust,ignore
fn reduce<D, O>(x: ..., axis: i32, combine_fn: fn(Self::Tensor<O>, Self::Tensor<O>) -> Self::Tensor<O>, keep_dims: bool) -> ...;
```

A closure that captures anything is rejected. This follows directly from
Chapter 3: the kernel body is compiled from captured source text, so the combine
function has to be a statically-known name that can be written out. A closure's
captured environment cannot be.

Python Triton has the same restriction — the combine function needs
`@triton.jit` — but the Rust error message does not mention kernels at all. It
is a generic closure-coercion complaint, and it is worth recognising:

```text
expected fn pointer `fn(...) -> ...`
found closure `[closure@src/...]`
```

## Cost

A reduction over `n` lanes takes `log2(n)` steps, not `n`. Halving the working
set each round is what makes it cheap, and it is why powers of two matter for
block sizes — Chapter 6's second rule.

A scan is more expensive: the standard algorithm makes two passes over the tree,
so roughly twice the work of a reduction. Still `O(log n)` depth, but do not
reach for `cumsum` where `sum` would do.

## Reducing across programs

Everything here reduces *within* one program. Getting a single number out of a
whole tensor that does not fit in one block is a different problem, and there are
two answers.

The first is two kernels: one produces a partial result per program, the second
reduces those partials. Predictable, deterministic, and needs a scratch buffer.

The second is atomics — every program folds its partial into one location in
memory. That is the next chapter, along with why it is not the default.
