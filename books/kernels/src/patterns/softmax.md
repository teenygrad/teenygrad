# Softmax: Your First Reduction

Every kernel so far has been embarrassingly parallel: lane `i` reads element
`i`, does arithmetic, writes element `i`. No lane needed to know anything about
any other.

Softmax breaks that. To compute one output you need a sum over the whole row,
which means the lanes have to combine their values. That operation is a
**reduction**, and it is the first genuinely new idea in this book.

## The operation

Softmax turns a row of numbers into a probability distribution:

```text
softmax(x)_i = exp(x_i) / sum_j exp(x_j)
```

Every output depends on every input in its row. The denominator is the
reduction.

## Why the obvious version is wrong

Write that formula directly and it breaks. `exp(x)` overflows `f32` at about
`x = 88`, and logits above 88 are entirely ordinary. You get `inf / inf`, which
is `NaN`, and the `NaN` spreads through the rest of your model.

The fix relies on softmax being invariant to shifts. Subtract any constant from
every element and the result is unchanged, because the constant cancels:

```text
exp(x_i - c) / sum_j exp(x_j - c)
```

Choose `c = max(x)`. Now the largest exponent is `exp(0) = 1`, nothing
overflows, and the terms that underflow to zero were negligible anyway.

That is what "numerically stable softmax" means, and it costs a second
reduction: one for the maximum, one for the sum.

## The kernel

Here is the library's implementation:

```rust
{{#include ../../../../kernels/teeny-kernels/src/nn/activation/softmax.rs:softmax_forward}}
```

The shape of it is different from anything in Part 2. **One program handles one
whole row.** `pid` is the row index, not a slice index, and `row_offset` jumps
to the start of that row.

There is no mask, and no `T::arange(0, BLOCK_SIZE) + block_start` either —
`col_offsets` covers the entire row in one go.

## The constraint, and why it is there

Look at the doc comment: `BLOCK_SIZE` must equal `n_cols`. The caller is
required to round the row length up to the next power of two and pass that as
the block size.

That is a real burden pushed onto the caller. In exchange:

- **No mask is needed**, because the block exactly covers the row.
- **No loop is needed**, because the whole row is in registers at once.
- **The reduction is a single tree**, with no partial-result bookkeeping.

The cost is that a row wider than the largest workable block size cannot use
this kernel at all, and a row of 513 elements pays for 1024.

This is a fair trade and a common one, but it is exactly the kind of constraint
that must be shouted rather than buried. If you write a kernel with a
precondition like this, say so in the doc comment, as this one does.

## Doing the reduction

The kernel uses `T::softmax`, a builtin that does the whole stable sequence.
Written out, it is:

```rust,ignore
let row_max = T::max(x, Some(0), true);      // reduce
let shifted = x - row_max;                    // broadcast back
let numerator = T::exp(shifted);
let denominator = T::sum(numerator, Some(0), true);  // reduce
let y = numerator / denominator;
```

Five lines, two of which are reductions. Three things about them:

**The `axis` argument selects what to reduce.** `Some(0)` reduces along
dimension 0; `None` reduces everything to a scalar.

**`keep_dims` decides the shape of the result.** With `true`, reducing a
`[128]` tensor gives `[1]` rather than a scalar — which is what lets `x -
row_max` broadcast back across the row. With `false` you get the scalar, and the
subtraction will not line up. This is the single most common mistake in a first
reduction.

**You do not write the reduction.** In CUDA, `T::sum` would be a shared-memory
tree: each thread writes a partial, barrier, half the threads combine pairs,
barrier, repeat. Here the compiler emits all of that. Chapter 2 promised this
would be the payoff of the block model, and this is it.

## Watch the masked lanes

The softmax kernel avoids masks entirely, which sidesteps a trap. Most reduction
kernels cannot, and then the `other` argument from Chapter 7 becomes essential:

```rust,ignore
// Summing: masked lanes must be 0, the identity for addition.
let zeros = T::zeros::<D>(&[BLOCK_SIZE]);
let x = T::load(ptr.add_offsets(offs), Some(mask), Some(zeros), &[], None, None, None, false);
let total = T::sum(x, Some(0), true);
```

If you leave `other` as `None`, the masked lanes hold undefined values, and
those undefined values go into the sum. The result is wrong in a way that
depends on whatever was in memory — so it will be right in testing and wrong in
production.

The identity depends on the reduction:

| Reduction | Fill masked lanes with |
|---|---|
| `sum` | `0` |
| `max` | the most negative representable value |
| `min` | the most positive representable value |
| product | `1` |

For a maximum, `T::full(&[BLOCK], D::from_f64(f64::NEG_INFINITY))`.

## Running it

The library kernel has tests, including one that runs on a device:

```bash
cargo test -p teeny-kernels --features cuda --test test_softmax
```

There is also a snapshot test that needs `teenyc` but no GPU, which compiles the
kernel and checks its MLIR — the Chapter 9 pattern.

## The backward pass

Softmax has an unusually neat gradient. Given the saved output `y` and the
upstream gradient `dy`:

```text
dx_i = y_i * (dy_i - sum_j(y_j * dy_j))
```

That inner sum is another row-wide reduction, and it is a scalar broadcast back
across the row — the same shape of computation as the forward pass. The library
implements it as `softmax_backward` in the same file, and Chapter 22 covers how
a backward kernel gets wired to its forward.

Next: the reduction's opposite problem — a kernel where the arithmetic, not the
memory, is the cost.
