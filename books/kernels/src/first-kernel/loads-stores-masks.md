# Loads, Stores, and Masks

A kernel that does not touch memory is not doing anything. This chapter is about
the two operations that do — and about the mask, which is the thing that stops
the last program in a launch from corrupting your program's memory.

## Pointers plus offsets

Memory in a kernel is addressed by pointer arithmetic, but on whole blocks at
once:

```rust,ignore
a_ptr.add_offsets(offsets)
```

`a_ptr` is a single pointer — the base of the array. `offsets` is a tensor of
`BLOCK_SIZE` integers. `add_offsets` combines them into a **tensor of pointers**,
one per lane.

That is the type to keep in your head. `T::Pointer<D>` is one address;
`T::Tensor<T::Pointer<D>>` is a block of them, and that is what `T::load` and
`T::store` take. The offsets are in *elements*, not bytes — the dtype is in the
type, so the scaling is done for you.

`add_offsets` comes from the `AddOffsets` trait, which is why every kernel in
this book carries this line:

```rust,ignore
T::Pointer<D>: AddOffsets<i32, 1, T::I32Tensor, Output = T::Tensor<T::Pointer<D>>>,
```

Read it as: "a pointer to `D`, offset by a rank-1 tensor of `i32`, gives a
tensor of pointers to `D`". It says exactly what the sentence above says, in
types.

## The mask

```rust
{{#include ../../../../kernels/teeny-triton/examples/vector_add.rs:mask}}
```

`offsets.lt(n_elements)` compares every lane against the length and produces a
`BoolTensor` — one `true` or `false` per lane. For every program but the last,
all 128 are `true`. For the last, some are `false`.

Here is why that matters. The grid is `1000 / 128` rounded up, which is 8. The
eighth program computes offsets 896 through 1023. But the array has 1000
elements, so lanes for offsets 1000 through 1023 point past its end.

Without a mask, that kernel reads 24 values that are not yours and writes 24
values over memory that is not yours. On a GPU that is not a segfault. It is
usually silence, sometimes a wrong number somewhere else in your program, and
occasionally a crash much later in something unrelated.

**The mask is not an optimisation. It is the bounds check.**

When a lane's mask is `false`, `T::load` does not read it and `T::store` does not
write it. Nothing else about the kernel changes.

### Choosing the fill value

A masked-off lane still holds *something* after a load. By default that value is
undefined, which is fine when you are about to mask the store as well — the
vector-add kernel never uses those lanes, so it does not care.

It is not fine when the lane feeds a reduction. A sum over a block where some
lanes are garbage gives a garbage sum. Pass `other` and the masked lanes get a
known value instead:

```rust,ignore
let zeros = T::zeros::<D>(&[BLOCK_A]);
let anchor_x = T::load(anchor_x_ptr.add_offsets(a_offs), Some(mask), Some(zeros), &[], None, None, None, false);
```

Zero is the right identity for a sum. For a maximum you want negative infinity,
via `T::full`. Getting this wrong is a classic reduction bug and Chapter 10 hits
it directly.

## The full signatures

`T::load` takes eight arguments and `T::store` takes six. Two of each matter
now; the rest have a sensible "no thanks" value that you will write a great many
times.

```rust,ignore
fn load<D: Dtype, const N: usize>(
    ptr: Self::Tensor<Self::Pointer<D>>,   // where to read
    mask: Option<Self::BoolTensor>,        // which lanes are real
    other: Option<Self::Tensor<D>>,        // what masked lanes get
    boundary_check: &[i32; N],             // block-pointer mode only
    padding_option: Option<PaddingOption>, // block-pointer mode only
    cache_modifier: Option<CacheModifier>, // L1/L2 behaviour
    eviction_policy: Option<EvictionPolicy>,
    volatile: bool,
) -> Self::Tensor<D>;

fn store<D: Dtype, const N: usize>(
    dest: Self::Tensor<Self::Pointer<D>>,
    src: Self::Tensor<D>,
    mask: Option<Self::BoolTensor>,
    boundary_check: &[i32; N],
    cache_modifier: Option<CacheModifier>,
    eviction_policy: Option<EvictionPolicy>,
);
```

The last four on each are performance hints and an alternative addressing mode.
`boundary_check` and `padding_option` do nothing unless you built the pointer
with `T::make_block_ptr`, which Chapter 17 covers. `cache_modifier` and
`eviction_policy` tell the hardware how to treat the data in cache; leave them
`None` until you are measuring.

In Python Triton the same call is `tl.load(a_ptr + offsets, mask=in_bounds)`,
because Python has keyword arguments with defaults and Rust does not. There is
no way around it today, and it is the single largest source of noise in these
kernels. It is recorded in
[`API-FRICTION.md`](https://github.com/teenygrad/teenygrad/blob/main/books/kernels/API-FRICTION.md)
as the first item.

## Comparisons

`lt` is one of six, and each has a scalar form:

| Method | Meaning |
|---|---|
| `lt`, `le` | less than, less than or equal |
| `gt`, `ge` | greater than, greater than or equal |
| `eq`, `ne` | equal, not equal |

Written `offsets.lt(n_elements)` as a method — that spelling comes from the
`Comparison` trait, the second of the three `where` clauses. There is also
`T::lt(x, y)` for two tensors and `T::lt_scalar(x, y)` for a tensor against a
scalar. All three exist; the method form is what the kernels in this tree use.

Masks combine with `&` and `|`, so a two-dimensional bounds check is one
expression:

```rust,ignore
let in_bounds = row_offsets.lt(n_rows) & col_offsets.lt(n_cols);
```

And `T::where_(cond, x, y)` selects between two tensors lane by lane — Triton's
`tl.where`, spelled with a trailing underscore because `where` is a Rust keyword.

## The rule

Every kernel that indexes memory with a computed offset needs a mask, unless you
can prove the size divides the block exactly.

Sometimes you can. The softmax kernel in Chapter 10 requires the caller to round
the row length up to a power of two and pass it as `BLOCK_SIZE`, so
`BLOCK_SIZE == n_cols` holds and no mask is needed. That is a real constraint
pushed onto the caller in exchange for a simpler, faster kernel — a trade worth
recognising, and worth documenting loudly when you make it.

Next: what the macro built out of all this.
