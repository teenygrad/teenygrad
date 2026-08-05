# Memory Coalescing and Tensor Layout

Most kernels are memory-bound. For a memory-bound kernel, the single largest
factor is not how much you load — it is the *order* you load it in.

This chapter is about that, and it is the one most likely to make a real
difference to a kernel you have already written.

## Coalescing

When the lanes of a program read memory, the hardware tries to service them with
as few transactions as possible. Memory comes in fixed-size chunks; if the
addresses your lanes want all fall in one chunk, that is one transaction. If
they are scattered, it is one transaction per chunk touched.

Contiguous access:

```text
lanes:      0    1    2    3    4    5    6    7
addresses:  0    1    2    3    4    5    6    7     → 1 transaction
```

Strided access, stride 8:

```text
lanes:      0    1    2    3    4    5    6    7
addresses:  0    8   16   24   32   40   48   56     → 8 transactions
```

Same number of values. Eight times the memory traffic, because each transaction
fetches a whole chunk and you use one value from it.

This is why `T::arange(0, BLOCK_SIZE) + block_start` appears in every simple
kernel. Consecutive lanes get consecutive addresses, which is the pattern the
hardware is built for.

## Seeing it in a real kernel

The naive matmul from Chapter 11 has both patterns, side by side:

```rust,ignore
// Row m of A — contiguous.
let a_offsets = k_offsets + m * K;

// Column n of B — stride N.
let b_col_offsets = k_offsets * N + n;
```

*From `kernels/teeny-kernels/src/math/gemm.rs`.*

`A` is row-major, so a row is contiguous and reads well. A *column* of a
row-major matrix has its elements `N` apart, so the same loop reads it badly.
The multiply by `N` in the offset expression is the tell — any offset expression
whose lane-varying term is multiplied by something is strided.

Two ways out, and tiling is the reason the tiled version of this kernel exists:
load a tile once into fast memory, then read it in whatever order you like.

## Strides, and reading an index expression

For a row-major array, the stride of a dimension is the product of all the
dimensions after it. For `[B, C, H, W]`:

| Dimension | Stride |
|---|---|
| `W` | 1 |
| `H` | `W` |
| `C` | `H * W` |
| `B` | `C * H * W` |

So the flat index is `b*(C*H*W) + c*(H*W) + h*W + w`, which is exactly what the
conv kernels compute:

```text
x[b, c_in, oh, ow] = x_flat[b*(C_IN*M) + c_in*M + oh*OW + ow]
```

The trick for reading these quickly: **find which term varies with the lane
index**. If it is the one with stride 1, the access is contiguous. Anything else
is strided, and the multiplier tells you how badly.

## NCHW versus NHWC

This is where layout choice becomes a design decision rather than an
observation.

A batch of images has four dimensions: batch `N`, channels `C`, height `H`,
width `W`. Two orderings are in common use.

**NCHW** stores all of one channel's pixels together. `[N][C][H][W]`, with `W`
contiguous. It is what PyTorch uses by default, and what this tree's conv
kernels assume.

**NHWC** stores all of one pixel's channels together. `[N][H][W][C]`, with `C`
contiguous.

Neither is better in general. Which one wins depends on what varies across your
lanes.

| Operation | Wants |
|---|---|
| Convolution over spatial extent | NCHW — neighbouring pixels of a channel are adjacent |
| Per-channel scale and bias | NHWC — a pixel's channels are adjacent |
| Matmul-style contraction over channels | NHWC — the reduction dimension is contiguous |
| Tensor Core paths | Usually NHWC |

This is why a fused conv+batchnorm is more interesting than it looks. The
convolution wants NCHW; the batch norm, which applies one scale per channel,
wants NHWC. Fusing them means one of the two runs against the grain — but that
is still far better than writing the intermediate to memory in one layout and
reading it back in the other.

The tree makes this explicit in `channel_bias_add`, whose doc comment says it
treats a `(B, C, H, W)` tensor as `NC` with `N = B*H*W` — a reinterpretation
that makes the channel dimension the fast one for that operation, without moving
any data.

**Converting between layouts costs a full pass over memory.** It is worth it only
if the destination layout saves more than one pass. Usually the answer is to
pick one layout for the whole model and live with it.

## Block pointers and descriptors

For tiled access there are two addressing modes that describe the tile rather
than computing every address.

`make_block_ptr` takes shape, strides, offsets, a block shape, and an order:

```rust,ignore
let ptr = T::make_block_ptr(base, &shape, &strides, &offsets, &block_shape, &order);
let tile = T::load(ptr, None, None, &[0, 1], Some(PaddingOption::Zero), None, None, false);
```

The `boundary_check` argument — the one that is `&[]` in every kernel in Part 2
— becomes useful here: it names the dimensions to check, and out-of-range lanes
get `padding_option` instead of a mask you built by hand.

`make_tensor_descriptor` is the TMA form from Chapter 11, and on hardware that
has it the copy is done by dedicated hardware rather than by the arithmetic
units.

Both let the compiler see the whole access pattern at once, which is more than
it can infer from arbitrary offset arithmetic.

## The alignment constraint

TMA requires rows aligned to 16 bytes — four elements for `f32`. A tensor whose
last dimension is not a multiple of four therefore cannot be addressed directly.

The SDK handles this by padding: `RuntimeOp::forward_output_row_stride` returns
the stride the buffer actually has, rounded up, and the executor allocates
accordingly and passes the real stride to `pack_args`.

Which means: **use the `output_row_stride` argument, not
`output_shape.last()`**. They are usually equal and occasionally not, and when
they differ, computing your own gives a kernel that reads the wrong addresses.
The same applies to `backward_grad_output_row_stride` on the backward path.

## Compiler hints

Three methods promise the compiler something it cannot prove:

| Method | Promise |
|---|---|
| `T::multiple_of(x, values)` | These values are multiples of these constants |
| `T::max_contiguous(x, values)` | This many elements are contiguous along each dimension |
| `T::max_constancy(x, values)` | This many elements are constant along each dimension |

They generate no code. They let the compiler emit wider loads it would otherwise
have to guard.

They are also unchecked promises. If you tell it offsets are multiples of 16 and
they are not, you get wrong results with no diagnostic. Use them when you know
something structural — a padded stride, an aligned base — and not otherwise.

## A checklist

1. **Find the lane-varying term** in each offset expression.
2. **Check its multiplier.** Stride 1 is contiguous; anything else is not.
3. **If it is strided, can you tile?** Load once, reuse from fast memory.
4. **If not, can you change the layout?** Sometimes the fix is upstream.
5. **Use the passed row stride**, never the shape's last dimension.
6. **Measure.** Chapter 18. Coalescing changes are usually dramatic and
   occasionally do nothing, and the only way to tell is to time it.

Next: how to time it.
