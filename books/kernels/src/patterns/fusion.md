# Fusing an Epilogue

Chapter 1 claimed that fusion is the most common reason to write a kernel. This
chapter is that claim, made concrete.

## The problem

A convolution followed by batch normalisation followed by a SiLU activation is
one of the most common sequences in vision models. Called as three library
operations, it does this:

```text
conv       read x, write t1
batchnorm  read t1, write t2
silu       read t2, write y
```

Six trips to memory for three operations. And the second and third operations
are nothing: batch norm is a multiply and an add, SiLU is `x * sigmoid(x)`. All
the time goes into moving `t1` and `t2` in and out of memory that the kernel had
already read once.

Fused, it is:

```text
conv+bn+silu   read x, write y
```

Two trips. The arithmetic is identical. You removed two thirds of the memory
traffic by doing the cheap work while the expensive data was still in registers.

## What an epilogue is

The pattern has a name. The **epilogue** is the work you do to a result after
computing it and before storing it, while it is still in registers.

Anything element-wise is a candidate: activations, bias adds, scaling, dropout
masks, casts to a narrower dtype. The rule of thumb is that if an operation
touches each element once and needs no neighbours, it belongs in the epilogue of
whatever produced those elements.

## The shape of it

Take the tiled matmul loop from Chapter 11. After the K loop finishes, `acc`
holds the output tile in registers. The unfused version stores it and moves on.
The fused version does the extra work first:

```text
acc = 0
for each k-chunk:
    acc += A_tile @ B_tile

# ── epilogue: the tile is in registers, use it ──
acc = acc * bn_scale + bn_bias
acc = acc * sigmoid(acc)

store acc
```

The library's `conv2d_bn_silu_gemm` kernel is exactly this. Its K loop is the
one from Chapter 11, and immediately after it comes:

```rust,ignore
// ── BatchNorm epilog ──────────────────────────────────────────────────────
let bn_off = T::arange(0, BLOCK_N) + pid_n * BLOCK_N;
let bn_n_mask = bn_off.lt(C_OUT);
let bn_scale = T::load(bn_scale_ptr.add_offsets(bn_off), Some(bn_n_mask), ...);
```

Note that the batch-norm parameters are loaded *inside* the same kernel. They
are small — one scale and one bias per output channel — so loading them costs
almost nothing next to the tile they modify.

## Three kernels for one operation

This tree ships three implementations of the same fused operation:

| Kernel | Approach |
|---|---|
| `conv2d_bn_silu` | Scalar. One output element per lane. |
| `conv2d_bn_silu_tiled` | Tiled over the output spatial extent. |
| `conv2d_bn_silu_gemm` | Convolution as a GEMM, with `T::dot` and Tensor Cores. |

Three, because none of them wins everywhere. The GEMM version has the best
arithmetic intensity but its tiles are wasted on a convolution with few output
channels. The scalar version has no setup cost and wins on small shapes.

So the lowering picks between them by shape, with thresholds hand-chosen and
written down in `kernels/teeny-kernels/src/graph/mod.rs`: the GEMM kernel for
1×1 convolutions with at least 32 output channels, the tiled kernel above 16,
the scalar one otherwise.

Two things worth taking from that.

**Hand-picked thresholds are normal.** Without an autotuner, somebody measures
and writes the number down. The bench in `benches/conv2d_bn_silu.rs` exists
specifically to check those thresholds still hold, and its doc comment says so.

**Shipping several kernels for one operation is normal too.** A single kernel
that is good at every shape is usually worse than three that are each good at
one.

## When not to fuse

Fusion is not free, and it is not always right.

**When the fused thing is also expensive.** Fusing two compute-bound operations
gains you nothing on memory and may cost you registers.

**When it costs occupancy.** A longer kernel needs more registers. Past a
threshold the card runs fewer programs at once, and the loss can exceed the
memory saved. This is measurable, and only measurable.

**When the intermediate is needed anyway.** If `t1` is consumed by something
else too, fusing means computing it twice.

**When it makes the kernel unmaintainable.** A kernel fusing five operations has
five times the ways to be wrong, and Chapter 9's MLIR is your only view into it.

The honest test is the one Chapter 18 sets up: measure the fused version against
the sequence it replaces, on the shapes you actually run.

## Fusing in the graph

Everything above is manual — you decide what to fuse and write one kernel that
does it.

There is a second kind, where the framework notices that two operations in a
graph could be merged and does it for you. teenygrad's lowering does some of
this, and Part 5 covers where the seam is between what you fuse by hand and what
the graph fuses for you.

Next: reductions beyond the sum.
