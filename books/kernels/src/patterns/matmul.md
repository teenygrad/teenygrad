# Matrix Multiplication

Matrix multiply is the operation most machine learning time is spent in, and the
first one in this book where arithmetic — not memory — is the cost.

It is also where the GPU's specialised hardware comes in, and where the shape of
your kernel starts to matter enormously.

## The naive version

The library ships a straightforward implementation. One program computes one
element of the output:

```rust
{{#include ../../../../kernels/teeny-kernels/src/math/gemm.rs:matmul_forward}}
```

Read the middle of it. To compute `C[m, n]` you need row `m` of `A` and column
`n` of `B`, multiply them element-wise, and sum — which is Chapter 10's
reduction again, as `T::sum(a_row * b_col, Some(0), true)`.

Two details worth stopping on.

**The stride.** `A` is stored row-major, so row `m` is contiguous: offsets
`m*K + 0..K`. Column `n` of `B` is not contiguous — consecutive elements are `N`
apart, hence `k_offsets * N + n`. That difference costs real performance, and
Chapter 17 is about why.

**The early return.** `if m >= M { return; }` is a bounds check on the program
rather than on the lanes. A whole program exits. That is fine and cheap, and is
the right tool when the entire block is out of range rather than part of it.

## Why this is slow

Count the memory traffic. Every program loads `K` elements of `A` and `K` of
`B`, then writes one number. For an `M × N` output that is `2·M·N·K` loads for
`M·N` results.

But every element of `A` is only actually needed `N` times and every element of
`B` is needed `M` times. The naive kernel re-reads them from memory on every
use, when it could have read them once and reused them.

That gap is the entire subject of fast matrix multiplication.

## Tiles

The fix is to compute a **tile** of the output at a time instead of one element.

A program that produces a `BLOCK_M × BLOCK_N` tile needs `BLOCK_M` rows of `A`
and `BLOCK_N` columns of `B`. It loads them once and performs `BLOCK_M ×
BLOCK_N` multiply-accumulates with them. The bigger the tile, the more
arithmetic per byte loaded.

That ratio has a name — **arithmetic intensity** — and raising it is how a
memory-bound kernel becomes compute-bound.

The `K` dimension usually will not fit in registers, so it is walked in chunks
of `BLOCK_K`, accumulating as you go:

```text
acc = 0
for each k-chunk:
    load A tile [BLOCK_M, BLOCK_K]
    load B tile [BLOCK_K, BLOCK_N]
    acc += A_tile @ B_tile
store acc
```

The accumulator lives in registers across the whole loop and is written to
memory exactly once.

## `T::dot`

That `A_tile @ B_tile` is one operation:

```rust,ignore
acc = T::dot::<f32, f32>(w_tile, x_tile, Some(acc), Some(InputPrecision::IEEE), None);
```

`T::dot` maps onto the card's **Tensor Cores** — dedicated units that do a small
matrix multiply as a single instruction, many times faster than the general
arithmetic units. This is the whole reason a tiled matmul is fast, and you get
it by calling `dot` rather than by writing multiply and add.

Its two type parameters are separate on purpose: `D` is the input dtype, `O` is
the accumulator's. Multiplying `f16` inputs into an `f32` accumulator is the
normal arrangement, and Chapter 19 explains why mixing them that way is not a
compromise but the correct choice.

The `acc` argument is what makes the K loop work. Passing `Some(acc)` adds the
product to the existing accumulator in one operation instead of two.

`InputPrecision` controls how `f32 × f32` is handled:

| Value | Meaning |
|---|---|
| `TF32` | 19-bit mantissa, Tensor Cores, fastest — the default on capable hardware |
| `TF32x3` | Three TF32 products to recover most of `f32`'s precision |
| `IEEE` | Full `f32`. Slowest, and exact |

The fused conv kernel in this tree passes `IEEE` with a comment explaining why:
it has to match cuDNN's `f32` accumulation, and TF32 would not. That is the
right kind of reason. Absent one, `TF32` is what you want.

## The real tiled loop

The complete tiled matmul in this tree is inside
`kernels/teeny-kernels/src/nn/fused/conv2d_bn_silu_gemm.rs`, which does a
convolution as a GEMM. Its inner loop:

```rust,ignore
let mut acc = T::zeros::<f32>(&[BLOCK_N, BLOCK_M]);
let k_tiles = T::cdiv(C_IN, BLOCK_K);
for k in 0..k_tiles {
    let x_tile = T::load_tensor_descriptor(x_desc, &[b * C_IN + k * BLOCK_K, pid_m * BLOCK_M]);
    let w_tile = T::load_tensor_descriptor(w_desc, &[pid_n * BLOCK_N, k * BLOCK_K]);
    acc = T::dot::<f32, f32>(w_tile, x_tile, Some(acc), Some(InputPrecision::IEEE), None);
}
```

The loads use **tensor descriptors** rather than the pointer arithmetic of
Chapter 7. A descriptor is built once from a shape, strides and a tile shape:

```rust,ignore
let w_desc = T::make_tensor_descriptor(
    w_ptr,
    &[C_OUT, C_IN],      // full shape
    &[C_IN, 1],          // strides
    &[BLOCK_N, BLOCK_K], // tile shape
    Some(PaddingOption::Zero),
);
```

after which `load_tensor_descriptor(desc, &[row, col])` fetches the tile at that
offset. Bounds are handled by the descriptor — out-of-range reads come back as
zero because of `PaddingOption::Zero`, so there is no mask in the loop at all.

On recent cards this maps to the Tensor Memory Accelerator, hardware that moves
tiles between global and shared memory without occupying the arithmetic units.
It also imposes an alignment requirement, which is what
`RuntimeOp::forward_output_row_stride` exists to satisfy — Chapter 17 returns
to it.

## Choosing the tile

Three constants to pick, and they interact:

- **`BLOCK_M` × `BLOCK_N`** is the output tile. Larger means better arithmetic
  intensity and more registers per program. Past a point the card can keep fewer
  programs in flight, and there is less work available to hide memory latency.
- **`BLOCK_K`** is how much of the reduction dimension is loaded per iteration.
  Larger means fewer iterations and more shared memory per program.

The fused conv kernel uses 32 for all three with a group size of 8. That is a
reasonable starting point, not a universal answer.

There is no autotuner to search this for you. Chapter 18 shows how to measure
the alternatives, which is the only honest way to choose.

> `kernels/teeny-kernels/src/math/matmul.rs` looks like it should be relevant.
> It is not — the whole file is commented-out Python, including a large
> `@autotune` configuration table, and it exports nothing. The working code is
> `gemm.rs` and the fused conv kernel above.

Next: what else to do while the data is already in registers.
