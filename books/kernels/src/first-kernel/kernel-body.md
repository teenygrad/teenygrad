# The Kernel Body

Chapter 5 showed a whole working kernel. This chapter takes the first three
lines apart: how a program finds its slice of the data, and how you choose how
big that slice should be.

## Finding your slice

```rust
{{#include ../../../../kernels/teeny-triton/examples/vector_add.rs:indices}}
```

`T::program_id(Axis::X)` is the only thing that differs between the copies of
your kernel that are running. Everything else — the code, the constants, the
pointers — is identical across all of them. That one integer is the whole
mechanism by which parallel work gets divided up.

The rest is arithmetic. If each program handles `BLOCK_SIZE` elements, then
program `pid` starts at `pid * BLOCK_SIZE`, and the indices it touches are that
start plus `0, 1, 2, …`.

`T::arange(0, BLOCK_SIZE)` produces those offsets. Note the bounds: it is a
half-open range, like Rust's `0..n`, so `arange(0, 128)` gives 0 through 127.

### Axes

The grid can have up to three dimensions, and `Axis` selects which one you are
asking about:

```rust,ignore
let row = T::program_id(Axis::X);
let col = T::program_id(Axis::Y);
```

Use `Axis::X` alone until you have a reason not to. Two- and three-dimensional
grids are convenient for tiled kernels, but a flat grid with the index arithmetic
done by hand is equally fast and much easier to reason about. Several kernels in
this tree do exactly that — `detect_decode` in vision-rs launches a flat grid and
recovers two coordinates with a divide and a remainder:

```rust,ignore
let a_tiles = T::cdiv(A, BLOCK_A);
let pid_b   = T::program_id(Axis::X) / a_tiles;
let a_tile  = T::program_id(Axis::X) % a_tiles;
```

`T::num_programs(axis)` tells a program how many others there are, which you
need when a program must loop over more data than one block holds.

## Choosing a block size

`BLOCK_SIZE` is the number of elements one program handles. It is fixed when the
kernel is built:

```rust,ignore
let kernel = VectorAdd::<f32>::new(128);
```

Three rules, in order of importance.

**Make it a multiple of 32.** The hardware executes threads in groups of 32
called warps, always, and a partial warp still occupies a full one. A block size
of 100 does the work of 128 and wastes a quarter of it.

**Make it a power of two.** Reductions — sums, maxima — are implemented as trees
that halve the working set at each step. A power of two divides evenly all the
way down; anything else needs padding.

**Start at 128 or 256.** Small blocks mean more programs, each doing less work,
and the fixed cost of starting one starts to dominate. Large blocks mean each
program needs more registers, and past a point the card can keep fewer of them
in flight at once, so there is less work available to hide memory latency.
Between 128 and 512 is the usual sweet spot for simple kernels.

Beyond that, measure. Chapter 16 covers what the number actually controls, and
Chapter 18 shows how to time the alternatives.

> There is no autotuner. Python Triton has `@triton.autotune`, which sweeps a
> list of configurations at run time and caches the winner per input shape.
> teenygrad has no equivalent, so the choice is yours and it is static. See
> [`KNOWN-GAPS.md`](https://github.com/teenygrad/teenygrad/blob/main/books/kernels/KNOWN-GAPS.md).

## Why it is a const generic

`BLOCK_SIZE` is a const generic parameter, not a function argument:

```rust,ignore
pub fn vector_add<T: Triton, D: Num, const BLOCK_SIZE: i32>(
```

That is not a stylistic choice. Chapter 3 explained that the kernel body is
captured as text and compiled separately — and that compiler needs the block
size to be a literal in the text it receives. It is, quite literally: the
generated entry point instantiates the kernel with the number baked in, and
Chapter 8 shows the result.

This buys real things. The compiler can unroll loops whose trip count it knows,
size registers exactly, and fold the bounds arithmetic. It also means a kernel
with two block sizes is two compiled kernels — which is fine, and is how
specialisation works throughout this SDK. Chapter 15 covers it properly.

The trade is that you cannot decide the block size from data at run time. You
can pick between pre-built kernels, but each one is built for its own constant.

## Choosing the grid

The block size is baked into the kernel. The grid — how many programs to start —
is computed on the CPU at launch, and it is the one number that depends on your
actual data:

```rust,ignore
let cfg = teeny_cuda::testing::launch_config(N, BLOCK_SIZE);
```

which is a division, rounded up:

```rust,ignore
grid = [(n_elements as u32).div_ceil(block_size as u32), 1, 1]
```

Rounding up is what makes the last program partial. 1000 elements in blocks of
128 gives 8 programs, and the eighth has only 104 real elements to work on. The
other 24 lanes must not touch memory.

Which is the next chapter.
