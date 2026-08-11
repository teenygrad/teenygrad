# Choosing a Block Size

Chapter 6 gave you three rules of thumb and told you to measure. This chapter is
what you are actually choosing between, and what the SDK does and does not let
you control.

## Two different numbers

The word "block" does two jobs, and conflating them causes real confusion.

**The const generic** — `BLOCK_SIZE`, `BLOCK_OW`, `BLOCK_M` — is how much
*data* one program covers. It is baked into the kernel at compile time, and it
determines the shape of every tensor inside the body.

**The launch block** — `CudaLaunchConfig::block` — is how many *threads* the
hardware gives that program.

**They are never the same thing, and you only choose one of them.** `teenyc`
decides the thread count and records it in the PTX as a `.reqntid` directive.
The driver enforces it: launch with any other block dimension and you get

```text
Error: CUDA error: 1 (invalid argument)
```

That is not a warning or a slowdown. It is a hard failure, and it is what you
hit the first time you assume `BLOCK_SIZE` is a thread count.

The sweep below makes it plain — six different `BLOCK_SIZE` values, and the
compiler picks 128 threads for every one of them.

For a tiled kernel the two are visibly different. `conv2d_bn_silu` has
`BLOCK_OW = 16` — sixteen output columns per tile — and its bench launches 128
threads:

```rust,ignore
let cfg = CudaLaunchConfig {
    grid: [grid, 1, 1],
    block: [128, 1, 1],
    cluster: [1, 1, 1],
};
```

Those 128 threads cooperate on a 16-wide tile. The compiler decides how. Nothing
requires the two numbers to match, and nothing checks that your choice is
sensible.

## What you control

```rust,ignore
pub struct CudaLaunchConfig {
    pub grid: [u32; 3],     // how many programs
    pub block: [u32; 3],    // threads per program
    pub cluster: [u32; 3],  // programs per cluster (Hopper and later)
}
```

Plus the const generics, chosen when you build the kernel. That is the whole
surface.

`RuntimeOp` exposes the program count through `grid()`; thread count comes from
the compiled PTX. The helpers in `teeny_cuda::testing` build a config for you:

| Helper | Use |
|---|---|
| `launch_config_with_grid(grid_x, &program)` | **The safe one.** Grid you computed, threads from the PTX |
| `launch_config_from_program(n, &program)` | Threads from the PTX, grid from the element count — correct only when one program handles exactly one thread's worth of data |
| `launch_config(n_elements, block_size)` | Both from you. Fails unless `block_size` happens to equal what the compiler chose |

Prefer the first. You know how much data one program covers — that is your
`BLOCK_SIZE` — so compute the grid from it and let the metadata supply the
threads:

```rust,ignore
let grid = N.div_ceil(BLOCK_SIZE as usize);
let cfg = teeny_cuda::testing::launch_config_with_grid(grid, &program);
```

`launch_config` is a trap in waiting. It works whenever your `BLOCK_SIZE`
coincides with the compiler's thread count, which for a simple elementwise
kernel at 128 it usually does — and then silently stops working when you change
the constant.

The loader parses the thread count out of `.reqntid` like this:

```rust,ignore
let threads = program.metadata.threads_per_block().max(1);
CudaLaunchConfig {
    grid: [(n_elements as u32).div_ceil(threads), 1, 1],
    block: [threads, 1, 1],
    cluster: [program.metadata.num_ctas.max(1), 1, 1],
}
```

So the compiler's own choice is available, and using it is usually right.

## What you do not control

This is the part that differs sharply from Python Triton.

**`num_warps` is not settable.** In Python it is a launch argument: how many
warps cooperate on one program. Here it exists only as a value parsed *out* of
compiled PTX — derived from `.reqntid`, rounded up to whole warps — and the
struct holding it is `pub(crate)`. You cannot read it directly, let alone set
it; the only access is through `launch_config_from_program`.

**`num_stages` does not exist at all.** In Python it controls the depth of the
compiler's software pipeline — how many loop iterations' loads are in flight at
once. There is no equivalent anywhere in this SDK.

Both are recorded as item 3 in
[`KNOWN-GAPS.md`](https://github.com/teenygrad/teenygrad/blob/main/books/kernels/KNOWN-GAPS.md).
It is why this chapter is narrower than a Triton performance guide would be:
much of what such a guide tells you to tune is not exposed.

What you *can* tune is the thread count and the tile shape, which between them
still cover most of the available performance.

## Occupancy

Occupancy is how many programs a card can keep resident at once. It matters
because it is how memory latency gets hidden: when one program stalls waiting
for a load, another runs.

Each program consumes:

- **Registers**, for the tensors in its body. Bigger blocks and more live
  tensors mean more registers.
- **Shared memory**, for reductions and tiles.

The card has a fixed budget of each per multiprocessor. Divide, and you get how
many programs fit.

That is the tension. A larger tile does more arithmetic per byte loaded — good
— but uses more registers, so fewer programs are resident, so there is less
other work to hide latency with — bad. The optimum is somewhere in the middle
and it moves with the kernel and the card.

Some of the inputs are visible in the PTX metadata the loader parses: the shared
memory a kernel wants, its global scratch requirements, the thread count. These
are exposed as a public API on `CudaProgram`. Registers are too, but only on
request — `ptxas` has to actually run to know them, which
[CUDA Diagnostics](../reference/cuda-diagnostics.md) covers. Short of that,
reading them means either the PTX itself or `nvdisasm`.

**Maximum occupancy is not the goal.** A kernel at 50% occupancy with good
arithmetic intensity routinely beats one at 100% that is memory-starved.
Occupancy is a diagnostic, not a target.

### The other kind of occupancy problem

Everything above is about *resources* — registers and shared memory limiting how
many programs fit. There is a second, simpler failure that looks identical from
a benchmark and is completely different underneath: **not launching enough
programs to fill the machine.**

A card has some number of streaming multiprocessors. If your grid is 40 programs
and the card has 48 SMs, most of the machine is idle regardless of how efficient
your kernel is. No amount of register tuning helps.

This is a real case in this tree. Profiling YOLO26n found conv layers running at
8% achieved occupancy with 100% *theoretical* occupancy — the giveaway that
resources were not the limit. Deep layers have small spatial extent and many
channels, and a fixed tile size produced as few as 40 blocks.

The fix was to pick the tile size from the shape:

```rust,ignore
let lowering = TritonLowering::default().with_sm_count(Some(48));
```

With `sm_count` set, the lowering chooses the largest candidate tile whose
resulting grid still clears a small multiple of the SM count — smaller tiles
where that means more blocks, larger where the shape already provides enough.
Left at `None`, the default, tile sizes stay fixed exactly as before.

Two things to take from this.

**Check your grid size before tuning anything else.** It is one division, and it
rules out the most embarrassing cause.

**The SM count is a parameter, not a query.** It sits alongside `ptx_version` in
`Options` and is deliberately not read from the local device, because
ahead-of-time compilation routinely targets a card that is not the one doing the
compiling — Chapter 23.

## A measured sweep

`examples/block_size.rs` compiles the vector-add kernel once per block size and
times each on 32M elements — three 128 MB buffers, far past any cache, so this
is memory and nothing else.

```bash
cargo run --release -p teeny-triton --features cuda --example block_size
```

On an **RTX 5070 (sm_120), CUDA 13.3, driver 610.43.02**:

| `BLOCK_SIZE` | threads | time | bandwidth | grid |
|---:|---:|---:|---:|---:|
| 32 | 128 | 1089.0 µs | 369.8 GB/s | 1048576 |
| 64 | 128 | 707.9 µs | 568.8 GB/s | 524288 |
| 128 | 128 | 676.6 µs | 595.1 GB/s | 262144 |
| 256 | 128 | 675.1 µs | **596.5 GB/s** | 131072 |
| 512 | 128 | 677.0 µs | 594.8 GB/s | 65536 |
| 1024 | 128 | 681.7 µs | 590.6 GB/s | 32768 |

Three things to read off it.

**The threads column never moves.** Six compilations, 128 threads every time.
This is the distinction at the top of the chapter, measured.

**32 is genuinely bad, and for a specific reason.** Each program has 128 threads
but only 32 elements to work on, so three quarters of every program's threads
have nothing to do. The bandwidth loss — 370 against 596 GB/s, a 38% drop — is
almost exactly that idle fraction.

**Everything from 128 up is the same.** 595, 596, 595, 591 GB/s: a 1% spread,
which is noise. Once each program has enough work to occupy its threads, this
kernel is limited by memory and nothing you do to the block size will change
that.

That flat plateau is the useful result. It says *stop tuning* — the kernel is at
the card's practical limit for this access pattern, and effort is better spent
somewhere else. Compare the plateau against your card's datasheet bandwidth; if
you are near it, you are done.

A sweep that looks like this is the common case for a simple memory-bound
kernel. Sweeps that do *not* plateau are the interesting ones.

## A procedure

Given no autotuner, this is the honest loop:

1. **Start at 128 or 256 threads**, a power of two, a multiple of 32.
2. **Get it correct.** Do not tune a kernel that is wrong.
3. **Sweep.** 64, 128, 256, 512. It is one constant and a rebuild each — the
   `id` from Chapter 8 keeps the compiled variants apart automatically.
4. **Measure properly.** Chapter 18. An unwarmed first run measures the
   compiler.
5. **Write the winner down, with the card it won on.** Six months later nobody
   remembers whether 256 was measured or guessed.
6. **Re-measure on a new target.** Chapter 24: correctness transfers between
   cards, performance does not.

For tiled kernels the sweep is two- or three-dimensional and gets expensive
quickly. Fix the tile shape from the problem — the fused conv kernels use 32 for
all three of `BLOCK_M`, `BLOCK_N`, `BLOCK_K` — then sweep the thread count
alone.

## Where the numbers in this tree came from

The shape-based dispatch thresholds in `kernels/teeny-kernels/src/graph/mod.rs`
— GEMM kernel for 1×1 convolutions with at least 32 output channels, tiled above
16 — are hand-picked constants from exactly this process. *Which* kernel runs is
still a hand-picked threshold; only the tile size inside it is now derived, and
only when `sm_count` is set.

The bench beside them exists to check they still hold. Its doc comment says so:
the shapes it benchmarks were chosen to straddle those thresholds, so the
measurements say whether the dispatch still picks the right kernel.

That is the pattern to copy. A tuned constant with no benchmark defending it
becomes folklore within a release or two.

Next: the choice that usually matters more than the block size.
