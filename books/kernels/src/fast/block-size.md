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

For an elementwise kernel these coincide. `vector_add` with `BLOCK_SIZE = 128`
is launched with 128 threads, one per element, and that is the natural reading.

For a tiled kernel they do not. `conv2d_bn_silu` has `BLOCK_OW = 16` — sixteen
output columns per tile — and its bench launches 128 threads:

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

`RuntimeOp` exposes the same two through `block()` and `grid()`, and the helpers
in `teeny_cuda::testing` build a config for you:

| Helper | Use |
|---|---|
| `launch_config(n_elements, block_size)` | Explicit block size, grid from the element count |
| `launch_config_from_program(n, &program)` | Block size read from the compiled PTX |
| `launch_config_with_grid(grid_x, &program)` | Grid you computed, block from PTX |

The second is worth understanding. `teenyc` records the thread count it wants in
the PTX, as a `.reqntid` directive, and the loader parses it back out:

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
memory a kernel wants, its global scratch requirements, the thread count. They
are not exposed as a public API, so reading them means either the PTX itself or
`nvdisasm`.

**Maximum occupancy is not the goal.** A kernel at 50% occupancy with good
arithmetic intensity routinely beats one at 100% that is memory-starved.
Occupancy is a diagnostic, not a target.

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
16 — are hand-picked constants from exactly this process.

The bench beside them exists to check they still hold. Its doc comment says so:
the shapes it benchmarks were chosen to straddle those thresholds, so the
measurements say whether the dispatch still picks the right kernel.

That is the pattern to copy. A tuned constant with no benchmark defending it
becomes folklore within a release or two.

Next: the choice that usually matters more than the block size.
