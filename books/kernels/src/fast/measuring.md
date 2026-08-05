# Measuring

Every chapter in this part has ended by telling you to measure. This is how, and
more importantly, how to get a number you can trust.

## Why this is harder than it looks

Four things will give you a wrong number, and the first two will give you one
that is wrong by an order of magnitude.

**The compile happens on the first call.** `compile_kernel` shells out to
`teenyc` and caches the result. Time that call and you have measured the
compiler. Compile outside the timing loop, always.

**Launches are asynchronous.** Handing a kernel to the driver returns before it
has run. A timing loop that launches and stops the clock has measured the cost
of asking, not of doing. You must synchronise before reading the clock.

**The first run is never representative.** Caches are cold, clocks have not
boosted, memory is not resident. Warm up.

**Clocks move.** GPUs throttle when hot and boost when idle. A benchmark run
immediately after another one is not measuring the same machine.

`criterion`, which this tree uses, handles warm-up and repetition and gives you a
distribution rather than a single number. It does not handle the first two — the
compile and the synchronisation are yours.

## The harness in this tree

`kernels/teeny-kernels/benches/conv2d_bn_silu.rs` is the pattern to copy. Its
structure:

**Compile once, outside the loop.**

```rust,ignore
let kernel = Conv2dBnSiluForward::new(kh, kw, stride_h, stride_w, pad_h, pad_w, 1, BLOCK_OW_SCALAR);
let ptx = std::fs::read(compile_kernel(&kernel, target, false)?)?;
let program = testing::load_program_from_ptx::<Conv2dBnSiluForward>(&ptx)?;
```

Note `force: false`. The cache is wanted here — you are not benchmarking
compilation.

**Allocate and fill once, outside the loop.**

```rust,ignore
let mut x_buf = device.buffer::<f32>(shape.nb * shape.c_in * shape.hh * shape.ww)?;
x_buf.to_device(&shape.x_host())?;
```

Host-to-device copies are slow and are not what you are measuring.

**Only the launch is inside.**

```rust,ignore
group.bench_function(format!("scalar/{}", shape.label), |b| {
    b.iter(|| { device.launch(&program, &cfg, (...)) })
});
```

**Use deterministic inputs.** The bench generates them arithmetically:

```rust,ignore
fn x_host(&self) -> Vec<f32> {
    (0..self.nb * self.c_in * self.hh * self.ww)
        .map(|i| (i as f32 % 17.0 - 8.0) * 0.1)
        .collect()
}
```

Not random. Two runs get identical data, and any difference between them is the
kernel.

## What to compare against

A number alone means nothing. `142 µs` is neither good nor bad.

**Against the alternative you would otherwise ship.** The fused kernel against
the three unfused ones. This is the comparison that decides whether the work was
worth it.

**Against the other implementations of the same thing.** The conv bench times
three kernels across shapes chosen to straddle the dispatch thresholds — so the
measurement answers "does the lowering still pick the right one?", not just "how
fast is this?".

**Against the hardware's limit.** For a memory-bound kernel, divide the bytes
moved by the elapsed time and compare with the card's peak bandwidth. At 80% you
are close to done. At 15% something is wrong, and Chapter 17 is where to look.
This is the most useful single check available, and it needs no reference
implementation.

## Choosing shapes

Benchmark the shapes you run, not round numbers.

Powers of two are the friendliest case: no ragged tail, no masked lanes, tiles
that divide evenly. A kernel benchmarked only at 1024×1024 can be much worse at
1000×1000, and 1000 is the realistic one.

The conv bench picks shapes deliberately either side of the dispatch thresholds,
which is the right instinct: benchmark where behaviour *changes*, not where it
is comfortable.

## Running it

```bash
cargo bench -p teeny-kernels --features cuda,training --bench conv2d_bn_silu
```

On Blackwell you may need the PTX-version workaround from Chapter 4:

```bash
TEENYC_PTX_VERSION=87 cargo bench -p teeny-kernels --features cuda,training --bench conv2d_bn_silu
```

`criterion` writes an HTML report and, on a second run, compares against the
previous one — which makes "did my change help?" a question it answers directly.

## Recording a result

A measurement without its context is not reproducible. Record:

- **The card**, by name and compute capability.
- **The shapes.**
- **The block sizes and tile shapes.**
- **The date.** Driver and toolchain versions move.

The format this book uses:

| Kernel | Shape | Block | Time | Card |
|---|---|---|---|---|
| *not yet measured* | | | | |

> **No measurements in this book are real yet.** This book has had no reference
> machine — every table like the one above is empty rather than filled with a
> plausible number, and Chapter 5's transcript is derived from the program's own
> arithmetic rather than captured from a run. When a machine exists, these
> tables get filled in and named. See item 8 in
> [`KNOWN-GAPS.md`](https://github.com/teenygrad/teenygrad/blob/main/books/kernels/KNOWN-GAPS.md).

An invented number is worse than a blank one. A blank prompts someone to
measure; an invented number gets quoted.

## Profiling

`criterion` tells you a kernel is slow. It does not tell you why.

For that you need a profiler — NVIDIA's Nsight Compute reports achieved
bandwidth, occupancy, warp stall reasons and instruction mix per kernel, which
is the level at which "why" gets answered.

Nothing in this SDK integrates with it, and no profile has been captured for
these kernels, so this book cannot teach reading one. It is item 9 in
`KNOWN-GAPS.md`.

What you *can* do without a profiler, and should do first:

1. **Compute achieved bandwidth by hand.** Bytes moved ÷ time. Compare with the
   card's specification.
2. **Read the MLIR.** Chapter 9. Count the loads and stores; check nothing is
   loaded twice.
3. **Vary one thing at a time.** Block size, tile shape, layout. The measurement
   tells you which mattered.

That covers most of what a first profiler session would have told you.

Next: the numbers themselves.
