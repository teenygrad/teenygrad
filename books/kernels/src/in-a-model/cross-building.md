# Building for Another Target

The machine you develop on and the machine that runs your model are often not
the same. A Jetson on a robot has an Arm CPU, a different GPU generation, and no
appetite for compiling anything.

Two separate problems, and it helps to keep them apart:

- **The CPU code** — your program — must be compiled for the board's
  architecture.
- **The GPU code** — your kernels — must be compiled for the board's compute
  capability.

`cargo-teeny` handles both, with a different subcommand each.

## Capability, not architecture

Chapter 4 listed the compute capabilities. The one that matters here is `sm_87`,
Jetson Orin — because it is the case where a developer's desktop and the target
differ in a way that silently produces a binary that will not run.

PTX gives you some slack. It is an intermediate form, and the driver compiles it
to machine code at load time, so PTX built for an older capability generally
runs on a newer card. What it will not do is use instructions the older
capability did not have. Build for `sm_75` and run on an `sm_90` and you get a
working kernel that leaves the newer Tensor Cores idle.

So: build for the capability you will run on.

```rust,ignore
let target = Target::new(Capability::Sm87);
let ptx_path = compile_kernel(&kernel, &target, false)?;
```

Nothing about this needs an `sm_87` device present. Compiling for a capability
and having one are unrelated — which is what makes cross-building possible at
all.

## Cross-compiling the program

```bash
cargo teeny build --target jetson-orin-nano
cargo teeny check --target jetson-orin-nano          # faster feedback
cargo teeny clippy --target jetson-orin-nano
cargo teeny build --target jetson-orin-nano --example yolo26
```

This delegates to [`cross`](https://github.com/cross-rs/cross), which builds
inside a container holding the target's toolchain, and handles two things
`cross` does not do on its own:

- It resolves the teenygrad workspace root from your `[patch.crates-io]` entries
  and mounts it, because `cross` mounts individual crate directories and that is
  not enough for workspace inheritance.
- It mounts the host's CUDA aarch64 target directory where the board's image
  expects it.

`cargo teeny check --target jetson-orin-nano` is a genuinely useful thing to run
on a laptop. It type-checks everything, including your kernels, for a board you
do not own.

If the board needs libraries you do not have locally, `cargo teeny sysroot` lays
out an FHS-style tree and can `rsync` the real thing off the device:

```bash
cargo teeny sysroot --host aarch64-unknown-linux-gnu --path ./sysroot \
  --type jetson-orin-nano --rsync-from ubuntu@jetson
```

## Compiling kernels ahead of time

Chapter 3 said kernel compilation happens at run time, on the first call, cached
on disk. On a development machine that is a one-off cost you never notice.

On a deployed board it is a problem. The first inference pays for compiling every
kernel in the model, `teenyc` has to be installed on the board, and a read-only
or space-constrained filesystem may not have anywhere to put a cache.

Ahead-of-time compilation moves that to build time:

```bash
cargo teeny aot --example yolo26 --device cuda --options "capability=sm_87,ptx-version=82"
```

The mechanism is worth understanding, because it is unusual. `aot` builds your
binary **for the host** — not cross-compiled — and *runs* it, with flags telling
it to compile its kernels for the named capability and stop. So the program
itself does the compiling; `cargo-teeny` forwards `--device`, `--options`,
`--cache-dir` and `--force` verbatim without parsing them.

What lands in the cache directory is PTX for `sm_87`, produced on your desktop.

## Packaging and deploying

```bash
cargo teeny package    # cross-compiles the binary + AOT-compiles its kernels
cargo teeny deploy     # ships the result over SSH
```

`package` runs both of the previous steps in one go, forcing the AOT cache
directory to `<dest>/cache` so the layout is right — `cache/` beside `bin/`.

That layout is not arbitrary. `default_cache_dir` looks for a `cache/` directory
next to the executable and uses it if present, falling back to
`TEENYC_CACHE_DIR` or `/tmp/teenyc_cache`. So a packaged binary finds its
precompiled kernels with no environment variables set — and a normal `cargo run`
during development, whose executable lives under `target/debug/` with no `cache/`
sibling, is unaffected.

## What to check before you ship

**The capability matches.** Building `sm_89` PTX for an Orin gets you a runtime
failure, not a compile error.

**The PTX version is one the board's driver accepts.** This is the failure that
catches people out. On some Blackwell cards `teenyc`'s default PTX version is
rejected outright:

```text
PTX .version 8.6 does not support .target sm_120a
```

The fix is `TEENYC_PTX_VERSION=87`, or `ptx-version=` in the AOT options. It is
a `teenyc`-side default, not something the SDK can work around, and this tree's
benches carry a note about it.

**The cache shipped.** A packaged binary with an empty `cache/` will try to
compile at run time and fail, because `teenyc` is not installed on the board.

**The dtypes match.** A model quantised on the host must be loaded as the dtype
it was written as. Chapter 19 covers what silently goes wrong when it is not.

Next: how much of any of this transfers.
