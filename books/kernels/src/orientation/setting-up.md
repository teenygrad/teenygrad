# Setting Up

Three things to install, in increasing order of how much trouble they are:

1. **Rust.** You have it.
2. **`teenyc`**, the compiler that turns kernel source into GPU code.
3. **The CUDA toolkit and a GPU**, to actually run anything.

You can do useful work with only the first. Each step below says what it buys
you, so you can stop when you have enough.

## Step 1 — Check out the workspace

```bash
git clone https://github.com/teenygrad/teenygrad
cd teenygrad
cargo check -p teeny-triton
```

If that succeeds, you can already write kernels and have the Rust compiler check
them. No GPU is involved: as Chapter 3 explained, kernel bodies are type-checked
by ordinary rustc, and only *compiling to PTX* needs anything special.

This is a real checkpoint, not a formality. Most kernel mistakes are type errors
— a mask that is not a mask, a pointer to the wrong dtype — and every one of
them is caught here.

## Step 2 — Install `teenyc`

`teenyc` is a modified rustc. It is distributed separately, and `cargo-teeny`
installs it:

```bash
cargo install --git https://github.com/teenygrad/cargo-teeny
cargo teeny install-toolchain
```

That downloads the toolchain package, checks its hash, unpacks it, and registers
it with rustup under a name containing `teenyc`. It is not made your default
toolchain, so nothing else you build is affected.

> `rustup toolchain install` cannot be used here. Rustup only accepts
> `stable`/`beta`/`nightly` or a version number as a toolchain name, and rejects
> anything else before it makes a network call — hence the separate command.

Check it landed:

```bash
rustup toolchain list | grep teenyc
rustup which --toolchain <the-name-you-saw> teenyc
```

The second command prints the binary's path. That is what `compile_kernel` will
find and run.

### When it cannot be found

`compile_kernel` looks in two places, in order:

1. `$TEENYC_PATH`, if set — an explicit path to the binary.
2. The one rustup toolchain whose name contains `teenyc`.

It deliberately does not fall back to a bare `teenyc` on your `$PATH`, because
that would work only by accident and fail confusingly. So there are exactly two
errors you can get:

```text
no teenyc rustup toolchain found; set TEENYC_PATH to the teenyc binary, or
install one with `cargo teeny install-toolchain` (see cargo-teeny)
```

```text
multiple teenyc rustup toolchains found (a, b); set TEENYC_PATH to disambiguate
```

Both are fixed by setting `TEENYC_PATH`.

The tree's tests and benches all call `dotenv().ok()` before doing anything, so
a `.env` file at the workspace root is the supported way to keep this set:

```bash
# .env
TEENYC_PATH=/home/you/.rustup/toolchains/stable-teenyc-x86_64-unknown-linux-gnu/bin/teenyc
TEENYC_CACHE_DIR=/home/you/.cache/teenyc
```

`TEENYC_CACHE_DIR` is where compiled PTX is kept. It defaults to
`/tmp/teenyc_cache`, which most systems clear on reboot — pointing it somewhere
durable saves recompiling.

## Step 3 — CUDA and a card

To run a kernel you need an NVIDIA GPU of compute capability **sm_75 or newer**
— Turing, from 2018, and anything since. That is the floor because Triton's
matrix acceleration needs it; sm_70 and sm_72 have only a deprecated
fused-multiply-add fallback path.

| Capability | Cards |
|---|---|
| `sm_75` | Turing: RTX 20xx, GTX 16xx, T4 |
| `sm_80` | Ampere datacenter: A100, A30 |
| `sm_86` | Ampere: RTX 30xx, A40, A10 |
| `sm_87` | Jetson Orin (AGX / NX / Nano) |
| `sm_89` | Ada Lovelace: RTX 40xx, L4, L40S |
| `sm_90` | Hopper: H100, H200 |
| `sm_100` | Blackwell datacenter: B100, B200, GB200 |
| `sm_120` | Blackwell: RTX 50xx |

You also need the CUDA toolkit — not just a driver. The `teeny-cuda` crate
generates its bindings from the toolkit's headers at build time, so without
`cuda.h` on the include path it fails to build at all:

```text
wrapper.h:17:10: fatal error: 'cuda.h' file not found
```

There is no feature flag that skips this. A machine without the toolkit cannot
build `teeny-cuda`, or anything that depends on it, which is why the workspace's
CI excludes those crates and why the book's examples are behind a `cuda`
feature.

Now run one:

```bash
cargo run -p teeny-triton --features cuda --example vector_add
```

The program opens the first device, prints its name and capability, compiles the
kernel for that exact card, and adds two vectors.

### If that fails

**`PTX .version 8.6 does not support .target sm_120a`** — a Blackwell card, where
`teenyc`'s default PTX version is newer than the driver accepts. Set
`TEENYC_PTX_VERSION=87`. This is a `teenyc`-side default; the SDK cannot work
around it.

You may well not hit it. On an RTX 5070 with CUDA 13.3 and driver 610.43.02,
everything in this book ran without the variable set. Newer drivers appear to
accept the version; try it plain first.

**A capability you want to override** — `TEENYC_CAPABILITY=sm_89` forces the
target, regardless of what the device reports. Useful for reproducing someone
else's build.

## What you have now

| After | You can |
|---|---|
| Step 1 | Write kernels and have them type-checked |
| Step 2 | Compile kernels to PTX and read the generated code |
| Step 3 | Run kernels and measure them |

Steps 1 and 2 cover Chapters 5 through 9 apart from the actual runs. Everything
in Part 4 needs Step 3, because you cannot optimise what you cannot time.

Next: the kernel itself.
