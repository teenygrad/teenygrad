# Vector Add, End to End

This chapter runs a kernel you wrote and gets numbers out of a GPU.

It is the whole thing at once: the kernel, compiling it, moving data to the
card, launching, and reading the answer back. Some of it will not make sense
yet. That is deliberate — you will get more out of the explanations if you have
already seen the thing they explain. Every part is named below with the chapter
that takes it apart.

The job is the simplest one that is still real: add two vectors, element by
element.

```text
a    = [0, 1, 2, 3, ...]
b    = [0, 2, 4, 6, ...]
out  = [0, 3, 6, 9, ...]
```

## The kernel

```rust
{{#include ../../../../kernels/teeny-triton/examples/vector_add.rs:kernel}}
```

Read it as a description of what **one** program does, not what the whole GPU
does. Many copies of this function run at once, and each one handles a different
slice of the vector. `program_id` is how a copy finds out which slice is its
own. Chapter 2 explains why the model works this way; Chapter 6 covers
`program_id` and block sizes properly.

Four things are worth naming now.

**`T: Triton` is the GPU.** Every operation the kernel can do is a method on it —
`T::load`, `T::store`, `T::arange`. Writing them through a type parameter rather
than calling free functions is what lets the compiler check a kernel before it
ever reaches a card.

**`BLOCK_SIZE` is a const generic, not an argument.** It is fixed when the kernel
is built, not when it is launched, so the compiler can use it — to unroll loops,
to pick register counts. Chapter 15 covers what else you can specialise this
way.

**`offsets` is a tensor, not a number.** `T::arange(0, BLOCK_SIZE)` produces
`BLOCK_SIZE` values at once, and `+ block_start` shifts all of them. You are
writing whole-block operations, which is the single biggest difference from
CUDA. Chapter 7 covers this properly.

**The mask is not optional.** 1000 elements in blocks of 128 is seven full
programs and a ragged eighth, which would otherwise read and write 24 elements
past the end of your buffers. `in_bounds` marks which lanes are real, and
`T::load` and `T::store` skip the rest. Chapter 7 shows what happens without it.

The three `where` clauses are the price of indexing memory at all. They are
identical in every kernel in this book, and you can copy them without
understanding them today — Chapter 8 explains what they are for.

## Running it

The kernel is one half. The other half runs on the CPU: it compiles the kernel
for your card, allocates memory on the device, copies data across, launches, and
copies the answer back.

```rust
{{#include ../../../../kernels/teeny-triton/examples/vector_add.rs:run}}
```

Six steps, in order:

1. **Ask the device what it is.** `setup_cuda_env` opens the first GPU and reads
   its compute capability — `sm_89`, `sm_120`, whatever you have. The kernel is
   then compiled for that card rather than for a guess.
2. **Build the kernel.** `VectorAdd::<f32>::new(BLOCK_SIZE)` is a type you never
   wrote. `#[kernel]` generated it from your function. Chapter 8 shows exactly
   what it generated and why.
3. **Compile it.** `compile_kernel` produces a PTX file — the assembly language
   NVIDIA cards accept. This is where your function stops being Rust. Chapter 9
   opens the PTX and reads it.
4. **Allocate and copy.** GPU memory is separate from your program's memory.
   `device.buffer` reserves space on the card; `to_device` copies into it.
5. **Launch.** The grid says how many programs to start. 1000 elements in blocks
   of 128 rounds up to 8. The tuple is the kernel's arguments, in the order the
   function declares them.
6. **Copy back and check.** `to_host` brings the answer into ordinary Rust
   memory, where it is just a `Vec<f32>`.

## Run it yourself

You need the `cuda` feature and a card of compute capability sm_75 or newer —
Turing, from 2018, or anything since. Chapter 4 covers getting the toolchain
installed.

```bash
cargo run -p teeny-triton --features cuda --example vector_add
```

On an RTX 5070:

```text
[1/9] CUDA available
[2/9] found 1 device(s)
[3/9] device: NVIDIA GeForce RTX 5070 (capability: sm_120)
compiled vector_add → /tmp/teenyc_cache/vector_add_5f69418a643d1353dba2ce66de8ed3dc4e1644c0d9474da4517ed6e7d3f67ff9.o
      loading PTX directly via driver JIT...
[CUDA-JIT] info: ptxas warning : .loc directive without .file directive is found, line information in generated binary may not be complete

[7/9] loaded PTX: module=0x60834a26a7e0 function=0x60834a958fd0 num_warps=4 num_ctas=1
launching 8 programs of 128 threads
out[0]   = 0
out[1]   = 3
out[999] = 2997
all 1000 elements match the CPU result
```

`out[i]` is `3i`, because `a[i] + b[i]` is `i + 2i`. The numbered lines and the
`ptxas` warning come from the setup helpers, not from this program.

Three things in that output are worth noticing now, and are explained later:

- **`num_warps=4`.** 128 threads is four warps of 32. You did not choose that
  number — it was read back out of the compiled PTX. Chapter 16.
- **The cache filename is a hash**, not the readable kernel id. Chapter 8.
- **The extension is `.o`, but the file is PTX text.** Chapter 9 opens it.

## What you just did

You wrote a function that never ran in your process. Its source text was
captured by a macro, compiled by a different compiler into GPU assembly, loaded
onto a card, and executed a thousand times over in parallel.

If that sequence sounds strange, it is, and it is worth understanding before you
write a second kernel. Chapter 3 explains it.

## What is missing

This kernel runs on its own. It is not part of a model, it has no gradient, and
nothing chose its block size but you.

None of that is needed to run a kernel, which is why none of it is in this
chapter. Part 5 attaches a kernel to a model's graph and gives it a backward
pass. Chapter 6 is about choosing that block size deliberately.

Next, though: what one program actually is, and how it finds its slice.
