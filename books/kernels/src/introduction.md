# Writing GPU Kernels in Rust

This book teaches you to write GPU kernels in Rust using teenygrad.

A kernel is a small program that runs on a graphics card. You write one when the
operation you need is not already fast — because it does not exist, or because
it exists as three separate operations that each read and write memory when one
could have done the job in a single pass.

Most GPU kernel work today happens in Python, through Triton. teenygrad gives
you the same programming model in Rust: you write a plain Rust function, and the
teenygrad toolchain turns it into machine code your GPU runs.

## Who this is for

You are comfortable in Rust. You have never written a GPU kernel, and you may
never have used CUDA either.

That is the whole prerequisite. Every GPU term is explained the first time it
appears. You should not have to open the Python Triton documentation to follow
any chapter here — if you do, that is a bug in this book, and there is an "Edit
this page" link at the bottom of every page.

## What you will be able to do

By the end of Part 2 you will have compiled and run your own kernel, and seen
the numbers it produced.

By the end of Part 3 you will have written a softmax, a matrix multiply, and a
kernel that fuses several operations into one pass over memory.

By the end of the book you will have measured a kernel against alternatives,
attached one to a model's computation graph, given it a backward pass so it can
be trained through, and built it for a different GPU.

## How the code in this book works

Every code sample in this book is real code from the teenygrad repository.
Nothing is retyped into the prose — the samples are pulled straight out of the
files, so a chapter cannot drift from code that builds.

They come from two places. The teaching examples are runnable programs under
`kernels/teeny-triton/examples/`. Later chapters teach from the library's own
kernels in `kernels/teeny-kernels/src/` instead, because a kernel that ships is
a better thing to learn from than a copy of one.

The examples you can run yourself:

```bash
cargo run -p teeny-triton --features cuda --example vector_add
```

The `cuda` feature is what says "I have a GPU and the CUDA toolkit". Without it
the examples are not built at all, so the rest of the workspace still compiles
on a laptop.

## A note on where this book is going

Parts 1 and 2 are the ones that matter most. If you cannot get from a clean
machine to a working kernel using only those chapters, the rest of the book has
not earned your time. They are written to be read in order, once.

Parts 3 to 6 are closer to reference material. Read the chapter you need.

> This book is being written. Chapters greyed out in the sidebar are drafted in
> [`OUTLINE.md`](https://github.com/teenygrad/teenygrad/blob/main/books/kernels/OUTLINE.md),
> alongside the exact API each one will use. Gaps between what the book wants to
> teach and what the SDK can currently do are recorded in
> [`KNOWN-GAPS.md`](https://github.com/teenygrad/teenygrad/blob/main/books/kernels/KNOWN-GAPS.md).
