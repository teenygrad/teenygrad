# What a Kernel Is

A kernel is a function that runs on a graphics card.

That is the whole definition. What makes it worth a book is everything that
follows from *where* it runs.

## Why a GPU is shaped the way it is

Your CPU has a handful of cores. Each one is enormously clever: it predicts
branches, reorders instructions, and keeps several megabytes of cache close by
so it rarely has to wait for memory. It is built to run one thread of control as
fast as physically possible.

A GPU gives up all of that. It has thousands of small, simple cores that run in
lockstep, and comparatively little cache. Any single one of them is far slower
than a CPU core. There are just a great many of them.

This is a good trade only when your work looks a particular way: the same
operation, applied to a lot of data, with no need to know one result before
computing the next. Adding two million-element vectors is the perfect shape.
Walking a linked list is the worst possible shape.

Machine learning is almost entirely the first shape, which is why it runs on
GPUs.

## What "kernel" means

Suppose you want to add two vectors of a million elements. On a CPU you write a
loop.

On a GPU you do not write the loop. You write the *body* of the loop — the part
that handles one piece of the work — and then you ask the card to run a million
copies of it at once. That function is the kernel.

The mental shift is that a kernel describes **one worker's job**, not the whole
job. Every copy runs the same instructions. The only thing that differs between
them is a number telling each copy which piece of the data is its own.

Everything else in this book is a consequence of that one idea.

## Why you would write your own

Libraries already contain fast kernels for the common operations. Matrix
multiply, convolution, softmax — those are solved, and yours will probably be
slower. So why write one?

Because of memory.

A GPU can do arithmetic far faster than it can fetch numbers to do arithmetic
on. On typical hardware the gap is more than an order of magnitude. That means
for a lot of real work, the arithmetic is free and the *loads and stores* are
the entire cost.

Now look at a sequence you find everywhere in vision models — a convolution,
then a batch normalisation, then a SiLU activation:

```text
conv       read input, write result   ← memory traffic
batchnorm  read result, write result  ← memory traffic
silu       read result, write result  ← memory traffic
```

Three separate library kernels means three round trips to memory. But the
batchnorm and the SiLU are a few arithmetic operations each — nothing. If you
write one kernel that reads the input, does all three steps while the numbers
are already in registers, and writes once, you have removed two thirds of the
memory traffic and almost none of the work.

That is called **fusion**, and it is the single most common reason to write a
kernel. teenygrad ships exactly this kernel — `conv2d_bn_silu` — for exactly
this reason.

The other reasons, in rough order of how often they come up:

- **The operation does not exist.** A new attention variant, a custom loss, an
  op from a paper published last month.
- **You know something the library cannot.** Your sequence length is always 512.
  Your weights are always sparse in a particular pattern. A general kernel
  cannot assume that; yours can.
- **You need it on hardware the library ignores.** Small embedded GPUs are often
  an afterthought upstream.

## The two costs, named

You will meet these terms constantly, so here they are once:

A kernel is **memory-bound** when it spends most of its time waiting for data.
Adding two vectors is memory-bound: three numbers moved for one addition. Making
it faster means moving fewer numbers, or moving them in a better order.

A kernel is **compute-bound** when it spends most of its time doing arithmetic.
A large matrix multiply is compute-bound: every number loaded gets used many
times over. Making it faster means using the hardware's specialised arithmetic
units properly.

The first question to ask about any kernel is which of the two it is, because
the answer decides what is worth optimising. Most kernels you write will be
memory-bound. Part 4 goes into this properly.

## What you are about to do

The next three chapters cover the programming model, what the toolchain does to
your code, and how to install it. Then you write a kernel and run it.

If you would rather see the thing before reading about it, the whole of Chapter
5 is one file, and on a machine with a GPU it runs now:

```bash
cargo run -p teeny-triton --features cuda --example vector_add
```
