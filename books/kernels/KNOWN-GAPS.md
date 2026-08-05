# Known gaps

Things this book wants to teach that the SDK cannot currently demonstrate, and
things in the tree that are wrong or unfinished in a way a book would have to
paper over.

Each entry names the chapter it blocks, what is missing, and — where one is
obvious — a suggested signature. Nothing here is worked around in prose. A
chapter is scoped down or dropped instead, and the reason said out loud.

Status legend: **blocks** a chapter cannot be written correctly until this is
resolved · **limits** the chapter can be written, but narrower than intended ·
**cosmetic** wrong but harmless.

---

## 1. `CustomOp::lower()`'s entry-point string is inconsistent — **blocks Ch 21**

**Where.** `core/teeny-core/src/graph/mod.rs:125` (the trait),
`kernels/teeny-kernels/src/graph/mod.rs:2645` (the consumer),
`vision-rs/src/models/yolo/kernels/detect_decode.rs:133` and
`.../attention/psa.rs:784,820,871,909` (the implementations).

**What.** The trait's doc comment says the third tuple element is "the PTX symbol
name, conventionally `{name}_entry_point`". Every built-in op in
`teeny-kernels` follows that: `entry_point: format!("{}_entry_point", name)`.
Every `CustomOp` in `vision-rs` returns the bare literal `"entry_point"`.

The value flows to `ExecutableOp::forward_kernel_entry_point()` and then to
`CompiledNode.entry_point` (`drivers/teeny-cuda/src/compiler/graph.rs:149`).
Separately, `KernelMetadata` parses the real entry name out of the PTX's
`.visible .entry` directive and falls back to the literal `"entry_point"`
(`drivers/teeny-cuda/src/device/program.rs:118-124`), and
`CudaCompilerOptions` defaults an `entry_point` field to `"entry_point"` too
(`drivers/teeny-cuda/src/compiler/options.rs:94`).

So there are two plausible readings: either the declared string is authoritative
and the `vision-rs` ops are wrong, or it is ignored in favour of the parsed PTX
symbol and the field is vestigial.

**Why it blocks.** Chapter 21 tells a reader what to return from their own
`lower()`. Getting this wrong produces a kernel that either fails to resolve at
load time or silently resolves to the wrong function.

**Cannot be settled by reading.** No GPU was available; the question is what the
loader does at runtime. Someone with a device should run a `vision-rs` YOLO
inference and a `teeny-kernels` graph model and see which path resolves.

**Suggested fix.** Drop the parameter and derive it. `Kernel::entry_point_name()`
already computes `format!("{}_entry_point", name)` from the name, and `lower()`
already returns the name:

```rust
fn lower(&self) -> Option<(String, String, Arc<dyn RuntimeOp>)>;
//                         name    source  runtime_op
```

One fewer thing for a kernel author to get wrong, and one fewer thing for this
book to explain.

---

## 2. There is no autotuner — **limits Ch 15, replaces the brief's Ch 15**

**Where.** Nothing in the tree. `kernels/teeny-kernels/src/math/matmul.rs`
contains Python `@autotune` / `Config` / `@heuristics` decorators as comments.

**What.** Python Triton's `@triton.autotune` sweeps a config list at runtime and
caches the winner per shape key. There is no equivalent. Block sizes are chosen
by hand and passed to `new()`.

**Effect on the book.** The autotuning chapter becomes a dtype-dispatch chapter.
Choosing a block size is taught as a measurement exercise in Ch 18 instead.

**Suggested shape, if it is ever built.** The `#[kernel]` macro already turns
const generics into runtime struct fields and already generates a `…Dispatch`
type. An autotuner could plausibly live there:

```rust
#[kernel(dtypes = [f32], autotune = [BLOCK_SIZE in [64, 128, 256, 512]])]
```

with the generated `…Dispatch::tune(dtype, &shape, &device)` caching a winner
per shape key.

---

## 3. `num_warps` and `num_stages` are not settable — **limits Ch 16, drops one chapter**

**Where.** `drivers/teeny-cuda/src/device/program.rs` (`KernelMetadata.num_warps`,
read from PTX), `drivers/teeny-cuda/src/testing/mod.rs:131`.

**What.** In Python Triton these are launch parameters: `num_warps` sets how many
warps cooperate on one program, `num_stages` sets the depth of the compiler's
software pipeline. Here `num_warps` is only ever *read back* from compiled PTX
metadata (derived from `.reqntid`, rounded up to whole warps), and `num_stages`
does not appear at all.

**Effect on the book.** The brief's "Warps, stages, and occupancy" chapter
becomes "Choosing a block size", scoped to `RuntimeOp::block`/`grid` and
`CudaLaunchConfig`. The brief's "Shared memory and software pipelining" chapter
is dropped entirely — neither is exposed.

---

## 4. `f16` and `bf16` cannot be monomorphized — **limits Ch 19**

**Where.** `macros/teeny-macros/src/macros/kernel.rs:83-96`
(`all_dtypes_for_bound`), whose comment says f16/bf16 "are marker-only and
cannot be monomorphized".

**What.** `DtypeRepr` has `F16` and `BF16` variants and `dtype_ident_to_repr`
maps them, so `#[kernel(dtypes = [f16])]` parses. But the implicit "all dtypes
for this bound" sets exclude them, because there is no concrete Rust impl to
monomorphize against.

**Effect on the book.** The numerics chapter can explain half precision as a
concept and show `Triton::cast`, but cannot show a working half-precision
kernel. Given that mixed precision is most of why people write kernels at all,
this is the largest single gap in the book.

**Unverified.** Whether `#[kernel(dtypes = [f16])]` fails at compile time or
produces something broken was not tested — it needs the `teenyc` toolchain.

---

## 4b. `#[kernel(dtypes = [...])]` has no in-tree user — **limits Ch 15**

**Where.** `macros/teeny-macros/src/macros/kernel.rs` generates the dispatcher;
nothing calls for it.

**What.** Grepping every kernel in `teeny-kernels`, `teeny-vision` and
`vision-rs` finds `#[kernel(backward = ...)]` in dozens of places and
`#[kernel(dtypes = [...])]` in none. The explicit dtype list appears only in doc
comments and in the macro's own implementation.

So the dispatcher is reached exclusively through the *implicit* path — "opt in
via `backward`, infer the dtype set from the trait bound". The explicit list
parses and generates code, but no test or kernel exercises it.

**Effect on the book.** Chapter 15 teaches the attribute because it is the
documented way to dispatch on a runtime dtype, and says plainly that nothing in
the tree uses it. If it turns out to be broken, the chapter is wrong.

**Suggested fix.** Either use it somewhere real, or add a test that does. A
generated API with no caller is a liability.

---

## 5. `math/matmul.rs` is not Rust — **cosmetic, but Ch 11 must not cite it**

**Where.** `kernels/teeny-kernels/src/math/matmul.rs`, 365 lines.

**What.** The entire file is commented-out Python Triton, including a large
`@autotune` config table. It exports nothing. The working matmul is
`math/gemm.rs` (`matmul_forward`, `matmul_backward_da`, `matmul_backward_db`);
the only tiled `T::dot` K-loop is in
`nn/fused/conv2d_bn_silu_gemm.rs`.

**Suggested fix.** Delete the file, or move it to a `reference/` directory that
is clearly not compiled. A reader who greps for "matmul" finds this first.

---

## 6. `kernel_group!` is documented but does not exist — **cosmetic**

**Where.** `core/teeny-core/src/model/mod.rs:32`, in `KernelInstance`'s doc
comment: "produced by a `#[kernel(dtypes = [...])]` dispatcher (or a
`kernel_group!`)".

**What.** No `kernel_group!` macro exists anywhere in the workspace. Either it
was removed or never landed. The reference-section list of macros a kernel
author touches must not include it.

---

## 7. `mdbook-linkcheck` does not support mdbook 0.5.x — **affects the build**

**Where.** `books/teenygrad/book.toml` already documents this: it fails parsing the
`RenderContext` with "missing field `sections`".

**What.** The brief asks for `mdbook-linkcheck` with the build failing on broken
links. It cannot be enabled.

**What is done instead.** The docs site build (`scripts/build-books.mjs` in
`teenygrad/docs`) reports every internal link that points at no chapter, and
Nitro's prerender fails on unresolvable routes. That covers internal links but
not external ones. Re-add `[output.linkcheck]` once upstream catches up.

---

## 8. No GPU was available — **limits Parts 3, 4 and 5**

**What.** Every timing table, every "show the output" block, and every claim
that an example *runs* is unverified.

It is worse than it sounds, because the examples cannot be compiled either.
They live in `kernels/teeny-triton/examples/` behind `required-features =
["cuda"]`, and `teeny-cuda`'s build.rs runs bindgen against the CUDA headers
with no feature gate to skip it — so a machine without the toolkit cannot build
them, and neither can CI (the same reason `ci.yml` excludes `teeny-cuda` and
`teeny-kernels`). What CI does check is that every `{{#include}}` still resolves
(`books/check-includes.py`), which catches a rename but not a type error.

Chapter 5's transcript is derived from the program's own `println!`s and its
arithmetic, and says so on the page. No number in this book is invented, but
until someone runs these on a card, none is confirmed either.

**What is needed.** One reference machine, named in the book, with its SM
capability recorded — the benches in `kernels/teeny-kernels/benches/` already
assume this and document an sm_120 PTX-version workaround
(`TEENYC_PTX_VERSION=87`) that nobody can confirm without the hardware.

---

## 9. Nsight Compute integration — **limits Ch 18**

**What.** The brief asks for a chapter reading Nsight Compute output. Nothing in
the tree integrates with Nsight, so Chapter 18 teaches `criterion` and
computing achieved bandwidth by hand.

**Partly closed.** Nsight *has* been used on these kernels — the shape-adaptive
conv tile-size work measured achieved occupancy as low as 8% against a
theoretical 100% on YOLO26n, which is what motivated `Options::sm_count`.
Chapter 18 now quotes that finding as a worked example of what a profiler
reports that a benchmark cannot.

What is still missing is a captured profile someone can walk through
section by section, and any tooling that makes producing one routine.
