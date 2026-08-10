# Outline — Writing GPU Kernels in Rust

The proposed shape of the book, with the exact API each chapter uses. Written
after reading `core/teeny-core`, `kernels/teeny-triton`, `kernels/teeny-kernels`,
`macros/teeny-macros`, `compiler/teeny-compiler`, `drivers/teeny-cuda`, the
`cargo-teeny` repo, and the `CustomOp` implementations in `vision-rs`.

Everything named under **Uses** exists in the tree today, at the path given.
Where a chapter needs something that does not exist, it is not written around —
it is recorded in [`KNOWN-GAPS.md`](./KNOWN-GAPS.md) and the chapter is scoped to
what the SDK can actually demonstrate.

**Where the examples live.** `kernels/teeny-triton/examples/`, one runnable
program per chapter, each behind `required-features = ["cuda"]`. Chapters pull
the parts they discuss out of those files by anchor. See
[`CONTRIBUTING.md`](./CONTRIBUTING.md).

**Not verified by running.** No CUDA GPU was available while this outline was
written, so every claim here comes from reading source, tests and snapshots.
Timing tables stay empty until they are measured on named hardware, and any
transcript derived rather than captured says so on the page.

---

## The skeleton every chapter builds on

The smallest complete working thing in this repo is **not** what the original
brief assumed. A kernel does not need a graph `Op` or a `Lowering` to run. Four
pieces are enough, and all four are in
`kernels/teeny-kernels/tests/test_elemwise_add.rs`:

1. **The kernel body** — a generic `fn` over `T: Triton`, annotated `#[kernel]`
   (`kernels/teeny-kernels/src/nn/tensor/elemwise_add.rs`).
2. **The generated struct** — `#[kernel]` emits `ElemwiseAddForward<D>` with a
   `new(block_size)` constructor and a `Kernel` impl.
3. **Compilation** — `compile_kernel(&kernel, &Target::new(capability), force)`
   returns a path to PTX (`teeny_cuda::compiler`).
4. **The launch** — `device.launch(&program, &cfg, (ptrs…, scalars…))` with
   buffers from `device.buffer::<f32>(n)`.

`CustomOp` / `RuntimeOp` are what you add *later*, to put a kernel inside a
model's graph. Deferring them to Part 5 is therefore correct, and it means Part
2 gets the reader to a running kernel far sooner than the original plan allowed.

The other structural fact that shapes the whole book: **`#[kernel]` captures your
function's source as a string.** `quote!(#vis #sig #block).to_string()` is stored
on the generated struct, wrapped in a generated `extern "C"` entry point, and
handed to an external compiler binary (`teenyc`) to compile. Your function is
never called by your program. Almost every constraint in Parts 2 and 3 follows
from this, so Chapter 3 explains it before any kernel is written.

---

## Part 1 — Orientation

### 1. What a Kernel Is

**Goal.** The reader can say what a GPU kernel is, and name a case where writing
one is the right move.

**Introduces.** GPU vs CPU; kernel; launch; memory-bound vs compute-bound; why
fusing operations saves time.

**Uses.** No API. Motivating example is the three-operation
conv → batchnorm → SiLU sequence that
`kernels/teeny-kernels/src/nn/fused/conv2d_bn_silu.rs` collapses into one pass.

**Example.** None — prose only.

### 2. You Program a Block, Not a Thread

**Goal.** The reader can predict what one program in a launch is responsible for.

**Introduces.** Program (Triton's unit of work); block; grid; the contrast with
the CUDA SIMT model of one thread per element; why block sizes are powers of two.

**Uses.** `Triton::program_id(Axis::X)`, `Triton::num_programs`, `Axis`
(`kernels/teeny-triton/src/triton/mod.rs`).

**Example.** None — a diagram and a walk through `elemwise_add_forward`'s first
three lines.

**Note.** The word "thread" still appears in this SDK's API —
`RuntimeOp::block()` is documented as "threads-per-CTA". The chapter has to name
that seam rather than pretend the CUDA layer is invisible.

### 3. From Rust to PTX

**Goal.** The reader understands that the kernel body is compiled as *text* by a
separate compiler, and can say what each stage does.

**Introduces.** `teenyc`; MLIR; the Triton dialect and its passes; LLVM IR; PTX;
SM capability; the kernel cache.

**Uses.** `teeny_macros::kernel` (source capture and entry-point generation,
`macros/teeny-macros/src/macros/kernel.rs`); `teeny_compiler::compiler::find_teenyc`;
`default_cache_dir`; `$TEENYC_PATH`, `$TEENYC_CACHE_DIR`, `$TEENYC_PTX_VERSION`;
`Capability` (`drivers/teeny-cuda/src/compiler/target.rs`: `Sm75`…`Sm120`);
`kernels/teeny-triton/build.rs`, which embeds the DSL's own source as the
`TRITON` string the kernel is compiled against.

**Example.** None — one mermaid diagram, plus the real MLIR from
`kernels/teeny-kernels/tests/snapshots/` shown as evidence rather than magic.

### 4. Setting Up

**Goal.** From a clean machine to a passing test.

**Introduces.** The toolchain split: normal `cargo` for your crate, `teenyc` for
kernel bodies.

**Uses.** `cargo install --git https://github.com/teenygrad/cargo-teeny`;
`cargo teeny install-toolchain`; `rustup which --toolchain <name> teenyc`;
`TEENYC_PATH` to disambiguate; `dotenv` (every test in the tree calls
`dotenv().ok()`, so a `.env` is the supported way to set these);
the `cuda` and `training` cargo features on `teeny-kernels`.

**Example.** None of its own. The chapter's checkpoint is `cargo check -p
teeny-triton` (proves the toolchain), then Chapter 5's `vector_add` example
(proves the card).

**Note.** Verifying "a working GPU" needs a GPU. That check is a separate,
clearly marked step, and the chapter must state which parts a reader on a laptop
can complete.

---

## Part 2 — Your First Kernel

### 5. Vector Add, End to End

**Goal.** The reader runs a kernel they wrote and gets numbers out.

**Introduces.** The four pieces of the skeleton above; `T::Pointer<D>`; const
generic block size; the entry point.

**Uses.** `#[kernel]`; `Triton::program_id`, `arange`, `load`, `store`;
`AddOffsets::add_offsets`, `Comparison::lt` (`core/teeny-core/src/dtype/mod.rs`);
`Kernel::{source, name, id, entry_point_name}`;
`compile_kernel`; `Target::new`; `Device::buffer`, `Buffer::{to_device, to_host}`,
`Device::launch`; `teeny_cuda::testing::{setup_cuda_env, load_program_from_ptx,
launch_config_from_program}`.

**Example.** `teeny-triton --example vector_add` — the whole thing in one file, then run.

**Note.** The chapter says plainly which lines will not make sense yet, and
names the chapter that explains each. This is the chapter that decides whether
the reader continues.

### 6. The Kernel Body

**Goal.** The reader can choose a grid and a block size for a new kernel and say
why.

**Introduces.** Mapping a program id to a slice of data; `cdiv`; why the grid is
computed on the host and the block size is baked in at compile time.

**Uses.** `Triton::{program_id, num_programs, cdiv}`; const generic parameters
and how `#[kernel]` turns them into runtime struct fields (lowercased — `const
BLOCK_SIZE` becomes `self.block_size`); `n.div_ceil(block)` on the host side.

**Example.** `teeny-triton --example block_size` — one kernel, three block sizes, same
answer.

### 7. Loads, Stores, and Masks

**Goal.** The reader can read and write memory safely when the data does not
divide evenly by the block size.

**Introduces.** Pointer arithmetic on tensors of pointers; masking; the fill
value for masked lanes; what happens without a mask.

**Uses.** `Triton::load` (all eight parameters, and which ones you can ignore);
`Triton::store`; `Triton::{zeros, full}`; `AddOffsets::add_offsets`;
`Comparison::{lt, ge}`; `Triton::where_`.

**Example.** `teeny-triton --example masking` — the same kernel with and without the mask,
on a length that is not a multiple of the block size.

### 8. What `#[kernel]` Generates

**Goal.** The reader can read the macro's output and debug it.

**Introduces.** The generated struct and its fields; the `Kernel` trait; the
`extern "C"` entry point and the `{name}_entry_point` symbol; the kernel id as a
cache key; `PhantomData` on dtype parameters.

**Uses.** `macros/teeny-macros/src/macros/kernel.rs` in full;
`teeny_core::device::program::{Kernel, KernelArg, KernelArgs, ArgVisitor}`;
`Kernel::{id, source, kernel_source, entry_point_source, entry_point_name}`;
`cargo expand` to show the real output.

**Example.** `teeny-triton --example expanded` — a two-line kernel and its expansion,
included side by side.

**Note.** This chapter is where the argument-order contract gets stated: the
tuple passed to `launch` must match the kernel's parameter list positionally,
and nothing checks it for you. See [`API-FRICTION.md`](./API-FRICTION.md).

### 9. Compiling and Reading the Output

**Goal.** The pipeline stops feeling like magic.

**Introduces.** Reading MLIR; reading PTX; the snapshot-test pattern as a way to
see what your change did to the generated code.

**Uses.** `compile_kernel(&kernel, &target, force)` and the `.mlir` file it
leaves beside the PTX; `insta::assert_debug_snapshot!`; `LlvmCompiler::new`,
`with_target_cpu`, `with_ptx_version`; `$TEENYC_PTX_VERSION` and the sm_120
driver-rejection case documented in
`kernels/teeny-kernels/benches/conv2d_bn_silu.rs`.

**Example.** `teeny-triton --example reading_ptx` — compile, print the MLIR, find the
loads and stores in it.

---

## Part 3 — Real Patterns

### 10. Softmax: Your First Reduction

**Goal.** The reader can write a kernel where programs cooperate across a row.

**Introduces.** Reduction; numerical stability (subtract the max); one program
per row; the `BLOCK_SIZE == n_cols` constraint and why it exists.

**Uses.** `Triton::{softmax, max, sum, exp}`;
`kernels/teeny-kernels/src/nn/activation/softmax.rs` as the reference
implementation.

**Example.** `teeny-triton --example softmax` — hand-rolled max/exp/sum first, then the
`T::softmax` builtin, with both outputs compared.

**Note.** The in-tree kernel requires the caller to round `n_cols` up to a power
of two and pass it as `BLOCK_SIZE`. That is a real constraint on a real kernel
and the chapter states it rather than hiding it.

### 11. Matrix Multiplication

**Goal.** The reader can write a tiled matmul and explain the accumulator.

**Introduces.** Tiling; the K loop; accumulators; Tensor Cores; block-size
tradeoffs.

**Uses.** `Triton::dot` with its `acc` parameter; `InputPrecision::{TF32,
TF32x3, IEEE}`; `Triton::{expand_dims, broadcast_to, permute}`;
`kernels/teeny-kernels/src/nn/fused/conv2d_bn_silu_gemm.rs` (the only real
`T::dot` tiled loop in the tree) and `kernels/teeny-kernels/src/math/gemm.rs`.

**Example.** `teeny-triton --example matmul` — square f32 matmul, checked against a CPU
reference.

**Note.** `kernels/teeny-kernels/src/math/matmul.rs` is 365 lines of
commented-out Python and contains no Rust. It is not a reference for anything
and the book must not cite it. Recorded in `KNOWN-GAPS.md`.

### 12. Fusing an Epilogue

**Goal.** The reader can fold cheap elementwise work into a kernel that has
already loaded the data.

**Introduces.** Epilogue; arithmetic intensity; why a fused kernel beats three
kernels even when the arithmetic is identical.

**Uses.** The three `conv2d_bn_silu` variants
(`kernels/teeny-kernels/src/nn/fused/`: scalar, `_tiled`, `_gemm`) and the
shape-based dispatch thresholds in `kernels/teeny-kernels/src/graph/mod.rs`.

**Example.** `teeny-triton --example fused_epilogue` — matmul, then matmul + bias + ReLU
in one kernel.

### 13. Reductions and Scans

**Goal.** The reader can pick the right reduction and write a custom one.

**Introduces.** Reduction vs scan; associativity; tie-breaking.

**Uses.** `Triton::{sum, max, min, max_with_indices, min_with_indices, argmax,
argmin, xor_sum, cumsum, cumprod, sort, histogram}`; `Triton::reduce` and
`associative_scan` with a `fn` pointer combine function.

**Example.** `teeny-triton --example reductions` — argmax, cumulative sum, and one custom
`reduce`.

**Note.** `reduce`/`associative_scan` take a plain `fn` pointer, not a closure —
the combine function must be statically known because it is compiled as source
text. That is worth a callout; it is where Python Triton users will trip.

### 14. Atomics

**Goal.** The reader knows when an atomic is necessary and what it costs.

**Introduces.** Race; read-modify-write; memory ordering; scope; scatter-add.

**Uses.** `Triton::{atomic_add, atomic_max, atomic_min, atomic_and, atomic_or,
atomic_xor, atomic_xchg, atomic_cas}`; `MemSem`; `MemScope`; real uses in
`kernels/teeny-kernels/src/nn/loss/nll.rs`, `loss/ranking.rs`, `conv/conv2d.rs`.

**Example.** `teeny-triton --example atomics` — a histogram written two ways, one wrong.

### 15. Compile-Time Parameters and Dtype Dispatch

**Goal.** The reader can write one kernel that serves several dtypes and several
block sizes.

**Introduces.** Monomorphization as specialisation; runtime dtype → compiled
kernel; why the dtype set is closed.

**Uses.** `#[kernel(dtypes = [f32, f64])]`; the generated `…Dispatch` struct,
its `SUPPORTED_DTYPES` and `dispatch(dtype, …)`; `DtypeRepr`;
`teeny_core::model::{KernelInstance, KernelInstanceBackward}`; the
`Dtype`/`Num`/`Int`/`Float`/`Bool` bounds and the implicit "all dtypes for the
bound" rule in `all_dtypes_for_bound`.

**Example.** `teeny-triton --example dispatch` — one kernel, dispatched at runtime for
`f32` and `f64`.

**Note.** This chapter replaces the brief's "autotuning" chapter. There is no
autotuner in this SDK — see `KNOWN-GAPS.md`. Choosing a block size is manual, and
Chapter 18 shows how to measure the choice.

---

## Part 4 — Making It Fast

### 16. Choosing a Block Size

**Goal.** The reader can set a launch configuration deliberately.

**Introduces.** CTA; occupancy; what the block size does and does not control
here.

**Uses.** `RuntimeOp::{block, grid}`; `CudaLaunchConfig`;
`teeny_cuda::testing::{launch_config, launch_config_from_program,
launch_config_with_grid}`; `program.metadata.num_warps` as a value you *read*.

**Example.** `teeny-triton --example launch_config` — one kernel, a sweep of block sizes,
a table of results.

**Note.** This chapter is deliberately narrower than the brief's "warps, stages
and occupancy". Triton's `num_warps` and `num_stages` are not settable from this
API — `num_warps` is only ever read back from compiled PTX metadata. Recorded in
`KNOWN-GAPS.md`. Software pipelining gets no chapter for the same reason.

### 17. Memory Coalescing and Tensor Layout

**Goal.** The reader can look at an indexing expression and say whether it will
be fast.

**Introduces.** Coalescing; strides; row-major layout; NCHW vs NHWC and what
each costs.

**Uses.** The indexing in `nn/conv/conv2d.rs` and
`nn/tensor/channel_bias_add.rs` (NC layout, `N = B*H*W`);
`RuntimeOp::forward_output_row_stride` and the TMA 16-byte alignment note;
`Triton::{make_block_ptr, advance, make_tensor_descriptor}`.

**Example.** `teeny-triton --example layout` — the same transpose read two ways.

### 18. Measuring

**Goal.** The reader can produce a number they trust.

**Introduces.** Warm-up; variance; what to compare against; why a first run is
never the number.

**Uses.** `criterion`; `kernels/teeny-kernels/benches/conv2d_bn_silu.rs` as the
in-tree pattern; the `force` flag on `compile_kernel` and the kernel cache's
effect on timing.

**Example.** A criterion bench over the Chapter 7 kernel. Benches are `benches/`
targets, not examples — `kernels/teeny-kernels/benches/conv2d_bn_silu.rs` is the
pattern to follow, including its `TEENYC_PTX_VERSION` note.

**Note.** Nsight Compute is out of scope until someone has run it against these
kernels and can show real output. Listed in `KNOWN-GAPS.md`.

### 19. Numerics

**Goal.** The reader can choose dtypes and accumulators without silently losing
precision.

**Introduces.** f32/f64/tf32/f16/bf16; accumulator width; what "numerically
stable" means concretely.

**Uses.** `teeny_core::dtype::{Dtype, Num, Int, Float, Bool}`; `DtypeRepr`;
`Triton::dot`'s separate input and output dtype parameters (`D` and `O`);
`InputPrecision`; `FpDowncastRounding`; `Triton::{cast, fdiv, div_rn, sqrt_rn,
fma}`.

**Example.** `teeny-triton --example numerics` — a sum that loses precision, and the fix.

**Note.** `f16` and `bf16` are marker-only in `all_dtypes_for_bound` — they
cannot be monomorphized, so a `#[kernel]` cannot dispatch to them today. That
limits this chapter and is recorded in `KNOWN-GAPS.md`.

---

## Part 5 — Kernels in a Real Model

### 20. Your Kernel as a Graph Op

**Goal.** The reader can make their kernel a node in a model's graph.

**Introduces.** The computation graph; symbolic vs concrete shapes; shape
inference.

**Uses.** `teeny_core::graph::{CustomOp, CustomData, Op, Graph, Shape,
SymTensor}`; `SymTensor::{input, record_custom}`; `CustomOp::{name,
infer_output_shape, as_any, lower}`; `TritonLowering`
(`kernels/teeny-kernels/src/graph/mod.rs`) and its `Op::Custom` arm;
`vision-rs`'s `DetectDecodeOp` as the worked reference.

**Example.** `teeny-triton --example custom_op` — the Chapter 7 kernel, as a graph op.

### 21. Wiring the Runtime

**Goal.** The reader can pack arguments and compute a grid for their op.

**Introduces.** Activation inputs vs parameters; parameter buffers and their
initial data; multi-launch ops.

**Uses.** `RuntimeOp` in full: `n_activation_inputs`, `param_shapes`,
`param_names`, `param_init_data`, `compute_concrete_output_shape`, `pack_args`,
`block`, `grid`, `n_launches`, `pack_args_for_launch`, `grid_for_launch`;
`ArgVisitor::visit_*`; `RawPtr`; `KernelExecutable` and `ExecutableOp`.

**Example.** `teeny-triton --example runtime_op` — an op with a constant parameter buffer,
uploaded via `param_init_data`.

**Note.** The `entry_point` string returned from `CustomOp::lower()` is
inconsistent between the in-tree lowering and the `vision-rs` implementations.
The chapter must state which is correct — see `KNOWN-GAPS.md`, item 1. **This is
the one item that should be resolved before Chapter 21 is written.**

### 22. Training: The Backward Kernel

**Goal.** The reader can make their op differentiable.

**Introduces.** Gradient; the backward pass; why the backward kernel needs the
forward's saved output.

**Uses.** `#[kernel(backward = …)]`; `LoweringMode::{Inference, Training}`;
`RuntimeOp::{has_backward, pack_backward_args, backward_block, backward_grid,
backward_grad_output_row_stride, n_backward_launches}`;
`CustomOp::lower_backward_source`; the `training` cargo feature;
`elemwise_add_backward` and `softmax_backward` as references.

**Example.** `teeny-triton --example backward` — forward and backward for one op, gradient
checked numerically.

### 23. Building for Another Target

**Goal.** The reader can build their kernel for a board they do not have in
front of them.

**Introduces.** Cross-compilation; ahead-of-time kernel compilation; sysroot;
deployment.

**Uses.** `Capability::{Sm75, Sm80, Sm86, Sm87, Sm89, Sm90, Sm100, Sm120}` and
`Capability::from_device_info`; `cargo teeny build --target jetson-orin-nano`;
`cargo teeny sysroot`; `cargo teeny aot`; `cargo teeny package`; `cargo teeny
deploy`; `teeny_cuda::compiler::aot`; the `cache/` layout `default_cache_dir`
looks for next to `bin/`.

**Example.** `teeny-triton --example cross_build` — a `cargo teeny check --target
jetson-orin-nano` that passes without the board.

### 24. What Is Portable

**Goal.** The reader knows what they are committing to.

**Introduces.** Backend; where the abstraction is real and where it is not.

**Uses.** The `Triton` trait as the portable surface;
`compiler/teeny-compiler/src/compiler/backend/` (`llvm`, `ndarray`);
`teeny_core::device::Device`; the CUDA-specific parts of the launch path.

**Example.** None — a table.

**Note.** The existing book's `kernels-and-backends/backends.md` is titled "CPU,
CUDA, and Vulkan Backends" but its body is accurate: only `teeny-cuda` exists,
`teeny-cpu` and `teeny-vulkan` are roadmap, and the `ndarray` path is the
current CPU story. Only the title oversells. This chapter says the same thing
without the misleading heading.

---

## Part 6 — Reference

### Python Triton to Rust

The whole `Triton` trait, alphabetical, each entry giving the Python Triton
spelling, the Rust signature, and any difference in behaviour. Generated from
`kernels/teeny-triton/src/triton/mod.rs` so it cannot drift.

Also covers what has no Rust equivalent: `tl.constexpr` (const generics instead),
`@triton.autotune`, `@triton.heuristics`, `num_warps`/`num_stages`.

### Common Compile Errors

Real error text and the fix. Seeded from the macro's own diagnostics
(`#[kernel] requires a type parameter with a Triton bound`; `` `dtypes` must be a
list ``; `cannot infer supported dtypes`; `` `x` is not a known scalar dtype ``)
and the compiler's (`no teenyc rustup toolchain found`; `multiple teenyc rustup
toolchains found`; `PTX .version 8.6 does not support .target sm_120a`).

### Glossary

Every GPU term the book uses, defined in one sentence.

### Appendix: Porting a Python Triton Kernel

One Python Triton kernel from the official tutorials, ported line by line, with
the Rust type system catching one thing the Python version got away with.

---

## What changed from the original brief

| Brief | Here | Why |
|---|---|---|
| Ch 5 needs `Op` + `Lowering` to run anything | Part 2 runs kernels with neither | `compile_kernel` + `device.launch` is the real minimum; the graph is optional |
| Ch 15 autotuning | Ch 15 dtype dispatch | No autotuner exists |
| Ch 16 warps, stages, occupancy | Ch 16 choosing a block size | `num_warps`/`num_stages` are not settable |
| Ch 18 shared memory and software pipelining | dropped | Not exposed by the API |
| Ch 19 Nsight Compute | Ch 18 criterion only | No GPU to produce real Nsight output |
| Ch 22 gradients | Ch 22, unchanged | Real: `#[kernel(backward = …)]` |
| mdbook-linkcheck | site link check | mdbook-linkcheck does not support mdbook 0.5.x |
| GitHub Pages + CNAME deploy | docs site registry entry | `docs.teenygrad.org` is the Nuxt site; a book is a registry entry there |
