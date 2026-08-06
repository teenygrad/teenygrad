# Glossary

Every GPU term this book uses, in one sentence each. Chapter references point at
where the term is introduced properly.

**Accumulator** — A tensor held in registers across a loop, summing partial
results, written to memory once at the end. Chapter 11.

**Anchor** — Not a GPU term: an mdbook comment marking a region of a source file
so a chapter can include exactly those lines.

**Arithmetic intensity** — Arithmetic performed per byte loaded. Raising it is
how a memory-bound kernel becomes compute-bound. Chapter 11.

**Atomic** — A read-modify-write that cannot be interleaved with another
program's. Chapter 14.

**Backward kernel** — The kernel computing an operation's gradient, given the
gradient of its output. Chapter 22.

**Block** — The slice of data one program handles, `BLOCK_SIZE` elements wide.
Chapter 2.

**Block pointer** — An addressing mode carrying shape, strides and a tile shape,
built with `make_block_ptr`, as an alternative to explicit offsets. Chapter 17.

**Broadcast** — Copying a scalar across every lane of a block so it can combine
with a tensor. `tt.splat` in the MLIR. Chapter 9.

**Capability** — An NVIDIA GPU generation, written `sm_75` through `sm_120`.
Kernels are compiled for one. Chapter 4.

**Coalescing** — Combining the memory accesses of many lanes into as few
transactions as possible. Contiguous access coalesces; strided access does not.
Chapter 17.

**Compute-bound** — Limited by arithmetic rather than by memory. Chapter 1.

**Const generic** — A compile-time constant parameter. How block sizes reach a
kernel, because the value must be a literal in the captured source. Chapter 6.

**Contention** — Many programs hitting the same address at once, forcing the
hardware to serialise them. Chapter 14.

**CTA** — Cooperative Thread Array. CUDA's name for what Triton calls a program;
appears in `RuntimeOp`'s documentation. Chapter 2.

**Dtype** — An element type: `f32`, `i32`, `bool`. `DtypeRepr` is its runtime,
type-erased form. Chapter 15.

**Entry point** — The `extern "C"` wrapper the macro generates, giving the
loader a predictable symbol, `{name}_entry_point`. Chapter 8.

**Epilogue** — Work done to a result while it is still in registers, before
storing. Chapter 12.

**Fusion** — Combining several operations into one kernel so intermediate
results never reach memory. Chapters 1 and 12.

**Grid** — How many programs to launch. Computed on the CPU, from the data size.
Chapter 6.

**Identity** — The value that leaves a reduction unchanged: 0 for a sum, 1 for a
product, −∞ for a maximum. What masked lanes must be filled with. Chapter 10.

**Kernel** — A function that runs on the GPU, executed by many programs at once.
Chapter 1.

**Lane** — One element's position within a block. Chapter 2.

**Launch** — Starting a grid of programs running a compiled kernel. Chapter 5.

**Lowering** — Turning a graph of operations into a DAG of compilable kernels.
Chapter 20.

**Mask** — A boolean tensor saying which lanes are real. The bounds check.
Chapter 7.

**Memory-bound** — Limited by moving data rather than by arithmetic. Most
kernels. Chapter 1.

**MLIR** — The intermediate representation `teenyc` produces, and the most
useful view into what your kernel compiled to. Chapter 9.

**Monomorphization** — Generating a separate compiled copy per concrete type or
constant. Chapter 15.

**Occupancy** — How many programs a card can keep in flight at once, limited by
registers and shared memory per program. Chapter 16.

**Program** — One instance of your kernel. What your code describes. Chapter 2.

**Program ID** — The index identifying which program you are, and hence which
slice is yours. Chapter 6.

**PTX** — NVIDIA's portable assembly, and what `compile_kernel` produces.
Compiled to machine code by the driver at load time. Chapter 3.

**Race** — Two programs reading and writing the same address with no ordering,
so one update is lost. Chapter 14.

**Reduction** — Combining many values into one: sum, maximum, count. Chapter 10.

**Register** — The fastest storage, private to a lane. Where tensors live inside
a kernel. Chapter 11.

**SASS** — The real machine code for a specific chip, produced from PTX by the
driver. Never seen directly. Chapter 3.

**Scan** — A prefix operation: each output holds the reduction of everything up
to it. Chapter 13.

**Scatter** — Writing to indices computed from data rather than from the program
id. The usual reason to need atomics. Chapter 14.

**Shared memory** — Memory shared between the lanes of one program, faster than
global and slower than registers. Used by reductions; not directly exposed.
Chapter 10.

**SIMT** — Single Instruction, Multiple Threads. CUDA's model, where you write
for one thread. Contrast with Triton's block model. Chapter 2.

**Specialisation** — Compiling a separate kernel per constant or dtype, so the
compiler can use the known values. Chapter 15.

**Stride** — The distance in elements between consecutive entries along a
dimension. 1 along a row-major row; the row length along a column. Chapter 17.

**Symbolic shape** — A shape with unknown dimensions, written `None`, resolved
when real data arrives. Chapter 20.

**Tensor** — In a kernel body, a block of values operated on as a unit. Not the
framework's `SymTensor`, which is a graph node handle. Chapter 2.

**Tensor Core** — Hardware doing a small matrix multiply as one instruction.
Reached through `T::dot`. Chapter 11.

**Tensor descriptor** — A TMA addressing mode built from shape, strides and a
tile shape, loading tiles without explicit offsets. Chapter 11.

**Tile** — A rectangular piece of a larger array that one program works on.
Chapter 11.

**TMA** — Tensor Memory Accelerator. Hardware moving tiles between global and
shared memory without occupying the arithmetic units. Imposes 16-byte alignment.
Chapters 11 and 21.

**Thread** — The hardware's unit of execution. A program is implemented as a
group of them. Chapter 2.

**Warp** — 32 threads executing in lockstep. Why block sizes are multiples of
32. Chapter 2.

**`teenyc`** — The modified rustc that compiles captured kernel source into GPU
code. Chapter 3.
