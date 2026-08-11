# CUDA Diagnostics: Registers and Spills from `ptxas`

Chapter 24 says nothing in this SDK talks to Nsight Compute, and for occupancy,
warp stalls, and instruction mix that is still true. One number it no longer
covers alone: registers and spills. `teenyc` can now run the real
PTX-to-machine-code assembler and hand the result back through
`CudaProgram`, without a profiler.

## Why the compiler alone can't tell you

PTX is a virtual-register ISA. The `.reg .b32 %r<47>` you see in a `.o` file is
"47 distinct values", not "47 hardware registers" — LLVM's NVPTX backend has no
concept of the card's physical register file and never spills anything.

Physical register allocation — and therefore spilling to local memory when a
kernel needs more registers than the hardware has — happens only in `ptxas`,
NVIDIA's PTX assembler, and only for one specific `sm_NN` target. There is no
way to get real register/spill numbers without actually running it, and
running it is what produces the cubin: a real subprocess compile, not free.

## Turning it on

`compile_kernel` does not invoke `ptxas` by default. Opt in with an
environment variable before compiling:

```bash
TEENYC_GENERATE_BIN=1 cargo test -p teeny-cuda --test test_elemwise_add
```

Leave it unset and everything works exactly as before — same PTX, no
subprocess, no register/spill data.

## Reading it back

`teenyc` appends the numbers `ptxas -v` reports as extra `// meta:` PTX
comment lines, next to the shared-memory and scratch metadata block from
Chapter 5. `teeny-cuda` decodes them into a `PtxasStats`:

```rust,ignore
{{#include ../../../../drivers/teeny-cuda/src/device/program.rs:ptxas_stats_struct}}
```

reachable from a loaded program:

```rust,ignore
{{#include ../../../../drivers/teeny-cuda/src/device/program.rs:ptxas_stats_accessor}}
```

which looks like this in practice:

```rust,ignore
{{#include ../../../../drivers/teeny-cuda/tests/test_elemwise_add.rs:ptxas_stats_usage}}
```

## Reading the numbers

- **`num_regs`** — registers used per thread. Feeds directly into the
  occupancy tension from Chapter 16: more registers per thread means fewer
  resident programs per multiprocessor.
- **`spill_stores` / `spill_loads`** — bytes moved to and from local memory
  because the kernel needed more registers than the hardware had. Local
  memory is off-chip; nonzero spills are usually worth reacting to (a smaller
  block size, fewer live accumulators) rather than living with.
- **`stack_frame`** — bytes of per-thread stack. Zero for most kernels in this
  tree; nonzero means something is spilling to a real stack frame rather than
  bare local-memory slots.
- **`cmem_banks`** — `(bank, bytes)` pairs. Bank `0` is almost always kernel
  parameters; other banks are user constants, when a kernel has any.

None of these are pass/fail thresholds. A kernel with a few spilled bytes and
high occupancy can still beat a spill-free kernel that launches too few
programs to fill the card — this is a second data point for the tension
described in Chapter 16, not a replacement for measuring the kernel itself
(Chapter 24).

## When it's `None`

`ptxas_stats()` returns `None`, not zeros, in two cases:

- `$TEENYC_GENERATE_BIN` was not set for the compile that produced this PTX —
  the common case, and the default.
- The program was loaded via `CudaProgram::try_new` from a cubin directly,
  which carries no PTX metadata to parse.

Absence means "not measured", never a fabricated zero — treat a `None` as "I
don't know", not "there were no spills".
