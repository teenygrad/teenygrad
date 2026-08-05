# What Is Portable

A book about a portable-looking abstraction owes you a straight answer about how
portable it actually is. This chapter is that answer, as of the code this book
was written against.

## The short version

**Your kernel bodies are portable. Everything around them is CUDA.**

The `Triton` trait is a genuine abstraction — a kernel written against it names
no vendor and no device. But there is exactly one driver crate, `teeny-cuda`,
and the launch path, the buffers, the compilation target and the capability enum
are all NVIDIA.

So the portability is real but latent: the kernels are ready for a second
backend that does not exist yet.

## Line by line

| Layer | Portable? |
|---|---|
| Kernel body (`Triton` trait) | Yes — no vendor in the API |
| `#[kernel]` and the generated struct | Yes — device-independent |
| `CustomOp`, the graph, `SymTensor` | Yes — no device concepts |
| `RuntimeOp` | Mostly — `block`/`grid` are a CUDA-shaped model |
| `compile_kernel`, `Target` | No — `driver::cuda`, `target::cuda` |
| `Capability` | No — `sm_75`…`sm_120` |
| `Device`, `Buffer`, `launch` | No — `teeny-cuda` is the only implementation |
| PTX | No — NVIDIA's assembly |

The seam is clean and in a sensible place. A second driver would need a new
`Device`/`Buffer`/`Program` implementation, a target description, and a
compilation path. It would not need your kernels rewritten.

## What exists today

Compilation backends, in `teeny-compiler`:

- **`llvm`** — the real one. Source → MLIR → Triton passes → LLVM → PTX.
- **`ndarray`** — a CPU path, on by default, for running graphs without a GPU.

Device drivers, in `drivers/`:

- **`teeny-cuda`** — the only one.

There is no Vulkan backend, no ROCm backend, no dedicated CPU driver. The
existing teenygrad book is straight about this: `teeny-cpu` and `teeny-vulkan`
are roadmap items, and the `ndarray` path is the current CPU story — which is a
different thing from a driver crate.

## Portable within CUDA

Between NVIDIA generations, most things do carry:

**PTX is forward-compatible.** Built for `sm_75`, it runs on anything newer,
because the driver compiles it for the actual chip at load time.

**Instructions are not backward-compatible.** A kernel that uses tensor
descriptors — the TMA path from Chapter 11 — needs hardware that has TMA. Build
it for an older capability and you get either different code or a failure.

**Block sizes do not transfer.** A block size tuned on an A100 is not the right
one for an Orin: different register files, different memory bandwidth, different
core counts. Nothing warns you; the kernel just runs slower than it should.

**PTX versions can be rejected.** Chapter 23's `sm_120a` case: a driver refusing
a PTX version newer than it knows. Forward compatibility has limits at both
ends.

So "portable across NVIDIA" means "will run", not "will run well". Retune per
target, or accept that you have optimised for one card.

## What would not survive a second backend

If a Vulkan or ROCm driver arrived tomorrow, these would need attention:

**Anything assuming warps of 32.** AMD's wavefronts are 64. Chapter 6's
"multiple of 32" rule is NVIDIA's number, and a block size chosen around it is
NVIDIA-shaped.

**Tensor Core specifics.** `InputPrecision::TF32` names an NVIDIA feature. Other
vendors have matrix units with different precision modes.

**PTX-level anything.** `T::inline_asm_elementwise` takes an assembly string.
Using it ends portability at that line, deliberately and obviously.

**The capability enum.** `Capability` is a list of SM versions. A second vendor
needs a different type, or that one generalised.

None of this is unusual — it is the normal cost of a portable layer over
hardware that is not actually alike. It is worth knowing which of your choices
are the portable kind.

## Practical advice

**Write kernels against `Triton` only.** Avoid inline assembly unless you have
measured that you need it, and mark it loudly where you do.

**Keep the tuning constants together.** Block sizes and tile shapes are the
per-device numbers. If they are named constants in one place, retuning for a new
card is an afternoon. If they are scattered through kernel bodies, it is a week.

**Do not build portability you cannot test.** With one driver, an abstraction
"for the second backend" is untested by construction, and untested abstractions
are usually wrong. The `Triton` trait is enough.

**Assume you will retune.** Correctness transfers between NVIDIA cards.
Performance does not.

## End of Part 5

Your kernel can now be a node in a model, produce gradients, and be built for a
board you have never touched.

Part 6 is reference material: the Python Triton translation table, the compile
errors you will actually hit, a glossary, and a worked port of a Python kernel.
