# Inkling-2-Mini on the SpacemiT K3: from 0.33 to 3.3 tokens/s

How `teeny_llm_core::models::inkling2` went from 0.33 to 3.3 tokens/s generating, from
0.31 to 6.0 tokens/s reading a prompt, and from a 66 s to a 12 s load, on a SpacemiT K3
board (`jupiter2`: 8 X100 cores at VLEN=256 and 8 A100 "AI" cores at VLEN=1024, 32 GB).
Every step was measured on the board, and every step kept greedy output identical to
the build before it, except where a step changed the arithmetic on purpose; those
cases are called out below.

The work came in two rounds. [Round 1](#the-steps) took generation from 0.33 to 2.3
tokens/s. [Round 2](#round-2-decode-loading-and-the-k3s-two-clusters) took it to 3.3,
cut loading to 12 s, and measured what the K3's second cluster and speculative decoding
can and cannot add.

## Where it ended up

| | Start (`ec517604d`) | Round 1 (`4d6c10a7e`) | Now (`24a7656a8`) |
|---|---|---|---|
| Generation | 0.33 tokens/s | 2.30 tokens/s | **3.2-3.3 tokens/s** (10x) |
| Prompt | 0.31 tokens/s | 5.1 tokens/s | **5.7-6.0 tokens/s** (19x) |
| Load (converted checkpoint) | 66 s | 67 s | **11-13 s** |
| llama.cpp, same board, same model | 0.42 tokens/s | | |

### How the numbers were taken

- Greedy, `--effort none`, 32 generated tokens, prompts "What is the capital of France?
  Answer in one word." and "Write a haiku about rain."; the prompt-speed figures also
  use a 50-token prompt. Each change was run back to back with the build before it on
  the otherwise idle board.
- "Identical" means the generated text matches token for token. Generation speed is the
  chat's own `tokens in .. s` line.
- Per-op times come from `TEENY_RUNTIME_PROFILE=<file>` (below), which accumulates
  per-node time without the per-launch printing that distorted the older
  `TEENY_RUNTIME_TRACE` numbers; whole-process hotspots from `perf record` on the board.

### llama.cpp, for comparison

Built natively on the board from `teenygrad/llama.cpp`'s `spacemit-repeat-dim1-broadcast`
branch with the SpacemiT backend (IME1 and IME2), run with `llama-bench -t 8`:

| GGUF | prompt (pp32) | generation (tg32) |
|---|---|---|
| `ggml-org` NVFP4 | 0.45 tokens/s | 0.42 tokens/s |
| `unsloth` UD-Q4_K_M | 0.45 tokens/s | 0.42 tokens/s |

Both are the same speed on the X100 and on the A100 cores. Both GGUFs keep the experts
in NVFP4 (the "Q4_K_M" names only the other tensors), and llama.cpp has no RISC-V NVFP4
kernel: `perf` put 52% of its time in the generic C `ggml_vec_dot_nvfp4_q8_0` and 21%
in `scalbnf`, decoding E4M3 scales with a libm call per 16 weights. IME, which needs the
Q4/Q8 integer formats, does not reach the experts at all. The 4 tokens/s quoted for
llama.cpp earlier in this PR does not reproduce on this board with either GGUF.

## The steps

Generation speed after each, all on the board:

| Step | Commit | Generation | Prompt |
|---|---|---|---|
| start | `ec517604d` | 0.33 | 0.31 |
| softcap tile back to 1024 lanes (a correctness fix, below) | `93de9b535` | 0.50 | 0.43 |
| NVFP4 decode through an in-register gather | `1a4995915` | 0.55 | 0.51 |
| integer NVFP4 experts | `2c6c3b770` | 1.03 | 1.00 |
| 16-row tiles | `ec8e7388d` | 1.11 | 1.08 |
| bf16 linear weights, two to a word | `45d318d01` | 1.39 | 1.35 |
| spinning launch pool | `c64a24549` | 1.53 | 1.50 |
| byte lanes for the NVFP4 experts | `40328eb2b` | 1.80 | 1.74 |
| widening multiply-accumulate | `8582b47b6` | 1.80 | 1.75 |
| rows-as-lanes MXFP8 experts | `b7d95568a` | 2.19 | 2.13 |
| experts converted to their final layout (format 2) | `6c2b374cd` | 2.18 | 2.12 |
| KV caches passed in place | `c43f0cbc2` | 2.34 | 2.26 |
| batched prompts, 16 tokens a step | `94c3e8217` | 2.32 | 3.61 |
| bf16 weights reused across a batch's rows | `56745fda0` | 2.31 | 4.25 |
| one pass over each NVFP4 expert per batch | `4d6c10a7e` | 2.30 | 5.10 |

Each commit message carries its own before-and-after measurement.

### 1. A correctness bug first: softcap at 128 lanes

`bce58d23a` cut softcap's tile from 1024 to 128 lanes along with replacing its
`expf` call, and on the board the model then answered with no visible text. The kernel
tests passed under qemu. Bisecting the four perf commits on the board, then splitting
`bce58d23a`, put it on the tile size alone: the in-lane exponential is correct at 1024
lanes. Why 128 fails is not known; it points at teenyc's RVV lowering. **qemu does not
model this; only a board run settles it.**

### 2. The NVFP4 experts: from f32 to bytes

Where the expert time went, and what each change removed:

- **In-register gather** (`1a4995915`). The gather loaded each lane's weight word and
  block scale through a per-lane address (`offs >> 3`, `offs >> 4`), which RVV can only
  do as indexed loads (`vluxei64`). It now loads the chunk's words and scales
  contiguously and spreads them with `T::gather`, and looks the E2M1 nibble up in a
  16-entry table, also with `T::gather`. Reading the expert id as a scalar made every
  load offset `arange + scalar`, which the compiler can prove contiguous.
- **Integer arithmetic** (`2c6c3b770`). E2M1 times two is an integer (0, ±1, ±2, ±3,
  ±4, ±6, ±8, ±12), so the weights are exact int8. Activations are quantised to int8 per
  16-element block (`QuantRowsQ8Op`, `round(x * 127 / amax)`); a block's products sum
  exactly in 16 bits (at most 16 × 12 × 127 = 24384) and its two scales apply once. Lanes
  are 32 **output rows**, the reduction runs down `k` with a scalar activation broadcast,
  and there is no horizontal reduction. That needs each row block's weights interleaved
  (`interleave_row_bytes`). **This is the one step that changes the arithmetic**: the
  int8 activations are an approximation, at most half a step of 1/127 of each block's
  largest value. One test prompt stayed identical for all 32 tokens; the other diverged
  after eight words onto another coherent continuation, and a later change in sum order
  flipped it back. A logit-level comparison against the f32 build has not been done.
- **16-row tiles** (`ec8e7388d`) cut the loop's spills from 22 to 8.
- **Byte lanes** (`40328eb2b`). The weight pointer is a byte pointer, so each step loads
  a byte of 32 rows (two weights each) and works in 8-bit lanes: a mask and a shift split
  the nibbles, two `vrgather.vv` look them up in an int8 table, and both
  multiply-accumulate into 16-bit sums. No spills.
- **`vwmacc`** (`8582b47b6`). Casting each activation through `i8` lets LLVM use
  `vwmacc.vx` on the int8 weights directly: about 56 instructions per 256 weights instead
  of 120. It did not move tokens/s, which is how `perf` came to find the next item.

### 3. The MXFP8 experts (layers 20 to 23)

A whole-process `perf` record found `expert_gather_mxfp8_forward` costing more than all
the NVFP4 experts together: 8 calls a token at ~16.6 ms against 40 at ~2.85 ms. The
profile's shared `mlp.experts` name had hidden it. `ExpertGemvMxfp8Op` (`b7d95568a`)
gives MXFP8 the same rows-as-lanes layout: one contiguous byte load for 32 rows per `k`,
an exact E4M3 decode by bit construction (`((b & 0x7F) + 960) << 20`, subnormals
`m * 2^-9`), the E8M0 row scales (`s << 23`) once per 32-element block. Activations stay
f32 here: E4M3 values are not integers at any useful scale.

### 4. Bytes everywhere else: bf16 linears

Every linear weight in the checkpoint is bf16, and the loader widened them to f32, so the
attention projections and the unembedding, already at memory bandwidth (~20 GB/s), read
twice the bytes they needed. `Linear4Bf16Op` (`45d318d01`) keeps them as bf16, two to a
32-bit word, and widens each exactly with a shift or a mask and a bitcast. The loader
refuses any value whose low 16 bits are not zero, so a weight that is really f32 is never
rounded.

### 5. Launch overhead

A token is ~3,450 kernel launches, most of them microseconds of work on a tensor in
cache. rayon's `broadcast` woke sleeping threads for each one, and the launcher looked up
the entry point with `dlsym` every time. `SpinPool` (`c64a24549`,
`drivers/teeny-riscv/src/spin_pool.rs`) keeps workers spinning on an epoch counter for
`TEENY_RISCV_SPIN_US` (default 200 µs) between jobs, then sleeping on a condvar; entry
points are cached. Small launches went from 66 to 41 µs.

Using the A100 cores as well (`TEENY_RISCV_THREADS=16 TEENY_RISCV_AI_THREADS=8`, and
8+4, 8+2) was **slower** every time (best 1.34 tokens/s against 1.54 on the eight
X100s): every launch waits for its slowest thread, and the clusters share one memory bus.

### 6. Loading and copying

- **Format 2** (`6c2b374cd`). The expert repacking ran at every start (about 70 s on
  the board). `inkling2_convert` now writes the expert tensors in their final layout and
  the loader reads them as stored; `CONVERTED_FORMAT` is 2, so a format-1 checkpoint is
  refused with an instruction to reconvert. Load went from 136 s back to 67 s.
  Reconverting on the board takes 377 s.
- **KV caches in place** (`c43f0cbc2`). Each of the 48 layer executions copied its two
  2 MB KV caches into the graph's input buffers: 200 MB a token. `RuntimeModel::run_feeds`
  takes `Feed::Borrow` inputs that the kernels read where they are, when
  `teeny_runtime::HOST_MEMORY` says they can (the native riscv64 launcher; not CUDA, not
  the QEMU launcher). Input uploads went from 34 ms to 1.5 ms a token.

### 7. Batched prompts

A prompt read every weight once per token. `Engine::step_many(tokens)` (`94c3e8217`)
runs up to 16 tokens through all 48 layer executions at once; the chat feeds prompts in
chunks (`--prompt-chunk`, default 16, 1 for the old path).

The layer graphs were already written for T tokens; they were only built for one. What
it took:

- **`RuntimeModel::set_rows`** (`6c0407b64`): compile once for 16 rows and run at any
  count up to that with one copy of the weights. A second, 16-row set of graphs would
  have needed a second 25 GB of weights.
- **Shape propagation**. The first board run answered "!!!!": ops downstream of one whose
  row count is not its input's (a linear after a regroup to `T * heads`, the experts'
  `T * top_k`) took `T` as their row count. Each node's concrete shape now comes from its
  own shape rule applied to its inputs' concrete shapes
  (`teeny_core::graph::infer_output_shape`, now public). Found by dumping every node's
  shape for the static and dynamic graphs (`INKLING2_DUMP_SHAPES`) and diffing; all 1876
  agree.
- **Weight reuse across rows** (`56745fda0`): the bf16 linears number their programs down
  the rows, so a batch reads each weight row from memory once and from cache after.
- **One pass per expert** (`4d6c10a7e`): the first slot routed to an NVFP4 expert computes
  every slot routed to it, with the expert's 32-row block (~54 KB) hot in cache. The same
  change to the MXFP8 projection made the prompt slower (its block is ~74 KB and does not
  stay in cache), so it was left out.

Chunk 1 and chunk 16 give identical output on both test prompts.

## Round 2: decode, loading and the K3's two clusters

| Step | Commit | Generation | Prompt | Load |
|---|---|---|---|---|
| round 1 | `4d6c10a7e` | 2.30 | 5.10 | 67 s |
| activation buffer pool kept across runs | `ad37a3c31` | 2.37 | | |
| loop controllers fused into two kernels each | `b84d22942` | 2.40-2.45 | | |
| bf16 linears with output features as lanes | `87b9d50dc` | 2.35-2.40 | 5.7-5.8 | |
| int8 attention linears and unembedding | `1c30d45a4` | 2.73-2.79 | 5.8-5.9 | 77-81 s |
| int8 stored in the converted checkpoint | `20e55bc16` | 2.80 | | 60 s |
| parallel, zero-copy parameter loading | `3fd9fce55` | 2.73-2.75 | | 26 s |
| kernel programs loaded once per thread | `523cbb478` | | | 27 s (disk-bound) |
| NVFP4 scales as E4M3 bytes; bf16 embedding | `5d079c6a2` | 2.98-3.01 | | 11-13 s |
| 32-lane int8 linears; 1024-token context | `9dca8b8ce` | 3.21-3.32 | 5.7-6.0 | |
| A100 matrix-engine linears, opt-in (`INKLING2_IME=1`) | `ca97c166b` | 3.20 | 6.5-6.9 | |
| key/value norms and relative bias fused | `24a7656a8` | 3.18-3.29 | | |

Each commit message carries its measurement. Prompt speed is with 16-token steps.

### 8. Fewer bytes per token

Decode reads every weight once per token, so its speed is bytes over bandwidth.
- **The attention projections and unembedding are int8.** Each output row is
  quantised symmetrically, with one scale per row (`LinearLanesI8Op`). That halves
  3.5 GB a token, the largest single read, to about 1.75 GB. Two prompts give identical
  text; the third rewords its reasoning and stays correct.
  `INKLING2_BF16_LINEARS=1` restores the exact bf16 weights.
- **NVFP4 expert scales are E4M3 bytes.** They had been folded to one f32 per 16
  weights, a third of every NVFP4 expert's bytes. The kernel now decodes the byte by
  bit construction, times one scale per expert.
- **The token embedding is bf16**, its checkpoint dtype, instead of f32.
- **Output features are the lanes** (`LinearLanesBf16Op`, then int8). A weight word is
  unpacked once and feeds up to four rows, with no horizontal sum. At 32 lanes the int8
  kernel spends half the instructions per byte that 16 lanes did, which gained 10% on
  decode.

### 9. Loading: 67 s to 12 s

`INKLING2_LOAD_TIMING=1` prints where the time goes.
- **The first version was one thread.** It decoded each tensor from the mapped
  checkpoint into a fresh vector at about 0.5 GB/s, then copied it into its buffer.
- **Now six loader threads read while the main thread compiles.**
  - Tensors stored as-is are borrowed from the mapping, not decoded.
  - Where the backend's buffers are host memory, the loader threads write straight
    into them.
  - Buffers are allocated zeroed by the allocator instead of written a zero at a
    time, which used to touch 22 GB first.
  - Each kernel program loads once per process, not once per layer.
- **The checkpoint gains compact tensors in place.**
  `inkling2_convert --model <converted> --add-int8` and
  `inkling2_convert --model <original> --add-compact <converted>` add extra shards:
  int8 linears, E4M3 scale bytes and a bf16 embedding. Existing tensors are untouched,
  so older builds still load it. With fewer bytes read, the load fits in page cache
  beside the 22 GB of parameters, and a repeated load no longer reads the SSD.
- Removing the superseded f32 tensors would save about 9 GB of disk, but older builds
  could then no longer load the checkpoint. Not done.

### 10. The K3's A100 cluster

The K3's second cluster has 8 A100 cores at VLEN 1024 with SpacemiT's matrix extension
(IME2: `smt.vmadot`, int4 and int8 tiles accumulating int32). A process reaches it
through `/proc/set_ai_thread`, not `taskset`. Measured on this board:

| | Read bandwidth | Note |
|---|---|---|
| X100 cluster, 8 threads | 17.8 GB/s | |
| A100 cluster, 6-8 threads | 21.8-23.5 GB/s | |
| Both clusters at once | 25.7 GB/s total | one shared memory path |
| Our kernels moved to the A100s | 0.85 tokens/s | ordinary vector code runs ~4x slower there |
| int8 matrix-vector through `smt.vmadot` (prototype) | 22.4 GB/s at 8 threads | about 3.5 GB/s per core below that |

A first run of the prototype gave 29 GB/s at 4 threads and never reproduced. Huge pages,
alignment, clocks and other load were all ruled out; every later run scales about
linearly to about 22 GB/s at 8 threads.

- **What went in (`ca97c166b`, opt-in, `INKLING2_IME=1`).**
  - A `RuntimeOp` can launch a prebuilt native library (`RuntimeOp::native_library`).
    `build.rs` cross-compiles `native/*.c` and the crate embeds the libraries.
  - `teeny-riscv` sends `*_a100_entry_point` kernels to a pool on the A100 cores, and
    the launching X100 thread only waits.
  - The int8 linears run on the matrix engine, with activations quantised per token.
    Up to 8 tokens share a tile, so prompts got 15% faster.
  - Decode only broke even, 3.20 against 3.15 tokens/s. Inside the model it streams at
    about 19.7 GB/s, and each launch costs about 8 us more than it would on the X100s.
- **The experts on the matrix engine don't pay.** NVFP4 has a scale per 16 weights,
  so every matrix op needs its result read out, converted and scaled. The best
  prototype reached 9.7 GB/s, against 13.5 GB/s for the X100 kernel. Only
  re-quantising the experts to the engine's int4 with block scales could change that,
  and that changes the model's numbers.
- **Overlapping the two clusters has a 9% ceiling.** `INKLING2_DUMP_DAG` and
  `scripts/inkling2-overlap-sim.py` list-schedule the real layer DAGs with measured
  per-node times. Of 312 ms of layer kernels a token, the X100 side alone is 240 ms,
  so overlap could save at most 30 ms. Shared memory bandwidth would take part of that.

### 11. What SGLang and vLLM fuse

Both serve Inkling-2 on GPUs. Their attention inputs run as one "prologue" kernel: the
k/v convolutions with their norms, the q, k and rel clips, the rel-bias projection and
the KV write. Our runtime gives an op one output, so `24a7656a8` fuses what fits that
limit:
- the key and value convolutions now end in their L2 clip and RMS norm (`sconv::Post`);
- the relative bias's clip, projection and gather run in one kernel (`RelBiasFusedOp`).

That's about 240 launches a token fewer, but no measured decode gain. The per-op profile
that would show why did not finish.

Their other fusions are candidates for the remaining small-op time:
- multi-query attention residuals, with query scaling at load;
- attention plus output gate plus sink in one kernel;
- the MoE tail: router with top-k and renormalisation, then combine with the MLP
  convolution and residual add.

The study estimated about 30 ms a token across all of them.

### 12. Speculative decoding (DSpark)

`mtp.safetensors` is a 3-layer DSpark drafter. It predicts blocks of 8 tokens from four
target features with a low-rank Markov head.
- **The engine can verify and roll back.** `Engine::keep_undo` and `truncate` roll a
  step back, and `step_many_logits` returns logits for every row. `--rewind-check`
  generates the same text as plain decoding.
- **The drafter works in a reference.** `scripts/inkling2-dspark-ref.py` is a PyTorch
  drafter replayed against the engine's own traces (`INKLING2_TRACE`). It accepts 1.6 of
  7 drafted tokens per step, using the target's visit *inputs* as features.
- **On this board it breaks even at best.** Each extra token in a verify step cost
  0.14 s against 0.43 s for one token, because each token routes to different experts.
  The drafter itself reads roughly 1 GB per draft, an estimate.
- The drafter is not ported. The step costs on the current, faster build were not
  re-measured before the board became unreachable.

## Where the time goes now

A generated token is about 310 ms of kernel time, about 2,900 launches:

| | ms | |
|---|---|---|
| experts (NVFP4 and MXFP8) | ~140 | bandwidth, about 13.5 GB/s |
| int8 attention linears | ~65 | bandwidth, about 15-17 GB/s |
| unembedding (int8) | ~21 | |
| ~2,200 small ops | ~80 | launch- and latency-bound |

Each token streams about 3.6 GB of weights. On the X100 cluster's 17.8 GB/s that is a
floor of about 200 ms, or about 5 tokens/s with no other cost at all. The loop drives
the byte count: layers 6-17 run three times a token, so 36 of the 48 layer executions
re-read weights the same token already read. Only three things move that floor:
- fewer bytes per weight;
- more tokens per weight read (speculative decoding, which MoE routing weakens here);
- more bandwidth (the A100 cluster, which only matrix-engine kernels can use).

## The compiler side (teeny-rustc)

Three commits on `teeny-rustc`'s `feat/rvv-tt-gather`, needed to build these kernels.
**The branch is not yet on GitHub** (no push access to `teenygrad/teeny`); until it is,
building needs a teenyc with them (`TEENYC_PATH=.../build/host/stage1/bin/teenyc`).

- **`tt.gather` lowers to `vrgather.vv`** (`a1d552717` on `main`). It had no
  `TritonToTritonCPU` pattern and crashed teenyc. A 1-D gather now puts the table and
  indices in the low lanes of a scalable vector and calls `llvm.riscv.vrgather.vv` with
  `vl` set to the result length; without RVV it goes through a stack buffer. Tested under
  qemu at VLEN 128, 256, 512 and 1024.
- **Sized for the target's minimum VLEN** (`ee8059410`). Sizing for VLEN=64 sent a
  32-lane f32 gather to the slow fallback. The pass option is `rvv-min-vlen`, from `v`
  and `zvl<N>b`.
- **Poison operands** (`777ee6987`). Zeros in lanes nothing reads forced a register-group
  copy before every `vrgather`; poison removed all 30 from the expert loop.

And in `teeny-triton`, `&` and `>>` on `u8` tensors and `*` by a scalar on `i16`
tensors, for the byte-lane kernels.

## Tools added along the way

- `TEENY_RUNTIME_PROFILE=<file>`: per-node-name time across all models, written as a TSV
  when a model is dropped. `TEENY_RUNTIME_TRACE` prints every node.
- `INKLING2_DUMP_SHAPES=<file>`: every compiled node's concrete shape.
- `--prompt-chunk N` on the chat; `--max-tokens` is optional (a reply runs to the
  model's end token), and `--context` defaults to 1024, the sliding window.
- `INKLING2_LOAD_TIMING=1`: per-graph compile, read and copy time at load.
- `INKLING2_TRACE=<file>`: every layer execution's output, plus the step's token ids
  and block states, for replaying the drafter.
- `INKLING2_DUMP_DAG=<file>` and `scripts/inkling2-overlap-sim.py`: the layer DAGs,
  and a two-cluster schedule simulation over them.
- `--rewind-check`: generation through step, rollback and per-row logits.
- `inkling2_convert --add-int8` / `--add-compact`: compact shards added in place.
- On the board, `perf record` attributes samples to the kernel `.so` files in the kernel
  cache, which is how the MXFP8 layers and llama.cpp's hotspots were found.

## Testing

63 kernel tests pass under qemu (VLEN 256). Round 2 added tests for:
- the lanes linears (bf16 and int8) at 1 to 7 rows;
- int8 against its dequantised weights;
- NVFP4 with E4M3 scale bytes, subnormals included;
- the bf16 embedding, exactly;
- the fused key/value norms (static and dynamic convolutions);
- the fused relative bias against a host reference;
- a `SpinPool` mode that never runs a job on the caller;
- IME weight packing.

The matrix-engine kernels need A100 hardware, which qemu does not emulate; they were
checked on the board against a scalar reference.

Round 1's tests include:

- the int8 projection against an exact host integer reference, and against the f32
  gather within the bound the activation rounding allows;
- q8 activation quantisation against a host reference, including an all-zero block;
- the MXFP8 projection over every non-NaN E4M3 byte with a different scale per row and
  block;
- the bf16 linear against the f32 linear;
- one compiled model giving bit-identical rows at 4 rows and at 1;
- tokens routed to the same expert in one batch.

Plus host unit tests for the spin pool, bf16 packing and both interleaves, and two
teeny-rustc tests that `tt.gather` becomes `vrgather.vv` with RVV and still compiles
without it.

## Not done

- **A logit-level check** of the int8 activations and int8 linears against the f32
  build, beyond greedy text.
- **Why the attention fusion did not speed decode up.** Its per-op profile did not
  finish.
- **DSpark port:** re-measure verify step costs on the current build, then port the
  drafter if the estimate beats plain decoding.
- **More fusions from SGLang** (above), about 30 ms a token in all.
- **Experts on the A100 matrix engine** would need re-quantising them to int4 with block
  scales, which needs a quality check first.
- **Pruning the superseded f32 tensors** from the converted checkpoint (about 9 GB of
  disk).
