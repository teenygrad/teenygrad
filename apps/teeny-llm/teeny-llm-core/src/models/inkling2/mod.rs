//! Inkling-2 (`Inkling2LoopForConditionalGeneration`), starting with Thinking
//! Machines' Inkling-2-Mini (`thinkingmachines/Inkling-2-Mini`): a 26.4B multimodal
//! mixture-of-experts model whose text path runs here on RISC-V.
//!
//! # The model
//!
//! 24 decoder layers, of which the middle twelve (`loop_config.core_start ..
//! core_end`, 6..18) run **three times over the same weights**, so a token costs
//! **48 layer executions from 24 sets of weights**. Each layer's feed-forward is a
//! sparse MoE: a router picks **8 of 128 experts** per token, and only those run.
//! 3.4B parameters are active per token, about 5.4B counting the extra passes.
//!
//! Attention is hybrid. Twenty layers are sliding-window; layers 5, 11, 17 and 23 are
//! full-attention and differ structurally -- separate q/k/v input norms instead of
//! one `attn_norm`, convolutions whose taps are projected per token, and three
//! attention-residual queries instead of one. There is no RoPE: position enters
//! through a relative bias and short causal convolutions on the attention branch.
//!
//! # How the runtime executes it
//!
//! [`engine::Engine`] compiles **one graph per layer, 24 in all, not one per layer
//! execution**. A second graph for a core layer would mean a second copy of its
//! 873 MB of expert weights, and the resident set is already ~26 GB against a 31 GB
//! board. The three passes differ only in their inputs.
//!
//! [`session::actions`] turns [`runner::plan`] into the sequence the engine walks:
//! run a layer, push a block state at a boundary, save the core's entry state before
//! the first pass, build the cross-pass messages, restore, repeat. The host carries
//! between layers what the graphs cannot: the hidden states, the block states, four
//! convolution windows per cache slot and the KV cache. [`engine::block_counts`]
//! checks the premise that makes one graph per layer sound -- that a core layer sees
//! the same number of committed block states on every pass -- and fails loudly if it
//! ever stops holding.
//!
//! Two subtleties in that walk, both of which produced plausible-looking wrong output
//! before they were found:
//!
//! - A boundary commits the layer's **input**, not its output, and its MLP residual
//!   attends over a stack one deeper than its attention did, because the append
//!   happens between the two.
//! - The cross-pass messages are retrieved from the state the **previous pass** ended
//!   in, so they must be built *before* the restore overwrites it.
//!
//! # The MoE block
//!
//! Per layer per token: [`kernels::router`] scores all 128 experts
//! (`sigmoid`, with `mlp.gate.bias` steering the choice but not the weights, then
//! normalise and scale by `route_scale`), and the top 8 are gathered.
//! [`kernels::expert_gather_nvfp4`] and [`kernels::expert_gather_mxfp8`] multiply the
//! token by those experts' weights **straight out of their quantised form**, decoding
//! each weight inside the matmul loop; [`kernels::swiglu`] applies the softcapped
//! gate; [`kernels::scatter_add`] combines the eight results by their router weights.
//!
//! **This is where essentially all the time goes.** Per-node tracing on the board
//! puts the two gathers at ~88% of a layer: 41 ms and 23 ms of 74 ms, against ~1.5 ms
//! for each of attention's big projections. Two reasons compound. The MoE is ~64M
//! MACs a layer against attention's ~26M, *and* each of its weights costs ~30
//! operations to unpack where attention's are already f32 -- about 5x the time per
//! MAC. Optimising anything else in the layer is not worth doing.
//!
//! ## Quantisation
//!
//! Layers 0..20 are **NVFP4**, layers 20..24 **MXFP8**; [`runner::quant_of`] decides.
//!
//! NVFP4 stores a weight as three factors: an E2M1 nibble, an E4M3 scale shared by 16
//! weights, and a per-expert `amax / outer_scale_denominator` where the denominator
//! is 2688 = 448 x 6, so that the largest representable product is exactly the stored
//! `amax`. The last two are constant per block, so
//! [`weights::Checkpoint::folded_scale`] multiplies them at load time and the kernel
//! loads one f32 instead of decoding an E4M3 byte per weight. MXFP8 is one E4M3 byte
//! per weight with an exponent-only E8M0 scale per 32, and is not folded.
//!
//! `mlp.experts.w13_weight` **interleaves the gate and up projections down its rows**
//! -- gate, up, gate, up -- rather than storing two contiguous halves, which is why
//! transformers converts it with `Interleave(dim=1)` rather than renaming it.
//! [`kernels::swiglu`] reads alternating lanes accordingly. Splitting it into halves
//! instead pairs every gate with the wrong up, and the model still emits fluent
//! nonsense, so nothing short of a reference comparison catches it.
//!
//! # Checkpoint names
//!
//! The checkpoint ships Thinking Machines' own tensor names, not the reference's.
//! **`transformers/conversion_mapping.py` is the authority** -- diff against it
//! rather than inferring from names, which is how the interleave was missed.
//! [`runner::checkpoint_name`] encodes the mapping; the non-obvious entries:
//!
//! | graph slot | checkpoint |
//! |---|---|
//! | `controllers.<x>.{down,forget,gain}.weight` | `controllers.<x>.*_weight` |
//! | `attn.rel_logits_proj.weight` | `attn.rel_logits_proj.proj`, **transposed** |
//! | `mlp.experts.weight` | `mlp.experts.w13_weight`, **interleaved** |
//! | `mlp.experts_down.weight` | `mlp.experts.w2_weight` |
//! | `attn_res.attn_queries_<n>.query` | `attn_res.attn_queries.{layer - 1}` |
//! | `attn_res.output_query_0.query` | `attn_res.output_query` |
//! | `cross_pass.{queries,kv_queries}_*` | `middle_cycle_cross_pass_*` |
//!
//! `rel_logits_proj.proj` is `[d_rel, rel_extent]` because the reference uses it as
//! `states @ proj`, not as a linear's weight; it has the same element count either
//! way, so reading it untransposed loads cleanly and computes nonsense.
//!
//! # Loading
//!
//! The shipped checkpoint needs bf16 expanded, quantised tensors repacked four bytes
//! to an f32 container, and the NVFP4 scales folded -- none of which depends on the
//! run. The `inkling2_convert` example does it once and writes a converted
//! checkpoint; [`weights::CONVERTED_FORMAT`] guards it against going stale. That
//! takes the board's load from 195 s to 66 s for ~26 GB of parameters.
//!
//! # Verification
//!
//! Every level is checked against `transformers`' own `inkling2_loop` modules rather
//! than against a host reference written from the same reading of the model, because
//! a shared misreading passes silently -- which it did, repeatedly. The kernels have
//! recorded fixtures, individual layers are driven stage by stage with the real
//! weights (`scripts/inkling2-ref-layer.py`), and a whole token is compared against
//! `TextModel.forward` itself (`scripts/inkling2-ref-model.py`), which removes even
//! the transcription of the loop as a variable.
//!
//! # Speed
//!
//! On a SpacemiT K3 board, confined to its eight X100 cores: ~62 ms a layer, 3.07 s
//! a token, 0.32 tokens/s. Four rounds took that from 626 ms a layer: removing
//! `exp2` from the dequant, unpacking with shifts, telling LLVM the vector width is
//! 256 bits, and folding the scales. llama.cpp's Q4_0 build of the same model on the
//! same cores does 4 tokens/s.
//!
//! That gap is not the per-weight unpacking, as this note used to say. A token moves
//! 3.96 G active parameters, so llama.cpp's 4 tokens/s is ~8.9 GB/s of weights and our
//! 0.32 is ~0.7 GB/s: we are nowhere near a bandwidth limit, we are spending the time
//! elsewhere. The assembly says where. Vector values were sized at a whole register
//! file (`BLOCK = 256` at `e32`, VLEN=256) and spilled to the stack -- 229 whole-register
//! spill/fill operations in the NVFP4 gather against ~27 arithmetic ones. It is not
//! confined to that kernel: `nvfp4_dequant`, `gated_mul`, `softcap` and `swiglu` each
//! spill 370-450 times, and at 6500-7900 lines of assembly for ops this simple they
//! have a second problem, unrolling, which `BLOCK` does not explain.
//!
//! `BLOCK = 32` takes the gather's spills to 6 and its assembly to a fifth, with the
//! kernel tests unchanged. What that is worth in time is **unmeasured**: the board was
//! unreachable when it landed, and qemu misreports these kernels by 2-3x in the wrong
//! direction, so treat spill counts as a proxy and nothing more until a board run
//! replaces the figures in the first paragraph.

// The kernel tests lower through teeny-kernels and run through teeny-test, and the
// runner will use both once it exists; without this the crate lints as not using the
// dependencies its feature pulls in.
use teeny_kernels as _;
use teeny_test as _;

pub mod chat;
pub mod config;
#[cfg(any(feature = "cuda", feature = "qemu", feature = "riscv"))]
pub mod engine;
pub mod kernels;
pub mod load;
pub mod model;
pub mod runner;
pub mod session;
pub mod weights;

pub use config::{Inkling2Config, Inkling2TextConfig, LoopConfig};
