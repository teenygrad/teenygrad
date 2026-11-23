/*
 * Copyright (c) 2026 teenygrad (https://teenygrad.org).
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

//! Inkling-2 (`Inkling2LoopForConditionalGeneration`), starting with Thinking
//! Machines' Inkling-2-Mini (`thinkingmachines/Inkling-2-Mini`).
//!
//! **Status: the MoE block runs; attention does not.** [`model::build_router`] and
//! [`model::build_moe`] lower and run as graphs, and the seven kernels in [`kernels`]
//! are tested one by one against the reference. A whole step graph is blocked on
//! attention, which needs two things [`crate::kernels::attention::AttentionSpec`]
//! does not have:
//!
//! - a **relative position bias** added to the attention logits before the softmax
//!   ([`kernels::rel_bias`] produces it, but nothing consumes it yet);
//! - a per-query **attention sink**, one extra logit with a zero value that absorbs
//!   probability mass, so the attention weights deliberately do not sum to one. This
//!   is why the reference sets `_supports_sdpa = False` and cannot use flash
//!   attention.
//!
//! **The checkpoint does not use the reference's names.**
//! `thinkingmachines/Inkling-2-Mini` ships the original tensor names, which
//! transformers renames on load: `attn.wq_du.weight` is the query projection,
//! `attn.wk_dv` / `attn.wv_dv` the key and value, `attn.wo_ud` the output,
//! `attn.gate_attn` its gate, `attn.wr_du` the relative linear, `attn_norm` and
//! `mlp_norm` the two layer norms. [`model`] names its slots the checkpoint's way,
//! so weights load without a rename pass. The shapes confirm the mapping.
//!
//! **A whole decoder layer needs two mechanisms that are not built yet**, and
//! neither appears in the part of the reference the kernels were written from:
//!
//! - **`attn_res`** (`Inkling2LoopBlockAttentionResidual`): attention from learned
//!   queries over the block states accumulated every `n_layers_per_attn_res_block`
//!   layers. The checkpoint holds `attn_res.attn_queries.{layer}` of `[3, 2304]` and
//!   `attn_res.mlp_queries.{layer}` of `[2304]`, so three learned queries before
//!   attention and one before the MLP.
//! - **`controllers`**: a rank-64 gate per layer, with `down_weight` `[64, 4608]`
//!   and `forget_weight` / `gain_weight` `[2304, 64]`, for `attn`, `k`, `v` and
//!   `mlp`. The 64 is `loop_config.gate_rank` and the 4608 is two hidden widths, so
//!   this is what mixes the three passes of the looped core.
//!
//! Also still to come: the runner, and NVFP4 dequantisation inside the expert
//! gather's inner loop.
//!
//! Four things make this a larger job than the Gemma port, which is why they are
//! named in [`config`] rather than discovered later:
//!
//! - **Looped depth.** Layers 6..18 run three times over the same weights, gated by a
//!   rank-64 projection, so 24 distinct layers execute as 60.
//! - **Mixture of experts.** 8 of 128 experts fire per token per layer, and the expert
//!   weights ship as NVFP4 with a few layers in MXFP8. Dequantisation is on the hot
//!   path, not a load-time step.
//! - **No RoPE.** Position enters through a relative scheme (`d_rel`, `rel_extent`)
//!   together with short convolutions on the attention branch, so
//!   [`crate::kernels::rope`] does not apply.
//! - **Softcapping and L2 clipping**, which the Gemma port rejects outright, are used
//!   on the MLP gate and up projections, the final logits, and the query and relative
//!   terms.

// The kernel tests lower through teeny-kernels and run through teeny-test, and the
// runner will use both once it exists; without this the crate lints as not using the
// dependencies its feature pulls in.
use teeny_kernels as _;
use teeny_test as _;

pub mod config;
pub mod kernels;
pub mod model;

pub use config::{Inkling2Config, Inkling2TextConfig, LoopConfig};
