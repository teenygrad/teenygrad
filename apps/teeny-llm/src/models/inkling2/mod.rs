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
//! **Slots that need renaming at load.** Most match the checkpoint exactly
//! (`attn.wq_du.weight`, `attn.q_l2_clip.scale`, `mlp.gate.weight`,
//! `mlp.gate.bias`, `attn_norm.weight`). These do not, because `Linear4Op` always
//! names its parameter `weight` and `ExpertGatherOp` likewise:
//!
//! | graph slot | checkpoint |
//! |---|---|
//! | `controllers.<x>.down.weight` | `controllers.<x>.down_weight` |
//! | `controllers.<x>.forget.weight` | `controllers.<x>.forget_weight` |
//! | `controllers.<x>.gain.weight` | `controllers.<x>.gain_weight` |
//! | `attn.rel_logits_proj.weight` | `attn.rel_logits_proj.proj` |
//! | `mlp.experts.weight` | `mlp.experts.w13_weight` |
//! | `mlp.experts_down.weight` | `mlp.experts.w2_weight` |
//! | `attn_res.<x>_queries.query` | `attn_res.<x>_queries.{layer}` |
//!
//! `test_controller_slot_names` pins the first three, so the map cannot drift
//! unnoticed.
//!
//! **The layers are not uniform, and the port builds only the common shape.**
//! Reading the shipped index showed six distinct layer shapes, not one.
//! `test_the_checkpoint_layer_shapes_are_what_the_port_assumes` pins all of this:
//!
//! - **Layers 5, 11, 17 and 23** (the last of each six-layer block) have no
//!   `attn_norm` and no plain `k_sconv.weight`. They carry `q_input_norm`,
//!   `k_input_norm` and `v_input_norm` separately, and a *projected* convolution
//!   (`k_sconv.weight_proj` with `static_w`), which is the reference's
//!   `Inkling2LoopDynamicShortConvolution`. **Neither is implemented.**
//! - **Layers 20 to 23 are MXFP8, not NVFP4** (no `_amax`, no
//!   `_outer_scale_denominator`), matching `hf_quant_config`'s eight MXFP8 tensors.
//!   [`kernels::expert_gather_nvfp4`] handles NVFP4 only.
//! - `attn_res.attn_queries` stops at layer 22 while `mlp_queries` runs to 23, so the
//!   last layer takes no attention message.
//!
//! So [`model::build_layer`] is right for 20 of 24 layers and wrong for four, and the
//! expert gather is right for 20 of 24 and wrong for four. Both gaps are additions
//! rather than corrections: the existing kernels keep working for the layers they
//! cover.
//!
//! Still to come: those four layers, the MXFP8 path, the token loop, and the packed
//! NVFP4 encoding (blocked on the `bitcast` flag, teenygrad#170).
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
pub mod runner;
pub mod session;
pub mod weights;

pub use config::{Inkling2Config, Inkling2TextConfig, LoopConfig};
