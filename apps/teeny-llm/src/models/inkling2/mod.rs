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
//! **Status: config only.** The step graph, runner and kernels are not written yet.
//! What is here parses and validates `config.json` and records what the rest of the
//! port has to account for.
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

// The kernel tests lower through teeny-kernels, and the runner will once it exists;
// without this the crate lints as not using the dependency its feature pulls in.
use teeny_kernels as _;

pub mod config;
pub mod kernels;

pub use config::{Inkling2Config, Inkling2TextConfig, LoopConfig};
