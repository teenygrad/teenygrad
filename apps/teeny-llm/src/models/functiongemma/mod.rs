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

//! Gemma 3 text models (`Gemma3ForCausalLM`) on teenygrad, starting with Google's
//! FunctionGemma (`google/functiongemma-270m-it`).
//!
//! [`model::build_step`] records one decoding step as a teeny graph; the host runs it
//! for the prompt and then once per generated token, appending each step's keys and
//! values to a [`KvCache`](teeny_kvcache::KvCache) and feeding RoPE tables from
//! [`rope_tables`](crate::kernels::rope::rope_tables).
//!
//! The step graph uses the shared kernels in [`crate::kernels`] (attention, linear,
//! RoPE, last row) and Gemma's own in [`kernels`] (scaled embedding, RMSNorm with its
//! `1 + w` scale, gated tanh-GELU). See `docs/functiongemma.md` for running it on CUDA
//! and on a RISC-V board.

#[cfg(not(any(feature = "cuda", feature = "riscv")))]
compile_error!(
    "the `functiongemma3` feature needs a backend to run on: enable `cuda`, `riscv` or `qemu` with it"
);

pub mod chat;
pub mod config;
pub mod kernels;
pub mod model;
#[cfg(any(feature = "cuda", feature = "riscv"))]
pub mod runner;
#[cfg(any(feature = "cuda", feature = "riscv"))]
pub mod serve;
pub mod tools;

pub use crate::kernels::attention::PastLen;
pub use config::GemmaConfig;
#[cfg(any(feature = "cuda", feature = "riscv"))]
pub use serve::FunctionGemmaChat;
#[cfg(any(feature = "cuda", feature = "riscv"))]
pub(crate) use serve::load;
