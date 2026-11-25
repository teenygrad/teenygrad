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
//! values to a [`KvCache`] and feeding RoPE tables from
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
pub mod tools;

pub use crate::kernels::attention::PastLen;
pub use config::GemmaConfig;

/// The host-side KV cache: per layer, `capacity` rows of `kv_heads * head_dim`, the
/// first `len` valid.
pub struct KvCache {
    /// Keys per layer.
    pub k: Vec<Vec<f32>>,
    /// Values per layer.
    pub v: Vec<Vec<f32>>,
    /// Row width (`kv_heads * head_dim`).
    pub row: usize,
    /// Rows per layer.
    pub capacity: usize,
    /// Valid rows.
    pub len: usize,
}

impl KvCache {
    /// An empty cache for `cfg` with room for `capacity` tokens.
    pub fn new(cfg: &GemmaConfig, capacity: usize) -> Self {
        let row = cfg.num_key_value_heads * cfg.head_dim;
        let layer = vec![0.0; capacity * row];
        Self {
            k: vec![layer.clone(); cfg.num_hidden_layers],
            v: vec![layer; cfg.num_hidden_layers],
            row,
            capacity,
            len: 0,
        }
    }

    /// Appends a step's keys and values (`k[l]`, `v[l]`, each `[t, row]`).
    pub fn append(&mut self, k: &[&[f32]], v: &[&[f32]]) -> anyhow::Result<()> {
        let t = k[0].len() / self.row;
        anyhow::ensure!(
            self.len + t <= self.capacity,
            "KV cache full: {} + {t} tokens exceed its {} rows",
            self.len,
            self.capacity
        );
        let at = self.len * self.row;
        for l in 0..self.k.len() {
            self.k[l][at..at + k[l].len()].copy_from_slice(k[l]);
            self.v[l][at..at + v[l].len()].copy_from_slice(v[l]);
        }
        self.len += t;
        Ok(())
    }
}

/// Loads FunctionGemma for [`Model::load`](crate::models::Model::load).
///
/// Serving it as a [`ChatModel`](crate::chat::ChatModel) is teenygrad-1oui.2; until then
/// the weights are located (so a missing download is reported) and loading is refused.
pub(crate) fn load(
    model: crate::models::Model,
    options: &crate::models::LoadOptions,
) -> crate::Result<crate::models::LoadedModel> {
    let dir = model.locate(options.model_dir.as_deref())?;
    Err(crate::Error::Weights(format!(
        "{model} (from {}) cannot be served yet: it does not implement ChatModel",
        dir.display()
    )))
}
