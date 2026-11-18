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
//! values to a [`KvCache`] and feeding RoPE tables from [`rope_tables`].

pub mod chat;
pub mod config;
pub mod kernels;
pub mod model;
pub mod runner;
pub mod tools;

pub use config::GemmaConfig;
pub use kernels::attention::PastLen;

/// RoPE `cos` and `sin` tables `[positions.len(), head_dim]` for base `theta`, as
/// Hugging Face's default rotary embedding builds them (both halves equal).
pub fn rope_tables(theta: f64, head_dim: usize, positions: &[usize]) -> (Vec<f32>, Vec<f32>) {
    let half = head_dim / 2;
    let mut cos = Vec::with_capacity(positions.len() * head_dim);
    let mut sin = Vec::with_capacity(positions.len() * head_dim);
    for &p in positions {
        let row: Vec<f32> = (0..half)
            .map(|i| {
                // As Hugging Face: inv_freq in f32, then pos * inv_freq in f32.
                let inv = (1.0 / theta.powf((2 * i) as f64 / head_dim as f64)) as f32;
                p as f32 * inv
            })
            .collect();
        for _ in 0..2 {
            cos.extend(row.iter().map(|a| a.cos()));
            sin.extend(row.iter().map(|a| a.sin()));
        }
    }
    (cos, sin)
}

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
