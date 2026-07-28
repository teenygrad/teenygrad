/*
 * Copyright (c) 2026 Teenygrad.
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

//! Key-value cache utilities for LLM inference in [teenygrad](https://teenygrad.org) — currently
//! [`DynamicCache`], tracking sequence length and per-layer sliding-window/compileability state
//! across generation steps.

/// Tracks per-layer KV cache state (sequence length, sliding-window/compileability) across
/// generation steps.
#[derive(Debug, Clone, Default)]
pub struct DynamicCache {}

impl DynamicCache {
    /// Creates an empty cache.
    pub fn new() -> Self {
        Self {}
    }

    /// Returns the number of tokens currently cached.
    ///
    /// Not yet implemented.
    pub fn get_sequence_length(&self) -> usize {
        todo!()
    }

    /// Per-layer sliding-window state: `(window_size, is_sliding)` for each layer, or `None` if
    /// no layer uses a sliding window.
    ///
    /// Not yet implemented.
    pub fn is_sliding(&self) -> Option<&[(usize, bool)]> {
        todo!()
    }

    /// Whether this cache's current state is safe to pass into a compiled (AOT/JIT) kernel path.
    ///
    /// Not yet implemented.
    pub fn is_compileable(&self) -> bool {
        todo!()
    }
}
