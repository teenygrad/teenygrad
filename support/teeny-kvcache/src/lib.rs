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

//! Key/value caches for teenygrad's autoregressive models.
//!
//! A transformer decoding token by token keeps each layer's keys and values for the
//! positions it has seen, so a step only computes the new ones. This crate holds that
//! state behind the [`KvCache`] trait, so a model's session works with any cache:
//!
//! - [`HostKvCache`]: contiguous host memory, per layer `capacity` rows of `row` floats;
//! - [`PrefixCache`]: any cache plus the tokens it holds, keeping the longest prefix a new
//!   prompt shares with them, so re-rendering a conversation each turn only runs what
//!   changed.
//!
//! A cache owns its [`PastLen`]: the number of positions it holds, shared with the
//! attention ops of a compiled step graph, which read it while the step runs.
//!
//! Other caches (device-resident, paged, quantized, ring buffers for sliding-window layers)
//! implement the same trait. Today's step graphs take each layer's cache as a host
//! `[capacity, row]` input, which is what [`KvCache::keys`] and [`KvCache::values`] give.

mod host;
mod prefix;

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

pub use host::HostKvCache;
pub use prefix::PrefixCache;

/// The shape of a cache: `layers` layers, each with room for `capacity` positions of
/// `row` floats (a layer's `kv_heads * head_dim`) for keys and as many for values.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KvLayout {
    /// Layers.
    pub layers: usize,
    /// Floats per position, per layer.
    pub row: usize,
    /// Positions per layer.
    pub capacity: usize,
}

/// What can go wrong using a cache.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum KvCacheError {
    /// Appending would overflow the cache.
    #[error(
        "KV cache full: {len} + {adding} positions exceed its {capacity}; start a new \
         conversation or give the model a longer context"
    )]
    Full {
        /// Positions held.
        len: usize,
        /// Positions being appended.
        adding: usize,
        /// Positions it has room for.
        capacity: usize,
    },
    /// Keys or values were given for the wrong number of layers.
    #[error("expected keys and values for {expected} layers, got {keys} and {values}")]
    Layers {
        /// The cache's layers.
        expected: usize,
        /// Key slices given.
        keys: usize,
        /// Value slices given.
        values: usize,
    },
    /// A layer's keys or values are not `rows` whole positions.
    #[error("layer {layer}: {what} hold {floats} floats, not {rows} positions of {row}")]
    Shape {
        /// The layer.
        layer: usize,
        /// `keys` or `values`.
        what: &'static str,
        /// Floats given.
        floats: usize,
        /// Positions expected (from layer 0's keys).
        rows: usize,
        /// Floats per position.
        row: usize,
    },
    /// Truncating to more positions than the cache holds.
    #[error("cannot truncate a cache of {len} positions to {to}")]
    Truncate {
        /// Positions held.
        len: usize,
        /// Positions asked for.
        to: usize,
    },
    /// A [`PrefixCache`] was given a different number of tokens than positions.
    #[error("{tokens} tokens given for {positions} new positions")]
    Tokens {
        /// Tokens given.
        tokens: usize,
        /// Positions in the keys and values.
        positions: usize,
    },
}

/// The result of a cache operation.
pub type Result<T> = std::result::Result<T, KvCacheError>;

/// The number of positions in a cache, shared between the cache (which keeps it up to
/// date) and whatever reads it while a step runs: the attention ops of a compiled step
/// graph, which need to know how many cached rows are valid. Cloning it gives another
/// handle on the same count.
#[derive(Clone, Default, Debug)]
pub struct PastLen(Arc<AtomicUsize>);

impl PastLen {
    /// The current count.
    pub fn get(&self) -> usize {
        self.0.load(Ordering::SeqCst)
    }

    /// Sets the count. A [`KvCache`] keeps its own up to date; set one by hand only to
    /// drive attention without a cache (in a kernel test, say).
    pub fn set(&self, n: usize) {
        self.0.store(n, Ordering::SeqCst);
    }
}

/// A key/value cache for an autoregressive model.
///
/// Positions `0 .. len()` hold keys and values; a step appends its new ones with
/// [`append`](Self::append), and [`truncate`](Self::truncate) drops positions from the
/// end (to reuse a shared prefix). [`past_len`](Self::past_len) always reads `len()`.
pub trait KvCache: Send {
    /// The cache's shape.
    fn layout(&self) -> KvLayout;

    /// Positions held.
    fn len(&self) -> usize;

    /// Whether it holds no positions.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Positions it can still take.
    fn remaining(&self) -> usize {
        self.layout().capacity - self.len()
    }

    /// The handle attention ops read the length from; it follows every change.
    fn past_len(&self) -> PastLen;

    /// Keeps the first `len` positions.
    fn truncate(&mut self, len: usize) -> Result<()>;

    /// Appends new positions: per layer, keys and values of `n * row` floats each, the
    /// same `n` for every layer.
    fn append(&mut self, keys: &[&[f32]], values: &[&[f32]]) -> Result<()>;

    /// Layer `layer`'s keys, `[capacity, row]`, rows `0 .. len()` valid.
    fn keys(&self, layer: usize) -> &[f32];

    /// Layer `layer`'s values, `[capacity, row]`, rows `0 .. len()` valid.
    fn values(&self, layer: usize) -> &[f32];
}

/// Checks `keys` and `values` for appending to a cache of `layout`, returning how many
/// positions they hold.
pub fn appended_rows(layout: KvLayout, keys: &[&[f32]], values: &[&[f32]]) -> Result<usize> {
    if keys.len() != layout.layers || values.len() != layout.layers {
        return Err(KvCacheError::Layers {
            expected: layout.layers,
            keys: keys.len(),
            values: values.len(),
        });
    }
    let rows = keys.first().map_or(0, |k| k.len() / layout.row.max(1));
    for (layer, (k, v)) in keys.iter().zip(values).enumerate() {
        for (what, floats) in [("keys", k.len()), ("values", v.len())] {
            if floats != rows * layout.row {
                return Err(KvCacheError::Shape {
                    layer,
                    what,
                    floats,
                    rows,
                    row: layout.row,
                });
            }
        }
    }
    Ok(rows)
}
