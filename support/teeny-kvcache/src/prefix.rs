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

//! [`PrefixCache`]: a cache plus the tokens it holds, reused across requests.

use crate::{KvCache, KvCacheError, PastLen, Result};

/// Any [`KvCache`] plus the tokens whose keys and values it holds, in order.
///
/// A chat model re-renders the whole conversation each turn; [`reuse`](Self::reuse) keeps
/// the longest prefix of the new prompt the cache already holds, so only the rest runs.
pub struct PrefixCache {
    cache: Box<dyn KvCache>,
    tokens: Vec<u32>,
}

impl PrefixCache {
    /// Tracks `cache`, emptying it first, since the tokens it holds are unknown.
    pub fn new(mut cache: Box<dyn KvCache>) -> Self {
        // Truncating to empty cannot fail.
        let _ = cache.truncate(0);
        Self {
            cache,
            tokens: Vec::new(),
        }
    }

    /// The cache.
    pub fn cache(&self) -> &dyn KvCache {
        self.cache.as_ref()
    }

    /// The tokens it holds, in order.
    pub fn tokens(&self) -> &[u32] {
        &self.tokens
    }

    /// Positions held.
    pub fn len(&self) -> usize {
        self.cache.len()
    }

    /// Whether it holds nothing.
    pub fn is_empty(&self) -> bool {
        self.cache.is_empty()
    }

    /// Positions it has room for.
    pub fn capacity(&self) -> usize {
        self.cache.layout().capacity
    }

    /// The handle attention ops read the length from.
    pub fn past_len(&self) -> PastLen {
        self.cache.past_len()
    }

    /// How many leading tokens of `prompt` the cache holds.
    pub fn shared(&self, prompt: &[u32]) -> usize {
        self.tokens
            .iter()
            .zip(prompt)
            .take_while(|(a, b)| a == b)
            .count()
    }

    /// Keeps the longest prefix of `prompt` the cache holds, but leaves at least `min_new`
    /// of its tokens to run (a model needs one to produce the next token's logits), and
    /// drops the rest. Returns the positions kept.
    pub fn reuse(&mut self, prompt: &[u32], min_new: usize) -> usize {
        let keep = self
            .shared(prompt)
            .min(prompt.len().saturating_sub(min_new));
        // `keep` never exceeds what the cache holds, so this cannot fail.
        let _ = self.cache.truncate(keep);
        self.tokens.truncate(keep);
        keep
    }

    /// Appends `tokens` with their keys and values (per layer, `tokens.len()` rows each).
    pub fn append(&mut self, tokens: &[u32], keys: &[&[f32]], values: &[&[f32]]) -> Result<()> {
        let rows = crate::appended_rows(self.cache.layout(), keys, values)?;
        if rows != tokens.len() {
            return Err(KvCacheError::Tokens {
                tokens: tokens.len(),
                positions: rows,
            });
        }
        self.cache.append(keys, values)?;
        self.tokens.extend_from_slice(tokens);
        Ok(())
    }

    /// Forgets everything.
    pub fn clear(&mut self) {
        let _ = self.cache.truncate(0);
        self.tokens.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{HostKvCache, KvLayout};

    fn prefix() -> PrefixCache {
        PrefixCache::new(Box::new(HostKvCache::new(KvLayout {
            layers: 1,
            row: 1,
            capacity: 8,
        })))
    }

    fn feed(p: &mut PrefixCache, tokens: &[u32]) {
        let rows: Vec<f32> = tokens.iter().map(|&t| t as f32).collect();
        p.append(tokens, &[&rows], &[&rows]).unwrap();
    }

    #[test]
    fn test_reuses_the_shared_prefix_and_runs_the_rest() {
        let mut p = prefix();
        feed(&mut p, &[1, 2, 3, 4]);
        assert_eq!(p.shared(&[1, 2, 9]), 2);
        // A new turn: the first three tokens match.
        assert_eq!(p.reuse(&[1, 2, 3, 7, 8], 1), 3);
        assert_eq!(
            (p.tokens(), p.len(), p.past_len().get()),
            (&[1, 2, 3][..], 3, 3)
        );
        feed(&mut p, &[7, 8]);
        assert_eq!(&p.cache().keys(0)[..5], [1.0, 2.0, 3.0, 7.0, 8.0]);
        // The same prompt again still runs its last token.
        assert_eq!(p.reuse(&[1, 2, 3, 7, 8], 1), 4);
        p.clear();
        assert!(p.is_empty() && p.tokens().is_empty());
    }

    #[test]
    fn test_tokens_must_match_the_positions() {
        let mut p = prefix();
        assert_eq!(
            p.append(&[1, 2], &[&[0.0]], &[&[0.0]]),
            Err(KvCacheError::Tokens {
                tokens: 2,
                positions: 1
            })
        );
        assert!(p.is_empty());
    }
}
