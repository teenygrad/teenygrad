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

//! [`HostKvCache`]: a cache in contiguous host memory.

use crate::{KvCache, KvCacheError, KvLayout, PastLen, Result, appended_rows};

/// A cache in host memory: per layer, `capacity` rows of `row` floats for keys and as
/// many for values, allocated up front, so a step's graph can take a layer's cache as one
/// `[capacity, row]` input.
#[derive(Debug, Clone)]
pub struct HostKvCache {
    layout: KvLayout,
    keys: Vec<Vec<f32>>,
    values: Vec<Vec<f32>>,
    len: PastLen,
}

impl HostKvCache {
    /// An empty cache of `layout`.
    pub fn new(layout: KvLayout) -> Self {
        let layer = vec![0.0; layout.capacity * layout.row];
        Self {
            layout,
            keys: vec![layer.clone(); layout.layers],
            values: vec![layer; layout.layers],
            len: PastLen::default(),
        }
    }
}

impl KvCache for HostKvCache {
    fn layout(&self) -> KvLayout {
        self.layout
    }

    fn len(&self) -> usize {
        self.len.get()
    }

    fn past_len(&self) -> PastLen {
        self.len.clone()
    }

    fn truncate(&mut self, len: usize) -> Result<()> {
        let current = self.len();
        if len > current {
            return Err(KvCacheError::Truncate {
                len: current,
                to: len,
            });
        }
        self.len.set(len);
        Ok(())
    }

    fn append(&mut self, keys: &[&[f32]], values: &[&[f32]]) -> Result<()> {
        let rows = appended_rows(self.layout, keys, values)?;
        let len = self.len();
        if len + rows > self.layout.capacity {
            return Err(KvCacheError::Full {
                len,
                adding: rows,
                capacity: self.layout.capacity,
            });
        }
        let at = len * self.layout.row;
        let end = at + rows * self.layout.row;
        for l in 0..self.layout.layers {
            self.keys[l][at..end].copy_from_slice(keys[l]);
            self.values[l][at..end].copy_from_slice(values[l]);
        }
        self.len.set(len + rows);
        Ok(())
    }

    fn keys(&self, layer: usize) -> &[f32] {
        &self.keys[layer]
    }

    fn values(&self, layer: usize) -> &[f32] {
        &self.values[layer]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cache() -> HostKvCache {
        HostKvCache::new(KvLayout {
            layers: 2,
            row: 2,
            capacity: 3,
        })
    }

    #[test]
    fn test_appends_rows_and_keeps_past_len_in_step() {
        let mut c = cache();
        let past = c.past_len();
        c.append(&[&[1.0, 2.0], &[3.0, 4.0]], &[&[5.0, 6.0], &[7.0, 8.0]])
            .unwrap();
        c.append(&[&[9.0; 4], &[10.0; 4]], &[&[11.0; 4], &[12.0; 4]])
            .unwrap();
        assert_eq!((c.len(), past.get(), c.remaining()), (3, 3, 0));
        assert_eq!(c.keys(0), [1.0, 2.0, 9.0, 9.0, 9.0, 9.0]);
        assert_eq!(c.values(1), [7.0, 8.0, 12.0, 12.0, 12.0, 12.0]);
        c.truncate(1).unwrap();
        assert_eq!(past.get(), 1);
        // A truncated row is overwritten by the next append.
        c.append(&[&[0.5; 2], &[0.5; 2]], &[&[0.5; 2], &[0.5; 2]])
            .unwrap();
        assert_eq!(&c.keys(0)[..4], [1.0, 2.0, 0.5, 0.5]);
    }

    #[test]
    fn test_refuses_what_does_not_fit() {
        let mut c = cache();
        assert_eq!(
            c.append(&[&[0.0; 8], &[0.0; 8]], &[&[0.0; 8], &[0.0; 8]]),
            Err(KvCacheError::Full {
                len: 0,
                adding: 4,
                capacity: 3
            })
        );
        assert!(matches!(
            c.append(&[&[0.0; 2]], &[&[0.0; 2]]),
            Err(KvCacheError::Layers {
                expected: 2,
                keys: 1,
                values: 1
            })
        ));
        assert!(matches!(
            c.append(&[&[0.0; 2], &[0.0; 4]], &[&[0.0; 2], &[0.0; 2]]),
            Err(KvCacheError::Shape {
                layer: 1,
                what: "keys",
                ..
            })
        ));
        assert_eq!(c.truncate(1), Err(KvCacheError::Truncate { len: 0, to: 1 }));
        let message = KvCacheError::Full {
            len: 2,
            adding: 4,
            capacity: 3,
        }
        .to_string();
        assert!(
            message.contains("2 + 4 positions exceed its 3"),
            "{message}"
        );
    }
}
