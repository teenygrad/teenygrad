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

use std::cmp::Reverse;
use std::collections::BinaryHeap;

struct TopK<T: Ord> {
    heap: BinaryHeap<Reverse<T>>,
    k: usize,
}

impl<T: Ord> TopK<T> {
    fn new(k: usize) -> Self {
        Self {
            heap: BinaryHeap::with_capacity(k + 1),
            k,
        }
    }

    fn push(&mut self, item: T) {
        if self.heap.len() < self.k {
            self.heap.push(Reverse(item));
        } else if let Some(Reverse(min)) = self.heap.peek() {
            if item > *min {
                // Replace the smallest in place, with no extra allocation
                *self.heap.peek_mut().unwrap() = Reverse(item);
            }
        }
    }

    fn into_sorted_desc(self) -> Vec<T> {
        let mut v: Vec<T> = self.heap.into_iter().map(|Reverse(x)| x).collect();
        v.sort_unstable_by(|a, b| b.cmp(a));
        v
    }
}
