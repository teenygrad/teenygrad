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

//! Legality checks on a propagated tiling (A9).
//!
//! Only `redundancy` and `covers` are here. A9's `realisable` (and its
//! `seed_from_consts`, `same_spans` and `same_decode`) checks a stage anchor
//! other than the group anchor, and when the tiling stages into the compute
//! level (`l + 1 == H`, always so over device and shared memory) the whole
//! group is one stage whose anchor is the group anchor, so it never fires.

use super::seed::UnitGrid;
use super::span::{DimSpan, Region, drives};

/// Overlap factor (≥ 1). Halo along tiled dims × replication along space axes that
/// don't drive this node. Reduce axes partition work, so they are not counted.
pub(super) fn redundancy(r: &Region, ext: &[u64], grid: &UnitGrid) -> f64 {
    let mut f = 1.0;
    for (s, &e) in r.iter().zip(ext) {
        if let DimSpan::Tiled { step, len, .. } = *s {
            f *= ((len.min(e) as f64) / (step.min(e).max(1) as f64)).max(1.0);
        }
    }
    let d = drives(r);
    for (a, ax) in grid.axes.iter().enumerate() {
        if !ax.reduce && !d.contains(&a) {
            f *= ax.units() as f64;
        }
    }
    f
}

/// A tensor stored for a consumer outside the group must be written in full (fix #4).
/// The windows `[s·u + lo, s·u + lo + len)` for `u = 0..n` leave no gap exactly
/// when `len ≥ s`, start at or before 0 when `lo ≤ 0`, and reach the end when
/// `s(n−1) + lo + len ≥ E`.
pub(super) fn covers(r: &Region, ext: &[u64], grid: &UnitGrid) -> bool {
    r.iter().zip(ext).all(|(s, &e)| match *s {
        DimSpan::Full | DimSpan::Chunk { .. } => true,
        DimSpan::Tiled {
            axis,
            step,
            lo,
            len,
        } => {
            let n = grid.axes[axis].units() as i64;
            lo <= 0 && len >= step && step as i64 * (n - 1) + lo + len as i64 >= e as i64
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::optimizer::anduin::schedule::seed::GridAxis;

    fn grid(axes: &[(u64, u64)]) -> UnitGrid {
        UnitGrid {
            axes: axes
                .iter()
                .map(|&(extent, block)| GridAxis {
                    extent,
                    block,
                    reduce: false,
                })
                .collect(),
        }
    }

    fn tiled(axis: usize, step: u64, lo: i64, len: u64) -> DimSpan {
        DimSpan::Tiled {
            axis,
            step,
            lo,
            len,
        }
    }

    #[test]
    fn test_r4_a_node_not_driven_by_a_space_axis_is_recomputed_per_unit() {
        // §12.7 R4: a broadcast row vector inside a [rows, cols] tiling. Its
        // region follows the row axis only, so every column unit recomputes it.
        let g = grid(&[(64, 8), (1024, 128)]);
        let row_vector = vec![tiled(0, 8, 0, 8), DimSpan::Full];

        assert_eq!(redundancy(&row_vector, &[64, 1], &g), 8.0);
        assert_eq!(
            redundancy(
                &vec![tiled(0, 8, 0, 8), tiled(1, 128, 0, 128)],
                &[64, 1024],
                &g
            ),
            1.0
        );
    }

    #[test]
    fn test_r7_a_padded_halo_with_a_ragged_tail() {
        // §12.7 R7, one tiling level: a 3-wide window, pad 1, over 8-wide
        // output tiles of a 30-wide input. Each unit reads 10 for 8, and the
        // two halo dims together recompute (10/8)² ≈ 1.5625; at a recompute
        // cap ≥ that the stored tensor is still written in full.
        let g = grid(&[(30, 8), (30, 8)]);
        let halo = vec![tiled(0, 8, -1, 10), tiled(1, 8, -1, 10)];

        assert_eq!(redundancy(&halo, &[30, 30], &g), 1.5625);
        assert!(covers(&halo, &[30, 30], &g));
    }

    #[test]
    fn test_r5_a_strided_region_leaves_gaps_and_does_not_cover() {
        // §12.7 R5: a stride-2 consumer reads one element in two of its
        // producer. If that producer is also a model output it has to be
        // stored in full, and its region would leave gaps.
        let g = grid(&[(16, 1)]);
        let strided = vec![tiled(0, 2, 0, 1)];

        assert!(!covers(&strided, &[32], &g));
        assert!(covers(&vec![tiled(0, 2, 0, 2)], &[32], &g));
        // Starting past 0 or stopping short of the end also leaves a gap.
        assert!(!covers(&vec![tiled(0, 2, 1, 2)], &[32], &g));
        assert!(!covers(&vec![tiled(0, 2, 0, 2)], &[33], &g));
    }
}
