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

//! Regions and their span algebra (ANALYSIS.md §12.3 "Regions", A7).

use std::collections::{BTreeMap, BTreeSet};

/// Index into the `UnitGrid` of the tiling that owns the region.
pub(super) type AxisId = usize;

/// What ONE unit touches along ONE tensor dim. Coordinates are relative to the
/// parent unit's region (absolute at level 0). `u_a` is the unit's index on axis `a`.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(super) enum DimSpan {
    /// The whole extent, in every unit.
    Full,
    /// `[step·u_a + lo, step·u_a + lo + len)` ∩ `[0, extent)`.
    Tiled {
        axis: AxisId,
        step: u64,
        lo: i64,
        len: u64,
    },
    /// The whole extent, walked inside the unit `len` elements at a time; the
    /// walk is shared by `split` sub-units (each walks `extent / split`).
    Chunk { len: u64, split: u64 },
}

pub(super) type Region = Vec<DimSpan>; // one per real dim (replaces `TileShape` in the scheduler)

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(super) struct RChunk {
    pub len: u64,
    pub split: u64,
}

/// Per reduce-extent name (P6): the chunk walked per iteration, and the sub-unit split.
pub(super) type RStep = BTreeMap<&'static str, RChunk>;

pub(super) fn ceil_div(a: u64, b: u64) -> u64 {
    a.div_ceil(b.max(1))
}

/// The smallest span covering both. Commutative, associative and idempotent
/// (it is just min and max), so A8's result does not depend on the order
/// consumers are visited in.
pub(super) fn union(a: DimSpan, b: DimSpan) -> DimSpan {
    use DimSpan::*;
    match (a, b) {
        (
            Tiled {
                axis: a1,
                step: s1,
                lo: l1,
                len: n1,
            },
            Tiled {
                axis: a2,
                step: s2,
                lo: l2,
                len: n2,
            },
        ) if a1 == a2 && s1 == s2 => {
            let lo = l1.min(l2);
            let hi = (l1 + n1 as i64).max(l2 + n2 as i64);
            Tiled {
                axis: a1,
                step: s1,
                lo,
                len: (hi - lo) as u64,
            }
        }
        (Chunk { len: x, split: p }, Chunk { len: y, split: q }) if p == q => Chunk {
            len: x.max(y),
            split: p,
        },
        _ => Full, // includes any Full, and mismatched axis/step/kind: conservative
    }
}

/// [`union`] of two regions of the same rank, dim by dim.
pub(super) fn union_region(a: &Region, b: &Region) -> Region {
    a.iter().zip(b).map(|(x, y)| union(*x, *y)).collect()
}

pub(super) fn contains(big: DimSpan, small: DimSpan) -> bool {
    use DimSpan::*;
    match (big, small) {
        (Full, _) => true,
        (
            Tiled {
                axis: a,
                step: s,
                lo: bl,
                len: bn,
            },
            Tiled {
                axis: b,
                step: t,
                lo: sl,
                len: sn,
            },
        ) => a == b && s == t && bl <= sl && sl + sn as i64 <= bl + bn as i64,
        (Chunk { len: x, split: p }, Chunk { len: y, split: q }) => p == q && y <= x,
        _ => false,
    }
}

/// Crossing a staged (non-lockstep) internal edge: the producer stage must finish
/// a walked dim before the consumer stage starts.
pub(super) fn chunk_to_full(r: &Region) -> Region {
    r.iter()
        .map(|s| {
            if matches!(s, DimSpan::Chunk { .. }) {
                DimSpan::Full
            } else {
                *s
            }
        })
        .collect()
}

/// Per-unit lengths for traffic and compute (a walked dim is read in full over the walk).
pub(super) fn lens_traffic(r: &Region, ext: &[u64]) -> Vec<u64> {
    r.iter()
        .zip(ext)
        .map(|(s, &e)| match *s {
            DimSpan::Full => e,
            DimSpan::Chunk { split, .. } => ceil_div(e, split),
            DimSpan::Tiled { len, .. } => len.min(e),
        })
        .collect()
}

/// Per-unit lengths resident at one time (footprint); `mult` = 2 when double-buffered.
pub(super) fn lens_fp(r: &Region, ext: &[u64], mult: u64) -> Vec<u64> {
    r.iter()
        .zip(ext)
        .map(|(s, &e)| match *s {
            DimSpan::Full => e,
            DimSpan::Chunk { len, .. } => len.min(e) * mult,
            DimSpan::Tiled { len, .. } => len.min(e),
        })
        .collect()
}

/// The extents a child tiling sees: one parent unit's region (one chunk of a walked dim).
pub(super) fn sub_lens(r: &Region, ext: &[u64]) -> Vec<u64> {
    lens_fp(r, ext, 1)
}

/// The grid axes that tile some dim of `r`.
pub(super) fn drives(r: &Region) -> BTreeSet<AxisId> {
    r.iter()
        .filter_map(|s| match s {
            DimSpan::Tiled { axis, .. } => Some(*axis),
            _ => None,
        })
        .collect()
}

/// How many chunks the walked dims of `r` take (A12-aux `walks`).
pub(super) fn walks(r: &Region, ext: &[u64]) -> u64 {
    r.iter()
        .zip(ext)
        .map(|(s, &e)| match *s {
            DimSpan::Chunk { len, split } => ceil_div(ceil_div(e, split), len),
            _ => 1,
        })
        .product()
}

#[cfg(test)]
mod tests {
    use super::*;
    use DimSpan::*;

    fn tiled(axis: AxisId, step: u64, lo: i64, len: u64) -> DimSpan {
        Tiled {
            axis,
            step,
            lo,
            len,
        }
    }

    /// A sample of every kind of span, including ones that do and don't share
    /// an axis or step.
    fn spans() -> Vec<DimSpan> {
        vec![
            Full,
            tiled(0, 4, 0, 4),
            tiled(0, 4, -1, 6),
            tiled(0, 4, 2, 3),
            tiled(1, 4, 0, 4),
            tiled(0, 8, 0, 8),
            Chunk { len: 16, split: 1 },
            Chunk { len: 32, split: 1 },
            Chunk { len: 16, split: 2 },
        ]
    }

    #[test]
    fn test_union_is_commutative_associative_and_idempotent() {
        let all = spans();
        for &a in &all {
            assert_eq!(union(a, a), a, "union({a:?}, {a:?}) should be {a:?}");
            for &b in &all {
                assert_eq!(union(a, b), union(b, a), "union of {a:?} and {b:?}");
                for &c in &all {
                    assert_eq!(
                        union(union(a, b), c),
                        union(a, union(b, c)),
                        "union of {a:?}, {b:?} and {c:?}"
                    );
                }
            }
        }
    }

    #[test]
    fn test_union_contains_both_of_its_arguments() {
        let all = spans();
        for &a in &all {
            for &b in &all {
                let u = union(a, b);
                assert!(contains(u, a), "{u:?} should contain {a:?}");
                assert!(contains(u, b), "{u:?} should contain {b:?}");
            }
        }
    }

    #[test]
    fn test_w3_a_halo_diamond_unions_to_one_wider_window() {
        // ANALYSIS.md §12.5 W3: one consumer reads rows [h·u - 1, h·u + h + 1)
        // of a producer the other reads [h·u, h·u + h). The producer's region
        // is the wider of the two.
        let h = 8;
        let halo = tiled(0, h, -1, h + 2);
        let plain = tiled(0, h, 0, h);

        assert_eq!(union(plain, halo), tiled(0, h, -1, h + 2));
    }

    #[test]
    fn test_r2_spans_on_different_axes_or_steps_union_to_full() {
        // ANALYSIS.md §12.7 R2: two tilings of one dim that don't line up have
        // no common box, so the union gives up and takes the whole extent.
        assert_eq!(union(tiled(0, 4, 0, 4), tiled(1, 4, 0, 4)), Full);
        assert_eq!(union(tiled(0, 4, 0, 4), tiled(0, 8, 0, 8)), Full);
        assert_eq!(union(tiled(0, 4, 0, 4), Chunk { len: 4, split: 1 }), Full);
        assert_eq!(
            union(Chunk { len: 4, split: 1 }, Chunk { len: 4, split: 2 }),
            Full
        );
    }

    #[test]
    fn test_lens_count_a_walked_dim_in_full_for_traffic_but_one_chunk_resident() {
        let r = vec![tiled(0, 4, -1, 6), Chunk { len: 16, split: 2 }, Full];
        let ext = [5, 100, 7];

        // A tile wider than the dim is clipped to it.
        assert_eq!(lens_traffic(&r, &ext), vec![5, 50, 7]);
        assert_eq!(lens_fp(&r, &ext, 1), vec![5, 16, 7]);
        assert_eq!(lens_fp(&r, &ext, 2), vec![5, 32, 7]);
        assert_eq!(walks(&r, &ext), 4); // ceil(ceil(100 / 2) / 16)
    }

    #[test]
    fn test_chunk_to_full_and_drives() {
        let r = vec![tiled(2, 4, 0, 4), Chunk { len: 16, split: 1 }, Full];

        assert_eq!(chunk_to_full(&r), vec![tiled(2, 4, 0, 4), Full, Full]);
        assert_eq!(drives(&r), BTreeSet::from([2]));
    }
}
