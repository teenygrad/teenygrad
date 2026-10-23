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

//! Unit grids, search knobs, and seeding a tiling from knob values
//! (ANALYSIS.md §12.3 "Regions"/"Search knobs", A11).
//!
//! No binding is ever walked: `TileAxisBinding` has no `walk` flag yet (P8),
//! so every block const is re-boxed into grid axes and walks never produce a
//! `Chunk`.

use super::span::{AxisId, DimSpan, Region};
use super::{Ctx, Lvl, Owner, Reject, meta};
use crate::graph::optimizer::anduin::tile_graph::NodeId;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(super) struct GridAxis {
    pub extent: u64,
    pub block: u64,
    /// A reduce axis splits a reduction across sub-units (Welder `reduce_thread`).
    /// It drives no output dim and is NOT recomputation.
    pub reduce: bool,
}
impl GridAxis {
    pub fn units(&self) -> u64 {
        self.extent.div_ceil(self.block.max(1))
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct UnitGrid {
    pub axes: Vec<GridAxis>,
}
impl UnitGrid {
    pub fn units(&self) -> u64 {
        self.axes.iter().map(|a| a.units()).product()
    }
    pub fn space_units(&self) -> u64 {
        self.axes
            .iter()
            .filter(|a| !a.reduce)
            .map(|a| a.units())
            .product()
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(super) enum KnobStyle {
    AnchorConsts,
    SinkDims,
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub(super) enum KnobName {
    Block(&'static str),
    Dim(usize),
}

#[derive(Clone, Debug)]
pub(super) struct Knob {
    pub name: KnobName,
    pub values: Vec<u64>, /* ascending, deduped */
}

/// The kernels' own block consts only exist at the level where units are blocks.
/// Every other level searches free tile shapes.
pub(super) fn style(cx: &Ctx, l: Lvl) -> KnobStyle {
    if cx.hw.owner_of_units(l) == Owner::Block {
        cx.caps.knob_style()
    } else {
        KnobStyle::SinkDims
    }
}

/// Every divisor of `n`, ascending.
pub(super) fn factors(n: u64) -> Vec<u64> {
    let mut small = Vec::new();
    let mut large = Vec::new();
    let mut i = 1;
    while i * i <= n {
        if n % i == 0 {
            small.push(i);
            if i * i != n {
                large.push(n / i);
            }
        }
        i += 1;
    }
    small.extend(large.into_iter().rev());
    small
}

/// Candidate block values (Welder DFS_smem_tile steps), ascending and deduped.
/// With `pow2` the last value is the next power of two at or above `ext`:
/// Triton masks the tail.
pub(super) fn block_values(ext: u64, pow2: bool) -> Vec<u64> {
    if ext <= 1 {
        return vec![1];
    }
    if pow2 {
        (0..)
            .map(|i| 1u64 << i)
            .take_while(|&v| v < ext)
            .chain([ext.next_power_of_two()])
            .collect()
    } else {
        let mut v = factors(ext);
        v.extend([2, 4, 8, 16, 32].into_iter().filter(|&s| 1 < s && s < ext));
        v.sort();
        v.dedup();
        v
    }
}

pub(super) fn knobs_for(cx: &Ctx, l: Lvl, anchor: NodeId, ext: &[u64]) -> Vec<Knob> {
    let pow2 = cx.caps.pow2_blocks();
    match style(cx, l) {
        KnobStyle::AnchorConsts => {
            let mut ks: Vec<Knob> = Vec::new();
            let Some(out) = meta(cx.g, anchor).spec().outputs.first().copied() else {
                return ks; // P9 unmet: `seed` rejects it
            };
            for b in out.axes {
                if b.block_const.parse::<u64>().is_ok() {
                    continue; // literal: fixed
                }
                if ks.iter().any(|k| k.name == KnobName::Block(b.block_const)) {
                    continue;
                }
                let e: u64 = b
                    .dims
                    .iter()
                    .map(|&d| ext.get(d).copied().unwrap_or(1))
                    .product(); // flattened extent
                ks.push(Knob {
                    name: KnobName::Block(b.block_const),
                    values: block_values(e, pow2),
                });
            }
            ks
        }
        KnobStyle::SinkDims => (0..ext.len())
            .map(|d| Knob {
                name: KnobName::Dim(d),
                values: block_values(ext[d], pow2),
            })
            .collect(),
    }
}

/// `θ[i]` where `knobs[i]` is the block const `c`.
pub(super) fn value_of(knobs: &[Knob], theta: &[u64], c: &str) -> Option<u64> {
    knobs
        .iter()
        .position(|k| matches!(k.name, KnobName::Block(b) if b == c))
        .map(|i| theta[i])
}

pub(super) fn zip_knobs(knobs: &[Knob], theta: &[u64]) -> Vec<(KnobName, u64)> {
    knobs
        .iter()
        .map(|k| k.name)
        .zip(theta.iter().copied())
        .collect()
}

/// θ → (unit grid, anchor output region). Err(NotBox) if a flattened binding can't be a box.
pub(super) fn seed(
    cx: &Ctx,
    l: Lvl,
    anchor: NodeId,
    ext: &[u64],
    knobs: &[Knob],
    theta: &[u64],
) -> Result<(UnitGrid, Region), Reject> {
    let mut grid = UnitGrid { axes: vec![] };
    let mut r: Region = vec![DimSpan::Full; ext.len()];
    match style(cx, l) {
        KnobStyle::SinkDims => {
            for d in 0..ext.len() {
                let a = push_axis(&mut grid, ext[d], theta[d]);
                r[d] = DimSpan::Tiled {
                    axis: a,
                    step: theta[d],
                    lo: 0,
                    len: theta[d],
                };
            }
        }
        KnobStyle::AnchorConsts => {
            let out = meta(cx.g, anchor)
                .spec()
                .outputs
                .first()
                .copied()
                .ok_or(Reject::Spec)?;
            if out.rank != ext.len() {
                return Err(Reject::Spec);
            }
            for b in out.axes {
                if b.dims.iter().any(|&d| d >= ext.len()) {
                    return Err(Reject::Spec);
                }
                let v = match b.block_const.parse::<u64>() {
                    Ok(v) => v,
                    Err(_) => value_of(knobs, theta, b.block_const).ok_or(Reject::Spec)?,
                };
                // P8 (a `walk` flag) is unmet, so every binding is boxed.
                rebox(&mut grid, &mut r, b.dims, ext, v)?;
            }
        }
    }
    Ok((grid, r))
}

/// A flat block `v` over dims D (outer..inner) as an axis-aligned box (summary §4 "re-boxing").
///
/// For a single dim there is no divisibility requirement: a ragged last tile is
/// masked. For flattened dims, `v` must be a whole number of inner rows and
/// must not straddle an outer index. A `v` covering the whole span is one
/// masked unit, checked first so `block_values`' `next_pow2` is always legal.
fn rebox(
    grid: &mut UnitGrid,
    r: &mut Region,
    dims: &[usize],
    ext: &[u64],
    v: u64,
) -> Result<(), Reject> {
    let m = dims.len();
    if v == 0 {
        return Err(Reject::NotBox);
    }
    if v >= dims.iter().map(|&d| ext[d]).product::<u64>() {
        for &d in dims {
            r[d] = DimSpan::Full;
        }
        return Ok(());
    }
    let mut inner = 1u64; // ∏ extents of the dims inside j
    for j in (0..m).rev() {
        if v % inner != 0 {
            return Err(Reject::NotBox);
        }
        let rj = v / inner;
        if rj <= ext[dims[j]] {
            if j > 0 && ext[dims[j]] % rj != 0 {
                return Err(Reject::NotBox); // would straddle an outer index
            }
            for &d in &dims[..j] {
                let a = push_axis(grid, ext[d], 1);
                r[d] = DimSpan::Tiled {
                    axis: a,
                    step: 1,
                    lo: 0,
                    len: 1,
                };
            }
            let a = push_axis(grid, ext[dims[j]], rj);
            r[dims[j]] = DimSpan::Tiled {
                axis: a,
                step: rj,
                lo: 0,
                len: rj,
            };
            for &d in &dims[j + 1..] {
                r[d] = DimSpan::Full;
            }
            return Ok(());
        }
        inner *= ext[dims[j]];
    }
    unreachable!("v < ∏ ext, so j = 0 has rj < ext[dims[0]]")
}

fn push_axis(g: &mut UnitGrid, extent: u64, block: u64) -> AxisId {
    g.axes.push(GridAxis {
        extent,
        block,
        reduce: false,
    });
    g.axes.len() - 1
}

#[cfg(test)]
mod tests {
    use super::*;
    use DimSpan::*;

    fn boxed(dims: &[usize], ext: &[u64], v: u64) -> Result<(UnitGrid, Region), Reject> {
        let mut grid = UnitGrid { axes: vec![] };
        let mut r = vec![Full; ext.len()];
        rebox(&mut grid, &mut r, dims, ext, v)?;
        Ok((grid, r))
    }

    fn tiled(axis: AxisId, step: u64) -> DimSpan {
        Tiled {
            axis,
            step,
            lo: 0,
            len: step,
        }
    }

    #[test]
    fn test_w6_a_flat_block_reboxes_only_when_it_tiles_whole_rows() {
        // ANALYSIS.md §12.5 W6: a flat BLOCK over (C, H, W) = (64, 56, 56).
        let ext = [64, 56, 56];

        let (grid, r) = boxed(&[0, 1, 2], &ext, 112).expect("112 is two whole rows of 56");
        assert_eq!(r, vec![tiled(0, 1), tiled(1, 2), Full]);
        assert_eq!(grid.units(), 64 * 28);

        assert!(
            matches!(boxed(&[0, 1, 2], &ext, 24), Err(Reject::NotBox)),
            "24 is not a whole number of 56-wide rows"
        );
    }

    #[test]
    fn test_a_block_covering_the_whole_span_is_one_masked_unit() {
        let (grid, r) = boxed(&[0, 1], &[3, 5], 16).expect("16 ≥ 15 covers the span");
        assert_eq!(r, vec![Full, Full]);
        assert_eq!(grid.units(), 1);
    }

    #[test]
    fn test_a_single_dim_block_may_leave_a_ragged_tail() {
        let (grid, r) = boxed(&[0], &[10], 4).expect("a single dim masks its tail");
        assert_eq!(r, vec![tiled(0, 4)]);
        assert_eq!(grid.units(), 3);
    }

    #[test]
    fn test_s1_w7_pow2_block_values_over_a_56_by_56_span() {
        // §12.7 S1 / §12.5 W7: a BLOCK_HW over H·W = 3136.
        let vals = block_values(56 * 56, true);
        assert_eq!(vals.first(), Some(&1));
        assert_eq!(vals.last(), Some(&4096));
        assert!(vals.contains(&8));
        assert!(vals.windows(2).all(|w| w[0] < w[1]));
        // 2, 4 and 8 divide a 56-wide row; 16 doesn't, so it can't be a box.
        for v in [1, 2, 4, 8, 56 * 2, 4096] {
            assert!(boxed(&[0, 1], &[56, 56], v).is_ok(), "{v} should rebox");
        }
        assert!(matches!(boxed(&[0, 1], &[56, 56], 16), Err(Reject::NotBox)));
    }

    #[test]
    fn test_s2_rank_zero_and_size_one_dims() {
        assert_eq!(block_values(1, true), vec![1]);
        assert_eq!(block_values(0, false), vec![1]);
        let (grid, r) = boxed(&[], &[], 1).expect("a scalar is one unit");
        assert!(r.is_empty());
        assert_eq!(grid.units(), 1);
        let (_, r) = boxed(&[0, 1], &[1, 8], 4).expect("a size-1 outer dim is fine");
        assert_eq!(r, vec![tiled(0, 1), tiled(1, 4)]);
    }

    #[test]
    fn test_s3_prime_extents_without_pow2() {
        assert_eq!(block_values(13, false), vec![1, 2, 4, 8, 13]);
        assert_eq!(block_values(12, false), vec![1, 2, 3, 4, 6, 8, 12]);
        assert_eq!(factors(36), vec![1, 2, 3, 4, 6, 9, 12, 18, 36]);
    }
}
