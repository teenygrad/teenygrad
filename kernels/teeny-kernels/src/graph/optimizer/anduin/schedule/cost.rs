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

//! Cost of one tiling: traffic, footprint, occupancy (A10, A16), and the
//! analytic [`Evaluator`].

use std::collections::{BTreeMap, BTreeSet};

use teeny_core::graph::DtypeRepr;

use super::seed::UnitGrid;
use super::span::{DimSpan, Region, ceil_div, drives, lens_fp, lens_traffic, union_region};
use super::{
    Ctx, Evaluator, GroupSchedule, Hierarchy, Lvl, Owner, Reject, meta_opt, partition,
    topo_sort_quotient,
};
use crate::graph::optimizer::anduin::tile_graph::{EdgeId, NodeId, TileGraph};

pub(super) fn dtype_bytes(dt: DtypeRepr) -> u64 {
    match dt {
        DtypeRepr::Bool | DtypeRepr::I8 | DtypeRepr::U8 => 1,
        DtypeRepr::I16 | DtypeRepr::U16 | DtypeRepr::F16 | DtypeRepr::BF16 => 2,
        DtypeRepr::I32 | DtypeRepr::U32 | DtypeRepr::F32 => 4,
        DtypeRepr::I64 | DtypeRepr::U64 | DtypeRepr::F64 => 8,
    }
}

fn bytes(lens: &[u64], dt: DtypeRepr) -> u64 {
    lens.iter().product::<u64>() * dtype_bytes(dt)
}

fn dtype(cx: &Ctx, n: NodeId) -> DtypeRepr {
    cx.g.nodes[n.0].dtype
}

/// Welder `coalesced_factor`: the contiguous run length of a sub-box in the full box.
pub(super) fn contiguous_run(lens: &[u64], ext: &[u64]) -> u64 {
    let mut run = 1;
    for d in (0..lens.len()).rev() {
        run *= lens[d];
        if lens[d] != ext[d] {
            break;
        }
    }
    run
}

/// Welder `coalesced_tensor_shape`: elements moved, inflated when the contiguous
/// run is shorter than one transaction of `txn` elements.
fn coalesced_elems(lens: &[u64], ext: &[u64], txn: u64) -> f64 {
    let n: u64 = lens.iter().product();
    if n == 0 {
        return 0.0;
    }
    n as f64 * txn as f64 / (txn.min(contiguous_run(lens, ext)).max(1) as f64)
}

fn coalesced_bytes(lens: &[u64], ext: &[u64], dt: DtypeRepr, txn_bytes: u64) -> f64 {
    let eb = dtype_bytes(dt);
    coalesced_elems(lens, ext, (txn_bytes / eb).max(1)) * eb as f64
}

/// The slice of one tensor a unit owns: each tile writes `step` of a tiled dim,
/// so overlapping (halo) regions are not stored twice.
fn owned(r: &Region, ext: &[u64]) -> Vec<u64> {
    r.iter()
        .zip(ext)
        .map(|(s, &e)| match *s {
            DimSpan::Tiled { step, .. } => step.min(e),
            _ => e,
        })
        .collect()
}

/// Bytes one tensor produced in the group writes to a slower level: only the
/// units that own a slice write, and each writes its non-overlapping slice.
fn store_bytes(r: &Region, ext: &[u64], dt: DtypeRepr, grid: &UnitGrid, txn: u64) -> f64 {
    let owners: u64 = drives(r).iter().map(|&a| grid.axes[a].units()).product();
    owners as f64 * coalesced_bytes(&owned(r, ext), ext, dt, txn)
}

/// The level an external operand of the current tiling is read from, given
/// its ancestors `parents` (outermost first).
fn read_level(cx: &Ctx, parents: &[&GroupSchedule], e: EdgeId) -> Lvl {
    for p in parents.iter().rev() {
        if p.staged.contains(&e) {
            return p.level + 1;
        }
        if p.nodes.contains(&cx.g.edges[e.0].src) {
            return cx.plan.level[e.0]; // sibling stage's buffer
        }
    }
    0 // placeholders and other kernels' outputs
}

/// The stages a tiling of `s.nodes` at `s.level` runs, in execution order:
/// the level-`l+1` components when there is a level between, and each node in
/// turn inside a lockstep group (`l + 1 == H`).
fn stages_of(cx: &Ctx, s: &GroupSchedule) -> Vec<Vec<NodeId>> {
    if s.level + 1 < cx.hw.top() {
        topo_sort_quotient(cx, &partition(cx, cx.plan, s.level + 1, Some(&s.nodes)))
    } else {
        s.nodes.iter().map(|&n| vec![n]).collect()
    }
}

/// One staged external tensor of a stage: its producer, the union of its
/// staged edges' regions, its extents, and those edges.
struct StagedTensor {
    src: NodeId,
    region: Region,
    extent: Vec<u64>,
    edges: Vec<EdgeId>,
}

fn staged_operands_by_tensor(cx: &Ctx, s: &GroupSchedule, st: &[NodeId]) -> Vec<StagedTensor> {
    let mut by_src: BTreeMap<NodeId, StagedTensor> = BTreeMap::new();
    for &e in &s.staged {
        let edge = &cx.g.edges[e.0];
        if !st.contains(&edge.dst) {
            continue;
        }
        let r = &s.in_region[&e];
        by_src
            .entry(edge.src)
            .and_modify(|t| {
                t.region = union_region(&t.region, r);
                t.edges.push(e);
            })
            .or_insert_with(|| StagedTensor {
                src: edge.src,
                region: r.clone(),
                extent: s.operand_extent[&e].clone(),
                edges: vec![e],
            });
    }
    by_src.into_values().collect()
}

/// The number of distinct later stages that read `n` over a level-`k` edge.
fn later_stages_reading_at(
    cx: &Ctx,
    stages: &[Vec<NodeId>],
    at: usize,
    n: NodeId,
    k: Lvl,
) -> usize {
    stages[at + 1..]
        .iter()
        .filter(|st| {
            cx.g.nodes[n.0].out_edges.iter().any(|e| {
                let edge = &cx.g.edges[e.0];
                cx.plan.level[e.0] == k && st.contains(&edge.dst)
            })
        })
        .count()
}

/// The distinct live tensors that stage `st` reads over a level-`k` edge.
fn tensors_read_by_stage_from(
    cx: &Ctx,
    st: &[NodeId],
    live: &BTreeMap<NodeId, (usize, usize)>,
    k: Lvl,
) -> BTreeSet<NodeId> {
    st.iter()
        .flat_map(|&x| cx.g.nodes[x.0].in_edges.iter())
        .map(|e| &cx.g.edges[e.0])
        .filter(|e| cx.plan.level[e.id.0] == k && live.contains_key(&e.src))
        .map(|e| e.src)
        .collect()
}

/// Welder `bestfit.py`: a best-fit offset allocator. Every block is rounded up
/// to 32 bytes, which the footprint relies on.
#[derive(Default)]
struct BestFit {
    blocks: Vec<Block>, // contiguous, by offset
    limit: u64,
    next_id: usize,
}

struct Block {
    start: u64,
    end: u64,
    owner: Option<usize>, // None = free
}

impl BestFit {
    const ALIGN: u64 = 32;

    fn malloc(&mut self, size: u64) -> usize {
        let size = size.div_ceil(Self::ALIGN) * Self::ALIGN;
        let id = self.next_id;
        self.next_id += 1;
        let found = self
            .blocks
            .iter()
            .enumerate()
            .filter(|(_, b)| b.owner.is_none() && b.end - b.start >= size)
            .min_by_key(|(_, b)| b.end - b.start)
            .map(|(i, _)| i);
        if let Some(i) = found {
            let b = &mut self.blocks[i];
            b.owner = Some(id);
            let remain = b.end - b.start - size;
            if remain != 0 {
                b.end -= remain;
                let start = b.end;
                self.blocks.insert(
                    i + 1,
                    Block {
                        start,
                        end: start + remain,
                        owner: None,
                    },
                );
            }
        } else if let Some(last) = self.blocks.last_mut()
            && last.owner.is_none()
        {
            last.end = last.start + size;
            last.owner = Some(id);
            self.limit = last.end;
        } else {
            self.blocks.push(Block {
                start: self.limit,
                end: self.limit + size,
                owner: Some(id),
            });
            self.limit += size;
        }
        id
    }

    fn free(&mut self, id: usize) {
        let Some(i) = self.blocks.iter().position(|b| b.owner == Some(id)) else {
            return;
        };
        self.blocks[i].owner = None;
        if i + 1 < self.blocks.len() && self.blocks[i + 1].owner.is_none() {
            self.blocks[i].end = self.blocks[i + 1].end;
            self.blocks.remove(i + 1);
        }
        if i > 0 && self.blocks[i - 1].owner.is_none() {
            self.blocks[i - 1].end = self.blocks[i].end;
            self.blocks.remove(i);
        }
    }

    fn high_water(&self) -> u64 {
        self.limit
    }
}

/// Peak bytes of level `l+1` one unit holds (Welder `_compute_shared_memory_usage`,
/// fix #12). Codegen must use the same allocator for the offsets to be valid.
pub(super) fn footprint(cx: &Ctx, s: &GroupSchedule) -> u64 {
    let (l, top) = (s.level, cx.hw.top());
    let mult = if cx.caps.double_buffer() { 2 } else { 1 };
    let stages = stages_of(cx, s);
    let mut alloc = BestFit::default();
    let mut live: BTreeMap<NodeId, (usize /* block */, usize /* readers left */)> = BTreeMap::new();
    for (i, st) in stages.iter().enumerate() {
        let mut transient = Vec::new();
        // 1. staged external operands of this stage (one buffer per tensor)
        for t in staged_operands_by_tensor(cx, s, st) {
            transient.push(alloc.malloc(bytes(
                &lens_fp(&t.region, &t.extent, mult),
                dtype(cx, t.src),
            )));
        }
        // 2. this stage's outputs, allocated before anything is freed: the stage
        //    is still reading its inputs while it writes them.
        for &n in st {
            let readers = later_stages_reading_at(cx, &stages, i, n, l + 1);
            let b = bytes(&lens_fp(&s.region[&n], &s.extent[&n], 1), dtype(cx, n));
            if readers > 0 {
                live.insert(n, (alloc.malloc(b), readers));
            } else if l + 1 == top {
                transient.push(alloc.malloc(b));
            }
        }
        for b in transient {
            alloc.free(b);
        }
        // 3. free intermediates whose last reader is this stage
        for t in tensors_read_by_stage_from(cx, st, &live, l + 1) {
            if st.contains(&t) {
                continue;
            }
            let entry = live.get_mut(&t).expect("read from `live`");
            entry.1 -= 1;
            if entry.1 == 0 {
                let (b, _) = live.remove(&t).expect("present above");
                alloc.free(b);
            }
        }
    }
    alloc.high_water()
}

/// This tiling's own traffic per level (A10): staged loads per stage, and
/// stores of tensors leaving the group when this is the last tiling level.
fn own_traffic(cx: &Ctx, s: &GroupSchedule, parents: &[&GroupSchedule]) -> Vec<f64> {
    let levels = &cx.hw.levels;
    let mut traffic = vec![0.0; levels.len()];
    let units = s.grid.units() as f64;
    for st in stages_of(cx, s) {
        for t in staged_operands_by_tensor(cx, s, &st) {
            let rl = t
                .edges
                .iter()
                .map(|&e| read_level(cx, parents, e))
                .min()
                .expect("a staged tensor has an edge");
            let trips = t
                .edges
                .iter()
                .map(|e| s.op_trips.get(e).copied().unwrap_or(1))
                .max()
                .unwrap_or(1);
            traffic[rl] += units
                * trips as f64
                * coalesced_bytes(
                    &lens_traffic(&t.region, &t.extent),
                    &t.extent,
                    dtype(cx, t.src),
                    levels[rl].read_txn_bytes,
                );
        }
    }
    if s.level + 1 == cx.hw.top() {
        for &n in &s.nodes {
            let ks: BTreeSet<Lvl> = cx.g.nodes[n.0]
                .out_edges
                .iter()
                .filter(|e| !s.nodes.contains(&cx.g.edges[e.0].dst))
                .map(|e| cx.plan.level[e.0])
                .collect();
            let trips = s.trips.get(&n).copied().unwrap_or(1) as f64;
            for k in ks {
                traffic[k] += trips
                    * store_bytes(
                        &s.region[&n],
                        &s.extent[&n],
                        dtype(cx, n),
                        &s.grid,
                        levels[k].write_txn_bytes,
                    );
            }
        }
    }
    traffic
}

/// Welder's ranking traffic at this level, per unit (not per grid: `key`
/// multiplies by `waves` as well).
fn key_traffic(cx: &Ctx, s: &GroupSchedule, parents: &[&GroupSchedule]) -> f64 {
    let l = s.level;
    let lv = &cx.hw.levels[l];
    let mut t = 0.0;
    for (&e, r) in &s.in_region {
        let src = cx.g.edges[e.0].src;
        if s.nodes.contains(&src) || read_level(cx, parents, e) != l {
            continue;
        }
        let ext = &s.operand_extent[&e];
        t += coalesced_bytes(
            &lens_traffic(r, ext),
            ext,
            dtype(cx, src),
            lv.read_txn_bytes,
        );
    }
    for &n in &s.nodes {
        let leaves = cx.g.nodes[n.0]
            .out_edges
            .iter()
            .any(|e| !s.nodes.contains(&cx.g.edges[e.0].dst) && cx.plan.level[e.0] == l);
        if leaves {
            let ext = &s.extent[&n];
            t += coalesced_bytes(
                &owned(&s.region[&n], ext),
                ext,
                dtype(cx, n),
                lv.write_txn_bytes,
            );
        }
    }
    t
}

/// Occupancy and the search key (A10, A16). The register terms apply only when
/// level `l+2` exists and is Thread-owned, never over device and shared memory.
pub(super) fn occupancy_and_key(
    cx: &Ctx,
    s: &mut GroupSchedule,
    parents: &[&GroupSchedule],
) -> Result<(), Reject> {
    let hw = cx.hw;
    let (l, fp, u) = (s.level, s.cost.footprint, s.grid.units());
    let into = &hw.levels[l + 1];
    if fp > into.capacity_per_owner {
        return Err(Reject::Capacity);
    }
    let threads = hw.levels.get(l + 2).filter(|lv| lv.owner == Owner::Thread);
    let reg_est = match threads {
        Some(_) => {
            2 * s
                .nodes
                .iter()
                .map(|n| bytes(&lens_fp(&s.region[n], &s.extent[n], 1), dtype(cx, *n)))
                .max()
                .unwrap_or(0)
        }
        None => 0,
    };
    if let Some(t) = threads
        && reg_est > t.capacity_per_cu
    {
        return Err(Reject::Registers);
    }
    let upc = match hw.owner_of_units(l) {
        Owner::Block => (into.capacity_per_cu / fp.max(1))
            .min(threads.map_or(u64::MAX, |t| t.capacity_per_cu / reg_est.max(1)))
            .min(hw.max_units_per_cu as u64),
        Owner::Device => 1,
        // Thread units are laid out by A13's fan-out, never searched here.
        Owner::Thread => return Err(Reject::Occupancy),
    };
    if upc == 0 {
        return Err(Reject::Occupancy);
    }
    let waves = match hw.owner_of_units(l) {
        Owner::Device => u,
        _ => ceil_div(u, upc * hw.compute_units as u64),
    };
    s.cost.units = u;
    s.cost.units_per_cu = upc;
    s.cost.waves = waves;
    s.cost.reg_est = reg_est;
    s.cost.key_traffic = key_traffic(cx, s, parents);
    s.cost.key = (s.cost.key_traffic + 1.0) * waves as f64;
    Ok(())
}

/// A12 `finish_cost`, the leaf case (no children: always so when `l + 1 == H`).
pub(super) fn finish_cost(cx: &Ctx, s: &mut GroupSchedule, parents: &[&GroupSchedule]) {
    s.cost.traffic = own_traffic(cx, s, parents);
    let space = s.grid.space_units() as f64;
    s.cost.flops = s
        .nodes
        .iter()
        .map(|&n| {
            let per_out = meta_opt(cx.g, n).map_or(1.0, |m| m.flops_per_output);
            let trips = s.trips.get(&n).copied().unwrap_or(1) as f64;
            let lens: u64 = lens_traffic(&s.region[&n], &s.extent[&n]).iter().product();
            trips * per_out * space * lens as f64
        })
        .sum();
    s.cost.est = cx.ev.estimate(cx.g, cx.hw, s);
}

/// The default analytic [`Evaluator`] (A10, A16): no measurements.
pub(super) struct AnalyticEvaluator;

impl Evaluator for AnalyticEvaluator {
    fn estimate(&self, _g: &TileGraph, hw: &Hierarchy, s: &GroupSchedule) -> f64 {
        let t_mem = s
            .cost
            .traffic
            .iter()
            .zip(&hw.levels)
            .map(|(&t, lv)| if t == 0.0 { 0.0 } else { t / lv.bandwidth })
            .fold(0.0, f64::max);
        let t_comp = s.cost.flops / hw.flops_per_s;
        let quant = (s.cost.waves * s.cost.units_per_cu * hw.compute_units as u64) as f64
            / s.cost.units.max(1) as f64;
        t_mem.max(t_comp) * quant.max(1.0) + hw.launch_overhead_s
    }

    fn measure(&self, _g: &TileGraph, _hw: &Hierarchy, _s: &GroupSchedule) -> Option<f64> {
        None
    }

    /// Like-for-like with `estimate`: every operand read and the output
    /// written once from level 0, or the flops, whichever takes longer.
    fn opaque(&self, g: &TileGraph, hw: &Hierarchy, n: NodeId) -> f64 {
        let numel = |m: NodeId| g.nodes[m.0].shape.iter().product::<u64>();
        let size = |m: NodeId| numel(m) * dtype_bytes(g.nodes[m.0].dtype);
        let reads: u64 = g.nodes[n.0]
            .in_edges
            .iter()
            .map(|e| size(g.edges[e.0].src))
            .sum();
        let t_mem = (reads + size(n)) as f64 / hw.levels[0].bandwidth;
        let per_out = meta_opt(g, n).map_or(1.0, |m| m.flops_per_output);
        let t_comp = per_out * numel(n) as f64 / hw.flops_per_s;
        t_mem.max(t_comp) + hw.launch_overhead_s
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_contiguous_run_stops_at_the_first_partial_dim() {
        assert_eq!(contiguous_run(&[4, 8, 16], &[4, 8, 16]), 4 * 8 * 16);
        assert_eq!(contiguous_run(&[2, 8, 16], &[4, 8, 16]), 2 * 8 * 16);
        assert_eq!(contiguous_run(&[2, 4, 16], &[4, 8, 16]), 4 * 16);
        assert_eq!(contiguous_run(&[4, 8, 2], &[4, 8, 16]), 2);
    }

    #[test]
    fn test_a_short_contiguous_run_is_charged_a_whole_transaction() {
        // A column of 4 f32s in a [4, 64] tensor, read in 128-byte (32-element)
        // transactions: each element costs a transaction.
        assert_eq!(
            coalesced_bytes(&[4, 1], &[4, 64], DtypeRepr::F32, 128),
            4.0 * 128.0
        );
        // A whole row is one long run, so nothing is wasted.
        assert_eq!(
            coalesced_bytes(&[1, 64], &[4, 64], DtypeRepr::F32, 128),
            256.0
        );
        // With no coalescing (txn = 1 byte) the bytes are the elements'.
        assert_eq!(coalesced_bytes(&[4, 1], &[4, 64], DtypeRepr::F32, 1), 16.0);
    }

    #[test]
    fn test_with_one_byte_writes_stored_bytes_are_the_tensor_size() {
        // A [64, 1024] f32 tensor tiled [8, 128]: 8 · 8 units, each storing its
        // own slice once.
        let grid = UnitGrid {
            axes: vec![
                super::super::seed::GridAxis {
                    extent: 64,
                    block: 8,
                    reduce: false,
                },
                super::super::seed::GridAxis {
                    extent: 1024,
                    block: 128,
                    reduce: false,
                },
            ],
        };
        let halo = vec![
            DimSpan::Tiled {
                axis: 0,
                step: 8,
                lo: -1,
                len: 10,
            },
            DimSpan::Tiled {
                axis: 1,
                step: 128,
                lo: 0,
                len: 128,
            },
        ];

        assert_eq!(
            store_bytes(&halo, &[64, 1024], DtypeRepr::F32, &grid, 1),
            (64 * 1024 * 4) as f64
        );
    }

    #[test]
    fn test_bestfit_reuses_freed_blocks_and_rounds_to_32_bytes() {
        let mut a = BestFit::default();
        let x = a.malloc(10); // [0, 32)
        let y = a.malloc(64); // [32, 96)
        assert_eq!(a.high_water(), 96);
        a.free(x);
        let z = a.malloc(20); // fits in x's freed 32
        assert_eq!(a.high_water(), 96);
        a.free(y);
        a.free(z);
        // Everything is free and merged: one 96-byte block, grown at the end.
        a.malloc(128);
        assert_eq!(a.high_water(), 128);
    }

    #[test]
    fn test_bestfit_picks_the_smallest_free_block_that_fits() {
        let mut a = BestFit::default();
        let big = a.malloc(128); // [0, 128)
        let _sep = a.malloc(32); // [128, 160)
        let small = a.malloc(32); // [160, 192)
        let _end = a.malloc(32); // [192, 224)
        a.free(big);
        a.free(small);
        a.malloc(32); // best fit is `small`'s 32, not `big`'s 128
        a.malloc(128); // so `big` is still whole
        assert_eq!(a.high_water(), 224);
    }
}
