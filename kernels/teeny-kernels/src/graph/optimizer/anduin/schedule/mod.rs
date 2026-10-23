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

//! The connection-plan search of ANALYSIS.md §12: Welder §3.2's
//! `GraphConnecting` (Fig. 7), with the paper's defects fixed.
//!
//! A [`ConnectPlan`] gives every edge the memory level its value is handed
//! over in: 0 is an ordinary kernel boundary, and `H = hw.top()` is the
//! compute level. [`schedule`] walks the edges in priority order (A15),
//! proposes each of an edge's legal levels (A1), closes a raise over
//! convexity (A4) so the plan stays valid (A3), and accepts the proposal that
//! most lowers the summed latency of the level-0 groups it changes (A14).
//!
//! ## Status
//!
//! A1-A12, A14 and A15 are implemented for a hierarchy whose tiling stages
//! straight into the compute level (`l + 1 == H`), which device and shared
//! memory always is. Over a deeper hierarchy [`tile_group`] needs A13's
//! children, which are not implemented, so it reports no candidates there and
//! every multi-node group stays infeasible.
//!
//! Two preconditions of §12.2 are unmet in `teeny-core`, and the search falls
//! back conservatively: no binding is walked (P8), and no reduce dim has an
//! rstep (P6), so both are read whole.
//!
//! ## Scope: scheduling only
//!
//! This does not rewrite the DAG into actually-fused kernels. Turning a
//! finished schedule into that is blocked on reworking `#[tiled_kernel]` to
//! compose `Tile<D>` functions instead of pointer-in/pointer-out kernels
//! (teenygrad-1nr.1; `pid` decode doesn't currently compose across a fused
//! call).

use std::collections::{BTreeMap, BTreeSet, HashMap, VecDeque};

use teeny_core::device::hardware::MemoryLevelKind;
use union_find::{QuickUnionUf, UnionBySize, UnionFind};

use crate::errors::Result;
use crate::graph::optimizer::anduin::tile_graph::{Edge, NodeKind};

use super::tile_graph::TileGraph;
use super::tile_graph::{EdgeId, NodeId};

mod cost;
mod legality;
mod propagate;
mod seed;
mod span;
#[cfg(test)]
mod testing;
mod tiling;

use seed::{KnobName, KnobStyle, UnitGrid};
use span::{RStep, Region};
use tiling::tile_group;

// ───────────────────────────── 1. Memory hierarchy ─────────────────────────────

/// Index into `Hierarchy::levels`. 0 = slowest *schedulable* level; same direction
/// as `MemoryLevel::level`, but hardware-managed caches (L1/L2/L3) are dropped
/// because nothing can be connected "at" a cache.
pub type Lvl = usize;

/// Who owns one instance of a level. A tiling at level `l` runs one unit per
/// owner of level `l+1` (block tiles staged in shared memory are run by blocks).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Owner {
    Device,
    Block,
    Thread,
}

pub struct LevelInfo {
    pub kind: MemoryLevelKind,
    pub owner: Owner,
    /// Bytes in ONE instance: shared memory per block, register bytes per thread.
    pub capacity_per_owner: u64,
    /// Bytes per compute unit, used for occupancy; `u64::MAX` when not per-CU.
    pub capacity_per_cu: u64,
    /// Bytes/s into or out of this level.
    pub bandwidth: f64,
    /// Coalescing granularity for reads from / writes to this level (Welder
    /// `transaction_size = [32 write, 128 read]` for global; 1 = none).
    pub read_txn_bytes: u64,
    pub write_txn_bytes: u64,
    /// Number of next-faster owners one owner of this level can be split into
    /// (threads per block), and whether all child groups of one unit must use the
    /// SAME number (true for threads per block; false for blocks per kernel).
    pub max_fanout: u32,
    pub fixed_fanout: bool,
}

pub struct Hierarchy {
    pub levels: Vec<LevelInfo>, // slowest → fastest; the last one is the compute level
    pub compute_units: u32,     // SMs
    pub max_units_per_cu: u32,  // Welder `sm_partition` (= 4)
    pub simt_width: u32,        // warp size
    pub flops_per_s: f64,
    pub launch_overhead_s: f64,
}
impl Hierarchy {
    pub fn top(&self) -> Lvl {
        self.levels.len() - 1
    }
    pub fn owner_of_units(&self, l: Lvl) -> Owner {
        self.levels[l + 1].owner
    }
}

// ───────────────────────────── 3. Connection plan ─────────────────────────────

/// `level[e]` = the memory level the value on edge `e` is handed over in. It is
/// never written to any slower level FOR THIS EDGE. 0 = materialised in the
/// slowest level (an ordinary kernel boundary); H = lockstep in registers.
/// Kept outside the graph so the search can clone and compare plans.
#[derive(Clone, PartialEq, Eq, Hash)]
pub struct ConnectPlan {
    pub level: Vec<Lvl>,
}

/// Connected components of the op nodes under edges with `level > l`.
pub struct Partition {
    pub group_of: Vec<Option<usize>>, // by NodeId; None for Placeholder/Output or outside `within`
    pub groups: Vec<Vec<NodeId>>,     // members sorted by `pos`
}

// ──────────────────────── 6. Schedules (the paper's TileConfig) ────────────────────────

/// Everything decided for one group at one tiling level, plus its stages.
#[derive(Clone)]
struct GroupSchedule {
    pub level: Lvl,         // tiles are staged into level + 1
    pub nodes: Vec<NodeId>, // sorted by pos
    pub anchor: NodeId,
    pub knobs: Vec<(KnobName, u64)>, // level-0: anchor consts; thread level: empty
    pub grid: UnitGrid,
    pub extent: BTreeMap<NodeId, Vec<u64>>, // each node's extents at THIS level (parent-relative)
    pub operand_extent: BTreeMap<EdgeId, Vec<u64>>, // each in-edge's operand extents at this level
    pub trips: BTreeMap<NodeId, u64>, // times node n's work here repeats per root unit (walks above)
    pub op_trips: BTreeMap<EdgeId, u64>, // times operand e is re-read per root unit (walks + rsteps above)
    pub region: BTreeMap<NodeId, Region>, // output region per unit
    pub in_region: BTreeMap<EdgeId, Region>, // operand region per unit, every in-edge of every node
    pub rstep: BTreeMap<NodeId, RStep>,
    pub staged: BTreeSet<EdgeId>, // external operands copied into level + 1
    pub fanout: u32,              // owners of level + 2 per unit; 1 if level + 1 == H
    pub children: Vec<GroupSchedule>, // stages = partition(level + 1) of `nodes`, topo order
    pub cost: Cost,
}

#[derive(Clone, Default)]
struct Cost {
    pub units: u64,
    pub units_per_cu: u64,
    pub waves: u64,
    pub key_traffic: f64,  // Welder's ranking traffic at this level (A10)
    pub traffic: Vec<f64>, // bytes per level, whole subtree, charged once (A10)
    pub footprint: u64,    // peak bytes of level + 1 per unit
    pub reg_est: u64,      // Welder's pre-thread register estimate (bytes per unit)
    pub flops: f64,        // includes redundancy
    pub key: f64,          // (key_traffic + 1) · waves
    pub est: f64,          // Evaluator::estimate of the subtree
    pub measured: Option<f64>,
}

#[derive(Clone)]
enum KernelChoice {
    Tiled(GroupSchedule),
    Opaque(NodeId),
}

#[derive(Clone, Hash, PartialEq, Eq)]
struct GroupKey {
    nodes: Vec<NodeId>,
    internal_levels: Vec<(EdgeId, Lvl)>,
}

#[derive(Clone)]
struct Lat {
    est: f64,
    measured: Option<f64>,
    choice: Option<KernelChoice>,
}
struct Schedule {
    pub plan: ConnectPlan,
    pub kernels: Vec<KernelChoice>, /* execution order */
}

// ─────────────────────── 7. Parameters, capabilities, evaluator ───────────────────────

struct SearchParams {
    pub top_k: usize,            // Welder --topk (10)
    pub max_visited: usize,      // 2000  (DFS_smem_tile)
    pub fanouts_per_tile: usize, // 1     (assign_block_size topk=1)
    pub max_closure: usize,      // 10    (_build_fusion_group dependency cap)
    pub max_recompute: f64, // 1.0   (≈ Welder's grid-equality rule; raise to allow halo fusion)
    pub min_gain: f64,      // 0.0   (relative improvement required to accept)
    pub max_passes: usize,  // 1     (the paper and Welder both do one greedy pass)
    pub max_measurements: usize, // budget for Evaluator::measure
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum ThreadTiling {
    Explicit,
    Delegated, /* e.g. Triton: only num_warps is chosen */
}

/// What codegen can realise. teenygrad today, which schedules device and
/// shared memory only (so H = 1): `fuse_level(e, 1) = dst is a rider passing
/// the rider check`, `Delegated`, `AnchorConsts`, `pow2_blocks = true`,
/// `halo_stage = false`.
/// Welder: `fuse_level(e,1) = true`, `fuse_level(e,H) = src.reduce && !dst.reduce`,
/// `Explicit`, `SinkDims`, `halo_stage = true`, `cross_unit_reduce = true`.
trait Capabilities {
    fn fuse_level(&self, g: &TileGraph, e: &Edge, k: Lvl) -> bool; // 1 ≤ k ≤ H

    /// Copy external operand `e` into level `into` (< H) before use? (Welder
    /// "cached tensors", `IRNode.footprint`.) Default, as Welder: the consumer has
    /// a reduction, or the operand is broadcast (lower rank than the consumer's
    /// output). Addition, not in Welder: the operand region has a halo
    /// (`len > step` somewhere), so overlapping reads by neighbours are served
    /// from the staged copy.
    fn stage_input(&self, g: &TileGraph, e: &Edge, into: Lvl, r: &Region) -> bool;

    fn halo_stage(&self) -> bool; // a non-root stage anchor may compute an offset/overlapping region
    fn cross_unit_reduce(&self) -> bool; // a reduce chunk may be split across sub-units
    fn pow2_blocks(&self) -> bool; // Triton tl.arange
    fn double_buffer(&self) -> bool; // staged chunks are double-buffered
    fn thread_tiling(&self) -> ThreadTiling;
    fn knob_style(&self) -> KnobStyle;
}

/// teenygrad's codegen today: a kernel can only splice a rider (an op passing
/// the pointwise-fuse probe) after its own body, so an edge fuses only into a
/// rider. It schedules device and shared memory only, so tilings always stage
/// into the compute level and `stage_input` is never asked.
struct TeenygradCaps;

impl Capabilities for TeenygradCaps {
    fn fuse_level(&self, g: &TileGraph, e: &Edge, _k: Lvl) -> bool {
        meta_opt(g, e.dst).is_some_and(|m| m.rider())
    }
    fn stage_input(&self, _g: &TileGraph, _e: &Edge, _into: Lvl, _r: &Region) -> bool {
        true
    }
    fn halo_stage(&self) -> bool {
        false
    }
    fn cross_unit_reduce(&self) -> bool {
        false
    }
    fn pow2_blocks(&self) -> bool {
        true
    }
    fn double_buffer(&self) -> bool {
        false
    }
    fn thread_tiling(&self) -> ThreadTiling {
        ThreadTiling::Delegated
    }
    fn knob_style(&self) -> KnobStyle {
        KnobStyle::AnchorConsts
    }
}

trait Evaluator {
    /// Cheap and always available. Used for ranking.
    fn estimate(&self, g: &TileGraph, hw: &Hierarchy, s: &GroupSchedule) -> f64;
    /// Compile and run; None if unavailable. Used only on the final top-K.
    fn measure(&self, g: &TileGraph, hw: &Hierarchy, s: &GroupSchedule) -> Option<f64>;
    /// A node that cannot be tiled, run as its own kernel.
    fn opaque(&self, g: &TileGraph, hw: &Hierarchy, n: NodeId) -> f64;
}

struct Ctx<'a> {
    pub g: &'a TileGraph,
    pub hw: &'a Hierarchy,
    pub caps: &'a dyn Capabilities,
    pub params: &'a SearchParams,
    pub ev: &'a dyn Evaluator,
    pub plan: &'a ConnectPlan,
}

#[derive(Debug, PartialEq, Eq)]
enum Reject {
    Spec,
    NotBox,
    SinkShape,
    Containment,
    Lockstep(EdgeId),
    Recompute(NodeId),
    Coverage(NodeId),
    Unrealisable(NodeId),
    Capacity,
    Registers,
    Occupancy,
    NoFanout,
}
pub enum Violation {
    ShortCircuit { edge: EdgeId, level: Lvl },
    Cycle { level: Lvl, groups: Vec<usize> },
}

fn out_edges(g: &TileGraph, n: NodeId) -> impl Iterator<Item = &Edge> {
    g.nodes[n.0].out_edges.iter().map(|e| &g.edges[e.0])
}

fn op_nodes(g: &TileGraph) -> impl Iterator<Item = NodeId> + '_ {
    g.order
        .iter()
        .copied()
        .filter(|&n| g.nodes[n.0].kind == NodeKind::Op)
}

fn successors(g: &TileGraph, n: NodeId) -> impl Iterator<Item = NodeId> + '_ {
    out_edges(g, n).map(|e| e.dst)
}

/// Some `m ∈ M` has a directed path to `x` (including `x ∈ M` via a zero-length path).
fn reachable_from_set(g: &TileGraph, m: &BTreeSet<NodeId>, x: NodeId) -> bool {
    if m.contains(&x) {
        return true;
    }
    let mut seen: BTreeSet<NodeId> = m.clone();
    let mut queue: VecDeque<NodeId> = m.iter().copied().collect();
    while let Some(n) = queue.pop_front() {
        for succ in successors(g, n) {
            if succ == x {
                return true;
            }
            if seen.insert(succ) {
                queue.push_back(succ);
            }
        }
    }
    false
}

/// `x` has a directed path to some `m ∈ M` (including `x ∈ M`).
fn reaches_set(g: &TileGraph, x: NodeId, m: &BTreeSet<NodeId>) -> bool {
    if m.contains(&x) {
        return true;
    }
    let mut seen = BTreeSet::from([x]);
    let mut queue = VecDeque::from([x]);
    while let Some(n) = queue.pop_front() {
        for succ in successors(g, n) {
            if m.contains(&succ) {
                return true;
            }
            if seen.insert(succ) {
                queue.push_back(succ);
            }
        }
    }
    false
}

fn changed_edges(p: &ConnectPlan, q: &ConnectPlan) -> impl Iterator<Item = EdgeId> {
    p.level
        .iter()
        .zip(&q.level)
        .enumerate()
        .filter(|(_, (a, b))| a != b)
        .map(|(i, _)| EdgeId(i))
}

fn legal_levels(cx: &Ctx, e: &Edge) -> Vec<Lvl> {
    let (p, c) = (&cx.g.nodes[e.src.0], &cx.g.nodes[e.dst.0]);
    // placeholder / output edges
    if p.kind != NodeKind::Op || c.kind != NodeKind::Op {
        return vec![0];
    }

    // opaque endpoint
    let (Some(_), Some(cm)) = (&p.meta, &c.meta) else {
        return vec![0];
    };

    // P2. A port the spec doesn't declare (P5 unmet) cannot be fused either.
    if cm
        .spec()
        .inputs
        .get(e.dst_port)
        .is_none_or(|t| t.rank != p.shape.len())
    {
        return vec![0];
    }

    let mut ks = vec![0];
    for k in 1..=cx.hw.top() {
        if cx.caps.fuse_level(cx.g, e, k) {
            ks.push(k);
        }
    }
    ks
}

fn partition(cx: &Ctx, plan: &ConnectPlan, l: Lvl, within: Option<&[NodeId]>) -> Partition {
    let inside =
        |n: NodeId| cx.g.nodes[n.0].kind == NodeKind::Op && within.map_or(true, |w| w.contains(&n));
    let mut uf = QuickUnionUf::<UnionBySize>::new(cx.g.nodes.len());

    for e in &cx.g.edges {
        if inside(e.src) && inside(e.dst) && plan.level[e.id.0] > l {
            uf.union(e.src.0, e.dst.0);
        }
    }

    // Walk nodes in priority order: groups are numbered by their earliest member,
    // and each group's members come out already sorted by pos.
    let mut group_of: Vec<Option<usize>> = vec![None; cx.g.nodes.len()];
    let mut index_of_root: BTreeMap<usize, usize> = BTreeMap::new();
    let mut groups: Vec<Vec<NodeId>> = Vec::new();
    for &n in &cx.g.order {
        if !inside(n) {
            continue;
        } // placeholders, outputs, nodes outside `within`
        let i = *index_of_root.entry(uf.find(n.0)).or_insert_with(|| {
            groups.push(Vec::new());
            groups.len() - 1
        });
        group_of[n.0] = Some(i);
        groups[i].push(n);
    }

    Partition { group_of, groups }
}

/// Distinct member lists from `partition(cx, plan, l, None)` that contain at least
/// one node of `touched`.
fn groups_containing(
    cx: &Ctx,
    plan: &ConnectPlan,
    l: Lvl,
    touched: &BTreeSet<NodeId>,
) -> BTreeSet<Vec<NodeId>> {
    let part = partition(cx, plan, l, None);
    let mut out = BTreeSet::new();
    for &n in touched {
        let Some(gi) = part.group_of[n.0] else {
            continue;
        };
        out.insert(part.groups[gi].clone());
    }
    out
}

fn meta(g: &TileGraph, n: NodeId) -> &super::tile_graph::OpMeta {
    g.nodes[n.0]
        .meta
        .as_ref()
        .expect("meta(g, n) requires an Op node with tile metadata")
}

fn meta_opt(g: &TileGraph, n: NodeId) -> Option<&super::tile_graph::OpMeta> {
    g.nodes[n.0].meta.as_ref()
}

fn shape(g: &TileGraph, n: NodeId) -> &[u64] {
    &g.nodes[n.0].shape
}

/// In-edges of nodes in `n_set` whose producer lies outside `n_set` (placeholders included).
fn external_edges(g: &TileGraph, n_set: &[NodeId]) -> Vec<EdgeId> {
    let set: BTreeSet<NodeId> = n_set.iter().copied().collect();
    let mut out = Vec::new();
    for &n in n_set {
        for &eid in &g.nodes[n.0].in_edges {
            let e = &g.edges[eid.0];
            if !set.contains(&e.src) {
                out.push(eid);
            }
        }
    }
    out
}

fn internal_edges(g: &TileGraph, n_set: &[NodeId]) -> Vec<EdgeId> {
    let set: BTreeSet<NodeId> = n_set.iter().copied().collect();
    g.edges
        .iter()
        .filter(|e| set.contains(&e.src) && set.contains(&e.dst))
        .map(|e| e.id)
        .collect()
}

fn key(cx: &Ctx, n_set: &[NodeId]) -> GroupKey {
    let mut nodes: Vec<NodeId> = n_set.to_vec();
    nodes.sort();
    let mut internal_levels: Vec<(EdgeId, Lvl)> = internal_edges(cx.g, n_set)
        .into_iter()
        .map(|e| (e, cx.plan.level[e.0]))
        .collect();
    internal_levels.sort_by_key(|(e, _)| e.0);
    GroupKey {
        nodes,
        internal_levels,
    }
}

fn group_latency(
    cx: &Ctx,
    n_set: &[NodeId],
    cache: &mut HashMap<GroupKey, Lat>,
    budget: &mut usize,
) -> Lat {
    let k = key(cx, n_set);
    if !cache.contains_key(&k) {
        cache.insert(k.clone(), compute_group_latency(cx, n_set, budget));
    }
    cache.get(&k).expect("group_latency cache insert").clone()
}

fn compute_group_latency(cx: &Ctx, n_set: &[NodeId], budget: &mut usize) -> Lat {
    let opaque = |n: NodeId| Lat {
        est: cx.ev.opaque(cx.g, cx.hw, n),
        measured: None,
        choice: Some(KernelChoice::Opaque(n)),
    };
    if n_set.len() == 1 && meta_opt(cx.g, n_set[0]).is_none() {
        return opaque(n_set[0]);
    }
    let mut cands = tile_group(cx, 0, n_set, &[]);
    if cands.is_empty() {
        return if n_set.len() == 1 {
            opaque(n_set[0])
        } else {
            Lat {
                est: f64::INFINITY,
                measured: None,
                choice: None,
            }
        };
    }
    if *budget >= cands.len() {
        for c in cands.iter_mut() {
            c.cost.measured = cx.ev.measure(cx.g, cx.hw, c);
        }
        *budget -= cands.len();
    }
    let all_measured = cands.iter().all(|c| c.cost.measured.is_some());
    let est = cands
        .iter()
        .map(|c| c.cost.est)
        .fold(f64::INFINITY, f64::min);
    let best = cands
        .into_iter()
        .min_by(|a, b| {
            let v = |c: &GroupSchedule| {
                if all_measured {
                    c.cost.measured.unwrap()
                } else {
                    c.cost.est
                }
            };
            v(a).total_cmp(&v(b))
        })
        .expect("non-empty candidates");
    Lat {
        est,
        measured: if all_measured {
            best.cost.measured
        } else {
            None
        },
        choice: Some(KernelChoice::Tiled(best)),
    }
}

/// Sum latencies for `old` under `plan_old` and `new` under `plan_new`. Uses
/// measured costs only when every group on both sides has a measurement; otherwise
/// uses each group's minimum estimate (`Lat.est`).
fn compare_sums(
    cx0: &Ctx,
    old: &BTreeSet<Vec<NodeId>>,
    new: &BTreeSet<Vec<NodeId>>,
    plan_old: &ConnectPlan,
    plan_new: &ConnectPlan,
    cache: &mut HashMap<GroupKey, Lat>,
    budget: &mut usize,
) -> (f64, f64) {
    let cx_old = Ctx {
        g: cx0.g,
        hw: cx0.hw,
        caps: cx0.caps,
        params: cx0.params,
        ev: cx0.ev,
        plan: plan_old,
    };
    let cx_new = Ctx {
        g: cx0.g,
        hw: cx0.hw,
        caps: cx0.caps,
        params: cx0.params,
        ev: cx0.ev,
        plan: plan_new,
    };

    let old_lats: Vec<Lat> = old
        .iter()
        .map(|g| group_latency(&cx_old, g, cache, budget))
        .collect();
    let new_lats: Vec<Lat> = new
        .iter()
        .map(|g| group_latency(&cx_new, g, cache, budget))
        .collect();

    let all_measured = old_lats.iter().all(|l| l.measured.is_some())
        && new_lats.iter().all(|l| l.measured.is_some());

    let sum = |lats: &[Lat]| -> f64 {
        if all_measured {
            lats.iter().map(|l| l.measured.unwrap()).sum()
        } else {
            lats.iter().map(|l| l.est).sum()
        }
    };

    (sum(&old_lats), sum(&new_lats))
}

fn group_min_pos(cx: &Ctx, p: &Partition, gi: usize) -> usize {
    p.groups[gi]
        .first()
        .map(|n| cx.g.pos[n.0])
        .unwrap_or(usize::MAX)
}

/// Kahn topological order over `p.groups`, with arcs from cross-group edges in
/// `cx.g` (valid when `p` was built at level `l` and only edges at level ≤ `l`
/// can cross components). Ties broken by the smallest member `pos`.
fn topo_sort_quotient(cx: &Ctx, p: &Partition) -> Vec<Vec<NodeId>> {
    let n = p.groups.len();
    if n == 0 {
        return vec![];
    }

    let mut arcs = BTreeSet::new();
    for e in &cx.g.edges {
        let (Some(a), Some(b)) = (p.group_of[e.src.0], p.group_of[e.dst.0]) else {
            continue;
        };
        if a != b {
            arcs.insert((a, b));
        }
    }

    let mut adj = vec![Vec::new(); n];
    let mut indegree = vec![0usize; n];
    for (a, b) in arcs {
        adj[a].push(b);
        indegree[b] += 1;
    }

    let mut ready: BTreeSet<(usize, usize)> = BTreeSet::new();
    for gi in 0..n {
        if indegree[gi] == 0 {
            ready.insert((group_min_pos(cx, p, gi), gi));
        }
    }

    let mut order = Vec::with_capacity(n);
    while let Some((_, gi)) = ready.pop_first() {
        order.push(p.groups[gi].clone());
        for &nb in &adj[gi] {
            indegree[nb] -= 1;
            if indegree[nb] == 0 {
                ready.insert((group_min_pos(cx, p, nb), nb));
            }
        }
    }

    if order.len() < n {
        for gi in 0..n {
            if indegree[gi] > 0 {
                order.push(p.groups[gi].clone());
            }
        }
    }

    order
}

/// Directed cycle in the group quotient, if any. `arcs` is a set of group indices
/// `(src, dst)` with `src != dst`. Returns every group index on one cycle.
fn find_cycle(n: usize, arcs: &BTreeSet<(usize, usize)>) -> Option<Vec<usize>> {
    if n == 0 {
        return None;
    }

    let mut adj = vec![Vec::new(); n];
    for &(a, b) in arcs {
        if a < n && b < n {
            adj[a].push(b);
        }
    }

    #[derive(Clone, Copy, PartialEq, Eq)]
    enum Color {
        White,
        Grey,
        Black,
    }

    let mut color = vec![Color::White; n];
    let mut parent = vec![None; n];

    fn dfs(
        u: usize,
        adj: &[Vec<usize>],
        color: &mut [Color],
        parent: &mut [Option<usize>],
    ) -> Option<Vec<usize>> {
        color[u] = Color::Grey;
        for &v in &adj[u] {
            match color[v] {
                Color::White => {
                    parent[v] = Some(u);
                    if let Some(cycle) = dfs(v, adj, color, parent) {
                        return Some(cycle);
                    }
                }
                Color::Grey => {
                    let mut cycle = vec![v];
                    let mut cur = u;
                    while cur != v {
                        cycle.push(cur);
                        cur = parent[cur].expect("back edge implies an ancestor in the DFS tree");
                    }
                    return Some(cycle);
                }
                Color::Black => {}
            }
        }
        color[u] = Color::Black;
        None
    }

    for u in 0..n {
        if color[u] == Color::White
            && let Some(cycle) = dfs(u, &adj, &mut color, &mut parent)
        {
            return Some(cycle);
        }
    }
    None
}

fn validate(cx: &Ctx, plan: &ConnectPlan) -> std::result::Result<(), Violation> {
    for l in 0..cx.hw.top() {
        // l = 0 ..= H-1
        let p = partition(cx, plan, l, None);
        let mut arcs = BTreeSet::new();
        for e in &cx.g.edges {
            let (Some(a), Some(b)) = (p.group_of[e.src.0], p.group_of[e.dst.0]) else {
                continue;
            };
            if plan.level[e.id.0] > l {
                continue;
            } // internal at l by construction
            if a == b {
                return Err(Violation::ShortCircuit {
                    edge: e.id,
                    level: l,
                });
            }
            arcs.insert((a, b));
        }
        if let Some(cycle) = find_cycle(p.groups.len(), &arcs) {
            return Err(Violation::Cycle {
                level: l,
                groups: cycle,
            });
        }
    }
    Ok(())
}

fn raise_with_closure(cx: &Ctx, plan: &ConnectPlan, e: EdgeId, k: Lvl) -> Option<ConnectPlan> {
    let (a, b) = (cx.g.edges[e.0].src, cx.g.edges[e.0].dst);
    // The kernels `e` joins, as they stand before the raise. Closure is measured against these.
    let p0 = partition(cx, plan, 0, None);
    let core: BTreeSet<NodeId> = [a, b]
        .iter()
        .filter_map(|n| p0.group_of[n.0])
        .flat_map(|i| p0.groups[i].clone())
        .collect();
    let mut pulled: BTreeSet<NodeId> = BTreeSet::new(); // nodes closure has added beyond `core`
    let mut p = plan.clone();
    p.level[e.0] = k;
    let bound = cx.hw.top() * cx.g.edges.len(); // every round raises ≥ 1 edge; levels ≤ H
    for _ in 0..=bound {
        match validate(cx, &p) {
            Ok(()) => return Some(p),
            Err(Violation::ShortCircuit { edge: f, level: l }) => {
                if f == e {
                    return None;
                } // the proposed edge itself: k is not realisable
                // another edge between two nodes that now share a level-l group: it must move up too
                let Some(&k2) = legal_levels(cx, &cx.g.edges[f.0])
                    .iter()
                    .find(|&&k2| k2 > l)
                else {
                    return None;
                };
                p.level[f.0] = k2;
            }
            Err(Violation::Cycle { level: l, groups }) => {
                let part = partition(cx, &p, l, None);
                let m: BTreeSet<NodeId> = groups
                    .iter()
                    .flat_map(|&i| part.groups[i].clone())
                    .collect();
                // D = op nodes outside M lying on a path M ⇝ x ⇝ M (plain DAG reachability)
                let d: BTreeSet<NodeId> = op_nodes(cx.g)
                    .filter(|x| {
                        !m.contains(x)
                            && reachable_from_set(cx.g, &m, *x)
                            && reaches_set(cx.g, *x, &m)
                    })
                    .collect();
                // Cap on the nodes pulled in (Welder's dependency cap). NOT |D|: for a single
                // cycle, M already holds every group on it and D is empty.
                pulled.extend(m.iter().chain(&d).copied().filter(|x| !core.contains(x)));
                if pulled.len() > cx.params.max_closure {
                    return None;
                }
                let mut raised = false;
                for f in &cx.g.edges {
                    let both = |n| m.contains(&n) || d.contains(&n);
                    if both(f.src) && both(f.dst) && p.level[f.id.0] <= l {
                        // lowest legal level that is internal at l (teenygrad today: only H)
                        let Some(&k2) = legal_levels(cx, f).iter().find(|&&k2| k2 > l) else {
                            return None;
                        };
                        p.level[f.id.0] = k2;
                        raised = true;
                    }
                }
                if !raised {
                    return None;
                } // defensive: a cycle always contains an edge ≤ l
            }
        }
    }
    None
}

fn schedule(cx0: &Ctx) -> Result<Schedule> {
    let mut plan = ConnectPlan {
        level: vec![0; cx0.g.edges.len()],
    }; // valid (A3)
    let mut cache = HashMap::new();
    let mut budget = cx0.params.max_measurements;
    for _pass in 0..cx0.params.max_passes {
        let mut changed = false;
        for &n in &cx0.g.order {
            for e in out_edges(cx0.g, n) {
                let cx = &Ctx {
                    plan: &plan,
                    ..*cx0
                };
                let ks = legal_levels(cx, e);
                if ks.len() == 1 {
                    continue;
                }
                let cur = plan.level[e.id.0];
                let mut best: Option<(f64, ConnectPlan)> = None;
                for &k in ks.iter().filter(|&&k| k != cur) {
                    let Some(p2) = (if k > cur {
                        raise_with_closure(cx, &plan, e.id, k)
                    } else {
                        let mut p = plan.clone();
                        p.level[e.id.0] = k;
                        validate(cx, &p).ok().map(|_| p)
                    }) else {
                        continue;
                    };
                    let touched: BTreeSet<NodeId> = changed_edges(&plan, &p2)
                        .flat_map(|f| [cx.g.edges[f.0].src, cx.g.edges[f.0].dst])
                        .collect();
                    let old = groups_containing(cx, &plan, 0, &touched); // dedup'd node sets
                    let new = groups_containing(&Ctx { plan: &p2, ..*cx0 }, &p2, 0, &touched);
                    let (lo, ln) =
                        compare_sums(cx0, &old, &new, &plan, &p2, &mut cache, &mut budget);
                    let delta = ln - lo;
                    if delta < -cx.params.min_gain * lo && best.as_ref().is_none_or(|b| delta < b.0)
                    {
                        best = Some((delta, p2));
                    }
                }
                if let Some((_, p2)) = best {
                    plan = p2;
                    changed = true;
                }
            }
        }
        if !changed {
            break;
        }
    }

    let cx = &Ctx {
        plan: &plan,
        ..*cx0
    };

    let kernels = topo_sort_quotient(cx, &partition(cx, &plan, 0, None))
        .into_iter()
        // Some(..) for every final group: singletons are always feasible (opaque fallback), and an
        // infeasible merge has Δ = +∞ so it is never accepted.
        .map(|grp| {
            group_latency(cx, &grp, &mut cache, &mut budget)
                .choice
                .unwrap()
        })
        .collect();

    Ok(Schedule { plan, kernels })
}

#[cfg(test)]
mod tests {
    use super::cost::AnalyticEvaluator;
    use super::testing::{Builder, device_and_shared, flat, softmax};
    use super::*;
    use crate::graph::TritonLowering;
    use crate::graph::tile_resolve::NoConsts;
    use teeny_core::graph::{DtypeRepr, Shape};
    use teeny_core::model::{ExecutableOp, GridSpec, KernelTileSpec, LoweringMode, TensorTileSpec};
    use teeny_test::graph::DecomposedSoftmaxBuilder;

    use super::super::tile_graph::{Node, OpMeta};

    /// teenygrad's codegen as ANALYSIS.md §12 describes it, except that any
    /// op -> op edge may fuse at H.
    struct TritonCaps;
    impl Capabilities for TritonCaps {
        fn fuse_level(&self, _g: &TileGraph, _e: &Edge, _k: Lvl) -> bool {
            true
        }
        fn stage_input(&self, _g: &TileGraph, _e: &Edge, _into: Lvl, _r: &Region) -> bool {
            false
        }
        fn halo_stage(&self) -> bool {
            false
        }
        fn cross_unit_reduce(&self) -> bool {
            false
        }
        fn pow2_blocks(&self) -> bool {
            true
        }
        fn double_buffer(&self) -> bool {
            false
        }
        fn thread_tiling(&self) -> ThreadTiling {
            ThreadTiling::Delegated
        }
        fn knob_style(&self) -> KnobStyle {
            KnobStyle::AnchorConsts
        }
    }

    /// The lowered op's own tile and grid specs, with no consts to go on.
    fn op_meta(_dag_idx: usize, op: Box<dyn ExecutableOp>) -> Option<OpMeta> {
        OpMeta::new(op, Box::new(NoConsts), 1.0)
    }

    /// The level `plan` gives the edge `src -> dst` on `dst`'s operand `port`.
    fn edge_level(g: &TileGraph, plan: &ConnectPlan, src: NodeId, dst: NodeId, port: usize) -> Lvl {
        let e = g.nodes[dst.0]
            .in_edges
            .iter()
            .map(|e| &g.edges[e.0])
            .find(|e| e.dst_port == port)
            .unwrap_or_else(|| panic!("node {} should have an operand {port}", dst.0));
        assert_eq!(
            e.src, src,
            "operand {port} of node {} should be fed by node {}",
            dst.0, src.0
        );
        plan.level[e.id.0]
    }

    #[test]
    fn test_softmax_from_primitives_schedules_as_one_group() {
        // x -> max -> sub(x, .) -> exp -> sum -> div(exp, .)
        // |__________^              |___________^
        //
        // ANALYSIS.md §12.7 D1 through the real `TritonLowering`. `x` has two
        // consumers, and both reduced tensors are broadcast back over the
        // column axis by the row-broadcast Sub/Div kernels. There are no
        // rsteps yet (P6), so `max` and `sum` read whole rows. Checks the
        // parts of D1 visible in a schedule's connect plan, over device
        // memory (level 0) and shared memory (level 1 = H).
        //
        // - (b) a full-row tile is a legal tiling of the whole group, and a
        //   row is small enough to fit in shared memory, so the five compute
        //   nodes should fuse: every edge between them is at level 1, and
        //   they form one tiled kernel.
        // - `x` is a placeholder, produced outside the group, so its two
        //   out edges stay at level 0.
        const ROWS: usize = 64;
        const COLS: usize = 1024;

        let softmax = DecomposedSoftmaxBuilder::default()
            .rows(ROWS)
            .cols(COLS)
            .build()
            .expect("every required field is set")
            .graph();

        let lowering = TritonLowering::default();
        let (dag, graph_to_dag, _) = lowering
            .lower_with_mapping(&softmax.graph, LoweringMode::Inference)
            .expect("lowering should not fail here");
        // `from_dag` numbers tile nodes in topological order, which need not
        // match graph order, so look each one up by its lowered kernel's name.
        // Every op in this graph lowers to a different kernel. `from_dag`
        // consumes the DAG, so read the names first.
        let kernel_names: Vec<String> = graph_to_dag
            .iter()
            .map(|&dag_idx| dag.node(dag_idx).value.name().to_string())
            .collect();
        let g = TileGraph::from_dag(dag, op_meta).expect("softmax shapes are static");

        let tile_node = |graph_idx: usize| {
            let name = kernel_names[graph_idx].as_str();
            let mut ids = g
                .nodes
                .iter()
                .filter(|n| n.kind != NodeKind::Output && n.name == name);
            let id = ids
                .next()
                .unwrap_or_else(|| panic!("graph node {graph_idx} (\"{name}\") has no tile node"))
                .id;
            assert!(ids.next().is_none(), "kernel name \"{name}\" is not unique");
            id
        };
        let [x, max, sub, exp, sum, div] = [
            softmax.input,
            softmax.max,
            softmax.sub,
            softmax.exp,
            softmax.sum,
            softmax.div,
        ]
        .map(tile_node);

        let hw = device_and_shared();
        let params = SearchParams {
            top_k: 10,
            max_visited: 2000,
            fanouts_per_tile: 1,
            max_closure: 10,
            max_recompute: 1.0,
            min_gain: 0.0,
            max_passes: 1,
            max_measurements: 0,
        };
        let start = ConnectPlan {
            level: vec![0; g.edges.len()],
        };
        let cx = Ctx {
            g: &g,
            hw: &hw,
            caps: &TritonCaps,
            params: &params,
            ev: &AnalyticEvaluator,
            plan: &start,
        };
        let Ok(schedule) = schedule(&cx) else {
            panic!("scheduling softmax should not fail here");
        };

        for (src, dst, port) in [(x, max, 0), (x, sub, 0)] {
            assert_eq!(
                edge_level(&g, &schedule.plan, src, dst, port),
                0,
                "x is produced outside the group, so edge {} -> {} should stay at level 0",
                src.0,
                dst.0
            );
        }

        for (name, src, dst, port) in [
            ("max -> sub", max, sub, 1),
            ("sub -> exp", sub, exp, 0),
            ("exp -> sum", exp, sum, 0),
            ("exp -> div", exp, div, 0),
            ("sum -> div", sum, div, 1),
        ] {
            assert_eq!(
                edge_level(&g, &schedule.plan, src, dst, port),
                1,
                "{name} should be fused at shared memory (level 1)"
            );
        }

        let [KernelChoice::Tiled(group)] = schedule.kernels.as_slice() else {
            panic!("the five compute nodes should be one tiled kernel");
        };
        let mut nodes = group.nodes.clone();
        nodes.sort();
        let mut expected = vec![max, sub, exp, sum, div];
        expected.sort();
        assert_eq!(nodes, expected);
    }

    // ─────────────── Connection plan: partition, validate, closure ───────────────
    //
    // ANALYSIS.md §12.7 C1-C12, over device and shared memory (H = 1). Not
    // here:
    // - C6 and C7 need a level strictly between 0 and H, and there is none.
    // - C12 needs `tile_group` to return candidates to measure.

    /// A rank-1 operand, as every [`FakeOp`] port declares.
    const OPERAND: TensorTileSpec = TensorTileSpec {
        param: "x",
        rank: 1,
        axes: &[],
        reduction_axis: None,
        untiled_dims: &[],
    };

    /// Up to three rank-1 operands, so every edge into a [`FakeOp`] passes P2.
    const FAKE_SPEC: KernelTileSpec = KernelTileSpec {
        inputs: &[OPERAND, OPERAND, OPERAND],
        outputs: &[OPERAND],
        loop_spec: None,
    };

    /// Just enough of an [`ExecutableOp`] for [`OpMeta::new`]: a rank-1
    /// output, and a tile and grid spec unless it is opaque.
    struct FakeOp {
        shape: Shape,
        tileable: bool,
    }

    impl ExecutableOp for FakeOp {
        fn name(&self) -> &str {
            "fake"
        }
        fn forward_kernel_source(&self) -> &str {
            ""
        }
        fn forward_kernel_entry_point(&self) -> &str {
            ""
        }
        fn output_shape(&self) -> &Shape {
            &self.shape
        }
        fn output_dtype(&self) -> DtypeRepr {
            DtypeRepr::F32
        }
        fn tile_spec(&self) -> Option<KernelTileSpec> {
            self.tileable.then_some(FAKE_SPEC)
        }
        fn grid_spec(&self) -> Option<GridSpec> {
            self.tileable.then_some(GridSpec { axes: &[] })
        }
        fn as_any(&self) -> &dyn std::any::Any {
            self
        }
    }

    /// `n` op nodes, numbered in topological order (which `order` then is),
    /// joined by `edges` as `(src, dst, dst_port)`. Edge `i` gets `EdgeId(i)`.
    /// The nodes in `opaque` have no meta.
    fn op_graph(n: usize, opaque: &[usize], edges: &[(usize, usize, usize)]) -> TileGraph {
        let mut g = TileGraph {
            nodes: Vec::with_capacity(n),
            edges: Vec::with_capacity(edges.len()),
            order: (0..n).map(NodeId).collect(),
            pos: (0..n).collect(),
        };
        for i in 0..n {
            let op = FakeOp {
                shape: vec![Some(8)],
                tileable: !opaque.contains(&i),
            };
            g.nodes.push(Node {
                id: NodeId(i),
                kind: NodeKind::Op,
                name: format!("n{i}"),
                in_edges: vec![],
                out_edges: vec![],
                shape: vec![8],
                dtype: DtypeRepr::F32,
                meta: OpMeta::new(Box::new(op), Box::new(NoConsts), 1.0),
            });
        }
        for (i, &(src, dst, dst_port)) in edges.iter().enumerate() {
            g.edges.push(Edge {
                id: EdgeId(i),
                src: NodeId(src),
                dst: NodeId(dst),
                dst_port,
            });
            g.nodes[src].out_edges.push(EdgeId(i));
            g.nodes[dst].in_edges.push(EdgeId(i));
        }
        g
    }

    fn search_params(max_closure: usize) -> SearchParams {
        SearchParams {
            top_k: 10,
            max_visited: 2000,
            fanouts_per_tile: 1,
            max_closure,
            max_recompute: 1.0,
            min_gain: 0.0,
            max_passes: 1,
            max_measurements: 0,
        }
    }

    /// Runs `f` with a context over `g` and `plan`, on device and shared memory.
    fn with_ctx<R>(
        g: &TileGraph,
        max_closure: usize,
        plan: &ConnectPlan,
        f: impl FnOnce(&Ctx) -> R,
    ) -> R {
        let hw = device_and_shared();
        let params = search_params(max_closure);
        f(&Ctx {
            g,
            hw: &hw,
            caps: &TritonCaps,
            params: &params,
            ev: &AnalyticEvaluator,
            plan,
        })
    }

    /// Raises edge `e` of `plan` to level 1 (H) with A4's closure. On success,
    /// also checks the properties A4 promises: the plan validates, and every
    /// edge sits at one of its legal levels.
    fn raise(g: &TileGraph, max_closure: usize, plan: &[Lvl], e: usize) -> Option<Vec<Lvl>> {
        let plan = ConnectPlan {
            level: plan.to_vec(),
        };
        with_ctx(g, max_closure, &plan, |cx| {
            let raised = raise_with_closure(cx, &plan, EdgeId(e), 1)?;
            assert!(
                validate(cx, &raised).is_ok(),
                "A4 returned a plan that fails validation: {:?}",
                raised.level
            );
            for edge in &g.edges {
                assert!(
                    legal_levels(cx, edge).contains(&raised.level[edge.id.0]),
                    "A4 put edge {} at level {}, which is not legal for it",
                    edge.id.0,
                    raised.level[edge.id.0]
                );
            }
            Some(raised.level)
        })
    }

    /// The level-0 groups of `plan` over `g`.
    fn groups(g: &TileGraph, plan: &[Lvl]) -> Vec<Vec<NodeId>> {
        let plan = ConnectPlan {
            level: plan.to_vec(),
        };
        with_ctx(g, 10, &plan, |cx| partition(cx, &plan, 0, None).groups)
    }

    fn nodes(ids: &[usize]) -> Vec<NodeId> {
        ids.iter().copied().map(NodeId).collect()
    }

    #[test]
    fn test_c1_raising_one_of_two_parallel_edges_raises_both() {
        // P -> mul(x, x): both ports of node 1 read node 0. Raising port 0
        // short-circuits port 1, which closure then raises too.
        let g = op_graph(2, &[], &[(0, 1, 0), (0, 1, 1)]);

        assert_eq!(raise(&g, 10, &[0, 0], 0), Some(vec![1, 1]));
    }

    #[test]
    fn test_c2_raising_a_diamond_edge_pulls_in_the_bypass() {
        // A -> B, A -> C, C -> B. Raising A -> B alone leaves {A, B} and {C}
        // in a cycle, so closure pulls C in and raises both of its edges.
        let g = op_graph(3, &[], &[(0, 1, 0), (0, 2, 0), (2, 1, 1)]);

        let raised = raise(&g, 10, &[0, 0, 0], 0).expect("C is one node, within the cap");
        assert_eq!(raised, vec![1, 1, 1]);
        assert_eq!(groups(&g, &raised), vec![nodes(&[0, 1, 2])]);
    }

    #[test]
    fn test_c3_a_diamond_joining_an_existing_group_closes_the_short_circuit() {
        // A -> B (edge 0), A -> C (edge 1), C -> B (edge 2), with one side of
        // the diamond already fused. Raising another edge puts all three
        // nodes in one group with the last edge still at 0, a short-circuit
        // (not a cycle) that closure must raise. Checked from both sides.
        let g = op_graph(3, &[], &[(0, 1, 0), (0, 2, 0), (2, 1, 1)]);

        // {B, C} fused first, then A -> B proposed: A -> C short-circuits.
        assert_eq!(raise(&g, 10, &[0, 0, 1], 0), Some(vec![1, 1, 1]));
        // {A, C} fused first, then A -> B proposed: C -> B short-circuits.
        assert_eq!(raise(&g, 10, &[0, 1, 0], 0), Some(vec![1, 1, 1]));
    }

    #[test]
    fn test_c4_a_long_residual_fuses_only_when_the_cap_covers_the_pulled_nodes() {
        // A -> B -> C -> D -> E plus A -> E (edge 4). Raising A -> E makes one
        // cycle through every group, so D = ∅ and only the cap on `pulled`
        // ({B, C, D}) can stop it. A cap on |D| would fuse even at 0.
        let g = op_graph(
            5,
            &[],
            &[(0, 1, 0), (1, 2, 0), (2, 3, 0), (3, 4, 0), (0, 4, 1)],
        );

        assert_eq!(raise(&g, 3, &[0; 5], 4), Some(vec![1; 5]));
        assert_eq!(raise(&g, 2, &[0; 5], 4), None);
    }

    #[test]
    fn test_c5_an_opaque_node_on_the_bypass_blocks_the_raise() {
        // A -> X -> B plus A -> B (edge 2), with X opaque. Closure has to pull
        // X in, but X's edges only allow level 0, so A -> B never fuses.
        let g = op_graph(3, &[1], &[(0, 1, 0), (1, 2, 0), (0, 2, 1)]);

        assert_eq!(raise(&g, 10, &[0; 3], 2), None);
    }

    #[test]
    fn test_c8_lowering_an_edge_inside_a_group_is_a_short_circuit() {
        // A -> B (edge 0) lowered to 0 while A -> C -> B stays at H. A and B
        // still share a group through C, so the lowered edge short-circuits
        // and A15 rejects the proposal, leaving the plan as it was.
        let g = op_graph(3, &[], &[(0, 1, 0), (0, 2, 0), (2, 1, 1)]);
        let lowered = ConnectPlan {
            level: vec![0, 1, 1],
        };

        let verdict = with_ctx(&g, 10, &lowered, |cx| validate(cx, &lowered));
        assert!(
            matches!(
                verdict,
                Err(Violation::ShortCircuit {
                    edge: EdgeId(0),
                    level: 0
                })
            ),
            "lowering A -> B inside {{A, B, C}} should short-circuit at level 0"
        );
    }

    #[test]
    fn test_c9_a_three_group_cycle_fuses_only_when_the_cap_covers_both_middles() {
        // A -> B -> C -> D plus A -> D (edge 3). Raising A -> D makes the
        // groups {A, D}, {B} and {C} a cycle: M is all four nodes, D = ∅, and
        // pulled = {B, C}.
        let g = op_graph(4, &[], &[(0, 1, 0), (1, 2, 0), (2, 3, 0), (0, 3, 1)]);

        let raised = raise(&g, 2, &[0; 4], 3).expect("pulled = {B, C} is within a cap of 2");
        assert_eq!(groups(&g, &raised), vec![nodes(&[0, 1, 2, 3])]);
        assert_eq!(raise(&g, 1, &[0; 4], 3), None);
    }

    #[test]
    fn test_c10_degenerate_graphs_schedule_in_a_fixed_order() {
        // Two disconnected chains, a graph of opaque nodes only, and a single
        // node. Every node should land in exactly one kernel, no kernel should
        // join the two chains, opaque nodes should stay opaque, and the
        // schedule should come out the same way every time.
        let cases = [
            (
                "disconnected",
                op_graph(4, &[], &[(0, 1, 0), (2, 3, 0)]),
                vec![nodes(&[0, 1]), nodes(&[2, 3])],
            ),
            (
                "all opaque",
                op_graph(3, &[0, 1, 2], &[(0, 1, 0), (1, 2, 0)]),
                vec![nodes(&[0]), nodes(&[1]), nodes(&[2])],
            ),
            ("single node", op_graph(1, &[], &[]), vec![nodes(&[0])]),
        ];

        for (name, g, expected) in &cases {
            let run = || {
                let start = ConnectPlan {
                    level: vec![0; g.edges.len()],
                };
                with_ctx(g, 10, &start, |cx| match schedule(cx) {
                    Ok(s) => s,
                    Err(e) => panic!("{name}: scheduling should not fail: {e}"),
                })
            };
            let kernels = |s: &Schedule| -> Vec<Vec<NodeId>> {
                s.kernels
                    .iter()
                    .map(|k| match k {
                        KernelChoice::Opaque(n) => vec![*n],
                        KernelChoice::Tiled(t) => t.nodes.clone(),
                    })
                    .collect()
            };
            let first = run();
            assert_eq!(&kernels(&first), expected, "{name}: kernels, in order");
            for k in &first.kernels {
                if let KernelChoice::Tiled(t) = k {
                    assert!(
                        t.nodes.iter().all(|n| g.nodes[n.0].meta.is_some()),
                        "{name}: an opaque node was tiled"
                    );
                }
            }

            let again = run();
            assert_eq!(again.plan.level, first.plan.level, "{name}: same plan");
            assert_eq!(kernels(&again), kernels(&first), "{name}: same kernels");
        }
    }

    #[test]
    fn test_c11_parallel_bypasses_fuse_whichever_cycle_is_reported_first() {
        // A -> B plus the bypasses A -> C -> B and A -> D -> B. Raising A -> B
        // makes two cycles; `find_cycle` returns one, and the other branch has
        // to arrive through D. Listing the bypasses in either order changes
        // which cycle `find_cycle` sees first, but not the result.
        let c_first = op_graph(
            4,
            &[],
            &[(0, 1, 0), (0, 2, 0), (2, 1, 1), (0, 3, 0), (3, 1, 2)],
        );
        let d_first = op_graph(
            4,
            &[],
            &[(0, 1, 0), (0, 3, 0), (3, 1, 2), (0, 2, 0), (2, 1, 1)],
        );

        for (name, g) in [("C first", &c_first), ("D first", &d_first)] {
            let raised = raise(g, 2, &[0; 5], 0)
                .unwrap_or_else(|| panic!("{name}: pulled = {{C, D}} is within a cap of 2"));
            assert_eq!(raised, vec![1; 5], "{name}");
            assert_eq!(groups(g, &raised), vec![nodes(&[0, 1, 2, 3])], "{name}");
            assert_eq!(
                raise(g, 1, &[0; 5], 0),
                None,
                "{name}: a cap of 1 is too small"
            );
        }
    }

    // ─────────────────────── Tiling: §12.7 D1, D2, S5 and W7 ───────────────────────

    /// The schedule of `g` with `caps`, over device and shared memory.
    fn schedule_with(g: &TileGraph, caps: &dyn Capabilities) -> Schedule {
        let hw = device_and_shared();
        let params = search_params(10);
        let start = ConnectPlan {
            level: vec![0; g.edges.len()],
        };
        let cx = Ctx {
            g,
            hw: &hw,
            caps,
            params: &params,
            ev: &AnalyticEvaluator,
            plan: &start,
        };
        match schedule(&cx) {
            Ok(s) => s,
            Err(e) => panic!("scheduling should not fail here: {e}"),
        }
    }

    fn sorted(mut v: Vec<NodeId>) -> Vec<NodeId> {
        v.sort();
        v
    }

    #[test]
    fn test_d1_hand_specced_softmax_schedules_as_one_group() {
        // ANALYSIS.md §12.7 D1, with the lowering's specs written out (see
        // `testing::softmax`), so the anchor and the tiling can be checked
        // too. Each greedy merge on the way
        // ({max, sub}, then exp, then sum, then div by closure) must itself be
        // legal and cheaper, or the last one is never reached.
        let s = softmax(64, 1024);
        let g = &s.g;

        let schedule = schedule_with(g, &TritonCaps);

        for (src, dst, port) in [(s.x, s.max, 0), (s.x, s.sub, 0)] {
            assert_eq!(edge_level(g, &schedule.plan, src, dst, port), 0);
        }
        for (name, src, dst, port) in [
            ("max -> sub", s.max, s.sub, 1),
            ("sub -> exp", s.sub, s.exp, 0),
            ("exp -> sum", s.exp, s.sum, 0),
            ("exp -> div", s.exp, s.div, 0),
            ("sum -> div", s.sum, s.div, 1),
        ] {
            assert_eq!(
                edge_level(g, &schedule.plan, src, dst, port),
                1,
                "{name} should be fused at shared memory (level 1)"
            );
        }
        let [KernelChoice::Tiled(group)] = schedule.kernels.as_slice() else {
            panic!("the five compute nodes should be one tiled kernel");
        };
        assert_eq!(
            sorted(group.nodes.clone()),
            sorted(vec![s.max, s.sub, s.exp, s.sum, s.div])
        );
        assert_eq!(group.anchor, s.div);
    }

    #[test]
    fn test_teenygrad_caps_fuse_only_into_riders() {
        // Of softmax's internal edges only sub -> exp ends at a rider, so
        // under teenygrad's own capabilities that is the one fusion.
        let s = softmax(64, 1024);

        let schedule = schedule_with(&s.g, &TeenygradCaps);

        let kernels: Vec<Vec<NodeId>> = schedule
            .kernels
            .iter()
            .map(|k| match k {
                KernelChoice::Tiled(t) => t.nodes.clone(),
                KernelChoice::Opaque(n) => vec![*n],
            })
            .collect();
        assert_eq!(
            kernels,
            vec![vec![s.max], vec![s.sub, s.exp], vec![s.sum], vec![s.div]]
        );
    }

    #[test]
    fn test_d2_layernorm_keeps_its_statistics_as_outputs() {
        // x -> mean -> d = sub(x, mean) -> sq = mul(d, d) -> var -> y = div(d, var),
        // with mean and var also model outputs. Neither is a pure sink (each
        // has a consumer in the group), so there is no sink-shape conflict,
        // and a whole-row tile writes each of them in full.
        use super::testing::{row_broadcast, row_reduce, row_tile};
        let (rows, cols) = (64, 1024);
        let mut b = Builder::default();
        let x = b.input("x", &[rows, cols]);
        let mean = b.op("mean", &[rows, 1], row_reduce(), &[x]);
        let d = b.op("d", &[rows, cols], row_broadcast(), &[x, mean]);
        let sq_spec = super::testing::kernel(&[row_tile(), row_tile()], row_tile());
        let sq = b.op("sq", &[rows, cols], sq_spec, &[d, d]);
        let var = b.op("var", &[rows, 1], row_reduce(), &[sq]);
        let y = b.op("y", &[rows, cols], row_broadcast(), &[d, var]);
        b.output(mean);
        b.output(var);
        let g = b.build();

        let schedule = schedule_with(&g, &TritonCaps);

        let [KernelChoice::Tiled(group)] = schedule.kernels.as_slice() else {
            panic!(
                "layernorm should be one tiled kernel, got {} kernels",
                schedule.kernels.len()
            );
        };
        assert_eq!(
            sorted(group.nodes.clone()),
            sorted(vec![mean, d, sq, var, y])
        );
        for n in [mean, var] {
            let stored = g.nodes[n.0]
                .out_edges
                .iter()
                .any(|e| g.nodes[g.edges[e.0].dst.0].kind == NodeKind::Output);
            assert!(stored, "{} is a model output", g.nodes[n.0].name);
        }
    }

    #[test]
    fn test_s5_a_node_too_big_to_tile_runs_as_an_opaque_kernel() {
        use super::testing::{bind, kernel, tensor};
        let mut b = Builder::default();
        let x = b.input("x", &[4, 1 << 20]);
        let spec = kernel(
            &[tensor(
                2,
                &[bind(&[0], "1", "R"), bind(&[1], "BLOCK_C", "C")],
            )],
            tensor(2, &[bind(&[0], "1", "R")]),
        );
        let r = b.op("r", &[4, 1], spec, &[x]);
        let e = b.op("e", &[4, 1], flat(2, 1), &[r]);
        let g = b.build();

        let schedule = schedule_with(&g, &TritonCaps);

        assert!(schedule.plan.level.iter().all(|&l| l == 0));
        assert!(matches!(
            schedule.kernels.as_slice(),
            [KernelChoice::Opaque(a), KernelChoice::Tiled(t)] if *a == r && t.nodes == [e]
        ));
    }

    #[test]
    fn test_w7_conv_batchnorm_silu_fuses_batchnorm_and_silu_through_the_real_lowering() {
        // §12.5 W7 through `TritonLowering`. conv2d tiles on its own: the
        // edge into batchnorm2d does not fuse, since `TeenygradCaps` fuses
        // only into a rider. batchnorm2d tiles its flattened H·W axis by
        // BLOCK_HW, and silu is a rider it splices after itself.
        //
        // batchnorm2d's spec lists B and C as untiled dims rather than
        // binding them (P4 unmet), so they propagate as `Full`: each unit's
        // region spans every (b, c), though the kernel's own grid gives each
        // CTA one. Traffic is unaffected, but the footprint is B·C times too
        // big, and the group's grid is not the kernel's.
        let block = teeny_test::graph::Conv2dBnSiluBuilder::default()
            .batch(2)
            .in_channels(3)
            .out_channels(8)
            .height(32)
            .width(32)
            .build()
            .expect("every required field is set")
            .graph()
            .expect("graph construction should not fail here");
        let (dag, graph_to_dag, _) = TritonLowering::default()
            .lower_with_mapping(&block.graph, LoweringMode::Inference)
            .expect("lowering should not fail here");
        let names: Vec<String> = graph_to_dag
            .iter()
            .map(|&i| dag.node(i).value.name().to_string())
            .collect();
        let g = TileGraph::from_dag(dag, op_meta).expect("conv block shapes are static");
        let node = |graph_idx: usize| {
            g.nodes
                .iter()
                .find(|n| n.kind == NodeKind::Op && n.name == names[graph_idx])
                .unwrap_or_else(|| panic!("no tile node for \"{}\"", names[graph_idx]))
                .id
        };
        let (conv, bn, silu) = (node(block.conv), node(block.bn), node(block.silu));
        assert!(
            g.nodes[conv.0].meta.is_some(),
            "conv2d lowers with both specs"
        );

        let schedule = schedule_with(&g, &TeenygradCaps);

        let [KernelChoice::Tiled(first), KernelChoice::Tiled(fused)] = schedule.kernels.as_slice()
        else {
            panic!("expected conv2d alone, then one tiled kernel");
        };
        assert_eq!(
            (first.nodes.as_slice(), first.anchor),
            ([conv].as_slice(), conv)
        );
        assert_eq!(fused.nodes, vec![bn, silu]);
        assert_eq!(fused.anchor, bn);
        let [(KnobName::Block("BLOCK_HW"), v)] = fused.knobs.as_slice() else {
            panic!("batchnorm2d's one knob is BLOCK_HW, got {:?}", fused.knobs);
        };
        assert!(
            *v >= 1024 || 1024 % *v == 0 || (*v % 32 == 0 && 32 % (*v / 32) == 0),
            "BLOCK_HW = {v} should box H·W = 32·32"
        );
    }

    #[test]
    #[ignore = "conv2d cannot fuse into batchnorm2d under TeenygradCaps, and the fused tiling \
                is not realisable until P3, P4 and P6 are met (teenygrad-1l7i)"]
    fn test_conv2d_batchnorm_silu_schedules_as_one_tiled_kernel() {
        // input -> conv2d -> batchnorm2d -> silu through `TritonLowering`,
        // scheduled with the production capabilities. The target is the
        // hand-fused `Conv2dBnSiluForward`: conv2d, batchnorm2d and silu in
        // one tiled kernel anchored at conv2d, the only node whose spec ties
        // the group to conv2d's own grid; the input stays apart.
        let block = teeny_test::graph::Conv2dBnSiluBuilder::default()
            .batch(2)
            .in_channels(3)
            .out_channels(8)
            .height(32)
            .width(32)
            .build()
            .expect("every required field is set")
            .graph()
            .expect("graph construction should not fail here");
        let (dag, graph_to_dag, _) = TritonLowering::default()
            .lower_with_mapping(&block.graph, LoweringMode::Inference)
            .expect("lowering should not fail here");
        let names: Vec<String> = graph_to_dag
            .iter()
            .map(|&i| dag.node(i).value.name().to_string())
            .collect();
        let g = TileGraph::from_dag(dag, op_meta).expect("conv block shapes are static");
        let node = |graph_idx: usize| {
            g.nodes
                .iter()
                .find(|n| n.kind == NodeKind::Op && n.name == names[graph_idx])
                .unwrap_or_else(|| panic!("no tile node for \"{}\"", names[graph_idx]))
                .id
        };
        let (conv, bn, silu) = (node(block.conv), node(block.bn), node(block.silu));

        let schedule = schedule_with(&g, &TeenygradCaps);

        let [KernelChoice::Tiled(fused)] = schedule.kernels.as_slice() else {
            panic!(
                "expected one tiled kernel for conv2d, batchnorm2d and silu, got {:?}",
                schedule
                    .kernels
                    .iter()
                    .map(|k| match k {
                        KernelChoice::Opaque(n) => format!("Opaque({})", g.nodes[n.0].name),
                        KernelChoice::Tiled(t) => format!(
                            "Tiled({:?}, anchor {})",
                            t.nodes
                                .iter()
                                .map(|n| &g.nodes[n.0].name)
                                .collect::<Vec<_>>(),
                            g.nodes[t.anchor.0].name
                        ),
                    })
                    .collect::<Vec<_>>()
            );
        };
        assert_eq!(fused.nodes, vec![conv, bn, silu]);
        assert_eq!(fused.anchor, conv);
    }
}
