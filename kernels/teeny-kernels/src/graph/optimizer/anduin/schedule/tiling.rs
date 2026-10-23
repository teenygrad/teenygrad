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

//! Sub-graph tiling (A5, A12, A16): pick the anchor, then search the anchor's
//! knobs best-first for the tilings of a group with the lowest key.
//!
//! Only the tiling that stages into the compute level is implemented
//! (`l + 1 == H`, always so over device and shared memory). A deeper hierarchy
//! needs A13's children, so [`tile_group`] reports no candidates there, and
//! every multi-node group on it stays infeasible.
//!
//! There is no rstep metadata yet (P6), so [`initial_rsteps`] assigns none and
//! [`expand_rsteps`] never grows one: every reduce dim is read whole.

use std::cmp::{Ordering, Reverse};
use std::collections::{BTreeMap, BinaryHeap, HashMap};

use super::cost::{finish_cost, footprint, occupancy_and_key};
use super::legality::{covers, redundancy};
use super::propagate::propagate_group;
use super::seed::{Knob, knobs_for, seed, zip_knobs};
use super::span::{RStep, Region, lens_traffic, sub_lens, walks};
use super::{Cost, Ctx, GroupSchedule, Lvl, Reject, external_edges, meta, out_edges, shape};
use crate::graph::optimizer::anduin::tile_graph::{EdgeId, NodeId};

/// A group's extents at one tiling level (A12-aux `extents_at`), plus the
/// repeat counts a child tiling inherits from walked dims above it.
pub(super) struct Extents {
    pub node: BTreeMap<NodeId, Vec<u64>>, // each node's extents at this level
    pub operand: BTreeMap<EdgeId, Vec<u64>>, // each in-edge's operand extents at this level
    pub trips: BTreeMap<NodeId, u64>,     // missing ⇒ 1
    pub op_trips: BTreeMap<EdgeId, u64>,  // missing ⇒ 1
}

/// The group's anchor and its pure sinks (the nodes with no consumer in `N`).
/// At `l + 1 == H` the root stage is all of `N`. When units are blocks and
/// knobs are the anchor's consts, the anchor is the program the riders are
/// spliced into. Every node after the latest non-rider is a rider, and a rider
/// runs on the anchor's decoded coordinates, so it must have the anchor's
/// output shape: softmax's `div` (`[R, C]`) cannot ride on `sum` (`[R, 1]`).
/// The anchor is therefore the earliest node, from the latest non-rider on,
/// whose shape every later node of the stage shares: the node that first
/// produces the stage's final shape. Its own grid is then the most structured
/// one available (a row-broadcast `sub` rather than the flat `exp` after it).
/// Otherwise the anchor is the last sink.
pub(super) fn choose_anchor(cx: &Ctx, l: Lvl, n_set: &[NodeId]) -> (NodeId, Vec<NodeId>) {
    let sinks: Vec<NodeId> = n_set
        .iter()
        .copied()
        .filter(|&n| out_edges(cx.g, n).all(|e| !n_set.contains(&e.dst)))
        .collect();
    let last = *sinks
        .iter()
        .max_by_key(|s| cx.g.pos[s.0])
        .expect("a finite DAG has a sink");
    let root_stage: Vec<NodeId> = if l + 1 < cx.hw.top() {
        let p = super::partition(cx, cx.plan, l + 1, Some(n_set));
        let gi = p.group_of[last.0].expect("`last` is an op node of N");
        p.groups[gi].clone()
    } else {
        n_set.to_vec()
    };
    let anchor = match (cx.hw.owner_of_units(l), super::seed::style(cx, l)) {
        (super::Owner::Block, super::seed::KnobStyle::AnchorConsts) => {
            let mut stage = root_stage.clone();
            stage.sort_by_key(|n| cx.g.pos[n.0]);
            // With no non-rider, every node is a candidate.
            let from = stage
                .iter()
                .rposition(|&n| !meta(cx.g, n).rider())
                .unwrap_or(0);
            let shared = |i: usize| {
                stage[i + 1..]
                    .iter()
                    .all(|&n| shape(cx.g, n) == shape(cx.g, stage[i]))
            };
            let at = (from..stage.len())
                .find(|&i| shared(i))
                .expect("nothing follows the last node");
            stage[at]
        }
        _ => last,
    };
    (anchor, sinks)
}

/// A12-aux: every extent the tiling of `N` sees. At the root (no parents) it
/// is each tensor's full static shape and every trip count is 1; under a
/// parent it is one parent unit's region, with the walks above counted.
fn extents_at(cx: &Ctx, n_set: &[NodeId], parents: &[&GroupSchedule]) -> Extents {
    let mut ext = Extents {
        node: BTreeMap::new(),
        operand: BTreeMap::new(),
        trips: BTreeMap::new(),
        op_trips: BTreeMap::new(),
    };
    let Some(p) = parents.last() else {
        for &n in n_set {
            ext.node.insert(n, shape(cx.g, n).to_vec());
            for &e in &cx.g.nodes[n.0].in_edges {
                ext.operand
                    .insert(e, shape(cx.g, cx.g.edges[e.0].src).to_vec());
            }
        }
        return ext;
    };
    for &n in n_set {
        ext.node.insert(n, sub_lens(&p.region[&n], &p.extent[&n]));
        let t = p.trips.get(&n).copied().unwrap_or(1);
        ext.trips.insert(n, t * walks(&p.region[&n], &p.extent[&n]));
    }
    for &n in n_set {
        for &e in &cx.g.nodes[n.0].in_edges {
            let src = cx.g.edges[e.0].src;
            let operand = if n_set.contains(&src) {
                ext.node[&src].clone()
            } else {
                sub_lens(&p.in_region[&e], &p.operand_extent[&e])
            };
            ext.operand.insert(e, operand);
            let t = p.op_trips.get(&e).copied().unwrap_or(1);
            ext.op_trips
                .insert(e, t * walks(&p.in_region[&e], &p.operand_extent[&e]));
        }
    }
    ext
}

/// Welder `_assign_reduce_step`. No spec declares a searchable rstep yet (P6),
/// so no reduce dim is chunked.
fn initial_rsteps(
    _cx: &Ctx,
    _n_set: &[NodeId],
    _ext: &Extents,
    _parents: &[&GroupSchedule],
) -> BTreeMap<NodeId, RStep> {
    BTreeMap::new()
}

/// Welder `_expand_reduce_axis`. A no-op until P6: there is no rstep to grow.
fn expand_rsteps(_cx: &Ctx, _s: &mut GroupSchedule, _parents: &[&GroupSchedule]) {}

/// A heap entry ordered by `(key, θ)` ascending; wrapped in `Reverse` for
/// `BinaryHeap`'s max-heap.
struct MinKey(f64, Vec<u64>);

impl PartialEq for MinKey {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}
impl Eq for MinKey {}
impl PartialOrd for MinKey {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for MinKey {
    fn cmp(&self, other: &Self) -> Ordering {
        self.0
            .total_cmp(&other.0)
            .then_with(|| self.1.cmp(&other.1))
    }
}

/// The group's fixed inputs to every θ evaluated by [`tile_group`].
struct Search<'a> {
    l: Lvl,
    n_set: &'a [NodeId],
    anchor: NodeId,
    sinks: Vec<NodeId>,
    knobs: Vec<Knob>,
    rstep: BTreeMap<NodeId, RStep>,
    ext: Extents,
    parents: &'a [&'a GroupSchedule],
}

/// Work per output item: seed and propagate only, no legality checks. The
/// elements every node computes per unit over the elements the anchor's
/// seed produces; ∞ if θ doesn't propagate.
fn wpi(cx: &Ctx, sr: &Search, theta: &[u64]) -> f64 {
    let Ok((_, seed_r)) = seed(
        cx,
        sr.l,
        sr.anchor,
        &sr.ext.node[&sr.anchor],
        &sr.knobs,
        theta,
    ) else {
        return f64::INFINITY;
    };
    let Ok(pr) = propagate_group(
        cx,
        sr.n_set,
        sr.anchor,
        &sr.sinks,
        &seed_r,
        &sr.ext.node,
        &sr.rstep,
    ) else {
        return f64::INFINITY;
    };
    let per = |r: &Region, e: &[u64]| lens_traffic(r, e).iter().product::<u64>() as f64;
    let work: f64 = sr
        .n_set
        .iter()
        .map(|n| per(&pr.region[n], &sr.ext.node[n]))
        .sum();
    work / per(&seed_r, &sr.ext.node[&sr.anchor]).max(1.0)
}

/// The tiling of `N` at level `l` for knob values θ, with its footprint,
/// occupancy and key, or why it is illegal.
fn eval_tiling(cx: &Ctx, sr: &Search, theta: &[u64]) -> Result<GroupSchedule, Reject> {
    let (l, n_set) = (sr.l, sr.n_set);
    let (grid, seed_r) = seed(cx, l, sr.anchor, &sr.ext.node[&sr.anchor], &sr.knobs, theta)?;
    let pr = propagate_group(
        cx,
        n_set,
        sr.anchor,
        &sr.sinks,
        &seed_r,
        &sr.ext.node,
        &sr.rstep,
    )?;
    for &n in n_set {
        let (r, e) = (&pr.region[&n], &sr.ext.node[&n]);
        if redundancy(r, e, &grid) > cx.params.max_recompute + 1e-9 {
            return Err(Reject::Recompute(n));
        }
        if out_edges(cx.g, n).any(|x| !n_set.contains(&x.dst)) && !covers(r, e, &grid) {
            return Err(Reject::Coverage(n));
        }
        // A9's `realisable` applies to stage anchors other than the group
        // anchor, and at l + 1 == H the group is one stage: see `legality`.
    }
    let staged = external_edges(cx.g, n_set)
        .into_iter()
        .filter(|&e| {
            l + 1 == cx.hw.top()
                || cx
                    .caps
                    .stage_input(cx.g, &cx.g.edges[e.0], l + 1, &pr.in_region[&e])
        })
        .collect();
    let mut s = GroupSchedule {
        level: l,
        nodes: n_set.to_vec(),
        anchor: sr.anchor,
        knobs: zip_knobs(&sr.knobs, theta),
        grid,
        extent: sr.ext.node.clone(),
        operand_extent: sr.ext.operand.clone(),
        trips: sr.ext.trips.clone(),
        op_trips: sr.ext.op_trips.clone(),
        region: pr.region,
        in_region: pr.in_region,
        rstep: sr.rstep.clone(),
        staged,
        fanout: 1,
        children: vec![],
        cost: Cost::default(),
    };
    s.cost.footprint = footprint(cx, &s);
    occupancy_and_key(cx, &mut s, sr.parents)?;
    Ok(s)
}

/// Up to `top_k` complete schedules of `N` at level `l`, cheapest estimate
/// first, or none if no θ is legal (Fig. 7 `SubGraphTiling`; Welder
/// `emit_config`).
///
/// The search starts from Welder's base tile: each knob at its minimum, then
/// each knob in turn jumped to its maximum if that strictly lowers the work
/// per output item. If that base is illegal the search has nowhere to start
/// and returns no candidates, as Welder's does. It then expands best-first on
/// `cost.key`, growing one knob a step at a time, until `max_visited` tilings
/// have been evaluated.
pub(super) fn tile_group(
    cx: &Ctx,
    l: Lvl,
    n_set: &[NodeId],
    parents: &[&GroupSchedule],
) -> Vec<GroupSchedule> {
    if l + 1 < cx.hw.top() {
        return vec![]; // needs A13's children: not implemented
    }
    let (anchor, sinks) = choose_anchor(cx, l, n_set);
    let ext = extents_at(cx, n_set, parents);
    let rstep = initial_rsteps(cx, n_set, &ext, parents);
    let knobs = knobs_for(cx, l, anchor, &ext.node[&anchor]);
    let sr = Search {
        l,
        n_set,
        anchor,
        sinks,
        knobs,
        rstep,
        ext,
        parents,
    };

    let mut base: Vec<u64> = sr.knobs.iter().map(|k| k.values[0]).collect();
    let mut w = wpi(cx, &sr, &base);
    for (i, k) in sr.knobs.iter().enumerate() {
        let mut t = base.clone();
        t[i] = *k.values.last().expect("block_values is never empty");
        let wt = wpi(cx, &sr, &t);
        if wt < w {
            base = t;
            w = wt;
        }
    }

    // Best-first expansion (Welder DFS_smem_tile). Knob values only grow from the base.
    let steps: Vec<Vec<u64>> = sr
        .knobs
        .iter()
        .zip(&base)
        .map(|(k, &b)| k.values.iter().copied().filter(|&v| v >= b).collect())
        .collect();
    let mut visited: HashMap<Vec<u64>, Option<GroupSchedule>> = HashMap::new();
    let mut heap: BinaryHeap<Reverse<MinKey>> = BinaryHeap::new();
    let push = |visited: &mut HashMap<_, _>, heap: &mut BinaryHeap<_>, theta: Vec<u64>| {
        if visited.contains_key(&theta) {
            return;
        }
        let r = eval_tiling(cx, &sr, &theta).ok();
        if let Some(s) = &r {
            heap.push(Reverse(MinKey(s.cost.key, theta.clone())));
        }
        visited.insert(theta, r); // invalid ⇒ never expanded
    };
    push(&mut visited, &mut heap, base);
    while visited.len() <= cx.params.max_visited {
        let Some(Reverse(MinKey(_, theta))) = heap.pop() else {
            break;
        };
        for i in (0..sr.knobs.len()).rev() {
            let at = steps[i]
                .iter()
                .position(|&v| v == theta[i])
                .expect("θ only takes values from `steps`");
            if at + 1 < steps[i].len() {
                let mut t = theta.clone();
                t[i] = steps[i][at + 1];
                push(&mut visited, &mut heap, t);
            }
        }
    }
    let mut cands: Vec<(Vec<u64>, GroupSchedule)> = visited
        .into_iter()
        .filter_map(|(t, s)| s.map(|s| (t, s)))
        .collect();
    cands.sort_by(|a, b| a.1.cost.key.total_cmp(&b.1.cost.key).then(a.0.cmp(&b.0)));

    // Finish candidates in key order until top_k complete schedules exist.
    let mut out = Vec::new();
    for (_, mut s) in cands {
        expand_rsteps(cx, &mut s, parents);
        s.fanout = 1;
        finish_cost(cx, &mut s, parents);
        out.push(s);
        if out.len() >= cx.params.top_k {
            break;
        }
    }
    // Stable, so equal estimates keep key order.
    out.sort_by(|a, b| a.cost.est.total_cmp(&b.cost.est));
    out
}

#[cfg(test)]
mod tests {
    use super::super::cost::AnalyticEvaluator;
    use super::super::seed::KnobName;
    use super::super::span::DimSpan;
    use super::super::testing::{
        Builder, Softmax, bind, device_and_shared, flat, kernel, looping, lowered_conv, softmax,
        tensor,
    };
    use super::super::{Capabilities, ConnectPlan, SearchParams, TeenygradCaps};
    use super::*;
    use crate::graph::optimizer::anduin::tile_graph::TileGraph;

    fn params(top_k: usize) -> SearchParams {
        SearchParams {
            top_k,
            max_visited: 2000,
            fanouts_per_tile: 1,
            max_closure: 10,
            max_recompute: 1.0,
            min_gain: 0.0,
            max_passes: 1,
            max_measurements: 0,
        }
    }

    /// Runs `f` over `g` with every edge inside `n_set` at H (1) and the rest
    /// at 0, on device and shared memory.
    fn with_group<R>(
        g: &TileGraph,
        n_set: &[NodeId],
        caps: &dyn Capabilities,
        top_k: usize,
        f: impl FnOnce(&Ctx) -> R,
    ) -> R {
        let hw = device_and_shared();
        let params = params(top_k);
        let plan = ConnectPlan {
            level: g
                .edges
                .iter()
                .map(|e| usize::from(n_set.contains(&e.src) && n_set.contains(&e.dst)))
                .collect(),
        };
        f(&Ctx {
            g,
            hw: &hw,
            caps,
            params: &params,
            ev: &AnalyticEvaluator,
            plan: &plan,
        })
    }

    fn whole(s: &Softmax) -> [NodeId; 5] {
        [s.max, s.sub, s.exp, s.sum, s.div]
    }

    /// The fixed inputs `tile_group` would search `n_set` with.
    fn search<'a>(cx: &Ctx, n_set: &'a [NodeId]) -> Search<'a> {
        let (anchor, sinks) = choose_anchor(cx, 0, n_set);
        let ext = extents_at(cx, n_set, &[]);
        let knobs = knobs_for(cx, 0, anchor, &ext.node[&anchor]);
        Search {
            l: 0,
            n_set,
            anchor,
            sinks,
            knobs,
            rstep: BTreeMap::new(),
            ext,
            parents: &[],
        }
    }

    #[test]
    fn test_the_anchor_is_the_latest_non_rider_and_sinks_have_no_consumer_in_n() {
        // x -> a -> r1 (rider) -> r2 (rider), a -> s (sink of its own). `a`
        // and `s` loop, so neither is a broadcast map.
        let mut b = Builder::default();
        let x = b.input("x", &[8]);
        let a = b.op("a", &[8], looping(flat(1, 1)), &[x]);
        let r1 = b.op("r1", &[8], flat(1, 1), &[a]);
        let s = b.op("s", &[8], looping(flat(1, 1)), &[a]);
        let r2 = b.op("r2", &[8], flat(1, 1), &[r1]);
        let g = b.build();

        let (anchor, sinks) = with_group(&g, &[a, r1, r2], &TeenygradCaps, 10, |cx| {
            choose_anchor(cx, 0, &[a, r1, r2])
        });
        assert_eq!((anchor, sinks), (a, vec![r2]));

        let (anchor, sinks) = with_group(&g, &[a, r1, s], &TeenygradCaps, 10, |cx| {
            choose_anchor(cx, 0, &[a, r1, s])
        });
        assert_eq!(anchor, s, "s is the latest non-rider");
        assert_eq!(sinks, vec![r1, s]);

        let (anchor, _) = with_group(&g, &[r1, r2], &TeenygradCaps, 10, |cx| {
            choose_anchor(cx, 0, &[r1, r2])
        });
        assert_eq!(
            anchor, r1,
            "with only riders, the earliest of the final shape"
        );
    }

    #[test]
    fn test_a_rider_wider_than_the_latest_non_rider_anchors_instead() {
        // A rider runs on the anchor's coordinates, so it needs the anchor's
        // shape. In softmax every node after `sum` ([R, 1]) is a rider, but
        // `div` is [R, C]: it anchors. Among nodes of the final shape the
        // earliest anchors, so in {max, sub, exp} it is the row-broadcast
        // `sub`, not the flat `exp` after it.
        let s = softmax(64, 1024);
        let anchor = |n_set: &[NodeId]| {
            with_group(&s.g, n_set, &TeenygradCaps, 10, |cx| {
                choose_anchor(cx, 0, n_set)
            })
            .0
        };

        assert_eq!(anchor(&whole(&s)), s.div);
        assert_eq!(anchor(&[s.max, s.sub, s.exp]), s.sub);
        assert_eq!(anchor(&[s.exp, s.sum]), s.sum, "no node follows sum");
    }

    #[test]
    fn test_p3_a_lowered_conv2d_reads_a_halo_not_the_whole_input_row() {
        // teenygrad-a19a: the lowered conv2d answers for STRIDE_W, PAD_W and
        // KW, so its windowed W axis propagates as a tile with a halo rather
        // than `Full`. (H stays `Full`: conv2d's OH is not bound yet, P4,
        // teenygrad-3f4z.) The search itself picks BLOCK_OW = OW, a whole row,
        // so this evaluates a narrower θ directly.
        for stride in [1u64, 2] {
            let (g, conv) = lowered_conv(stride as usize);
            let n_set = [conv];
            with_group(&g, &n_set, &TeenygradCaps, 10, |cx| {
                let sr = search(cx, &n_set);
                let names: Vec<KnobName> = sr.knobs.iter().map(|k| k.name).collect();
                assert_eq!(names, vec![KnobName::Block("BLOCK_OW")]);
                let v = 8;
                let s = eval_tiling(cx, &sr, &[v]).expect("an 8-wide conv tile is legal");
                let x = &s.in_region[&g.nodes[conv.0].in_edges[0]];
                let DimSpan::Tiled { step, lo, len, .. } = x[3] else {
                    panic!("stride {stride}: W should be tiled, got {:?}", x[3]);
                };
                // Unit u reads columns [8·s·u - 1, 8·s·u - 1 + 7·s + 3).
                assert_eq!(
                    (step, lo, len),
                    (v * stride, -1, (v - 1) * stride + 3),
                    "stride {stride}"
                );
                assert_eq!(x[2], DimSpan::Full, "OH is unbound, so H is whole");
            });
        }
    }

    #[test]
    fn test_d1_softmax_tiles_whole_rows_anchored_on_div() {
        let s = softmax(64, 1024);
        let n_set = whole(&s);

        let cands = with_group(&s.g, &n_set, &TeenygradCaps, 10, |cx| {
            tile_group(cx, 0, &n_set, &[])
        });
        assert!(!cands.is_empty(), "D1(b): a full-row tile is legal");
        assert!(cands.len() <= 10);
        assert!(cands.windows(2).all(|w| w[0].cost.est <= w[1].cost.est));
        for c in &cands {
            assert_eq!(c.anchor, s.div);
            let block = |name| {
                c.knobs
                    .iter()
                    .find(|(k, _)| *k == KnobName::Block(name))
                    .map(|k| k.1)
            };
            assert!(
                block("BLOCK_C") >= Some(1024),
                "every legal tile spans whole rows"
            );
            for n in n_set {
                assert_eq!(c.region[&n][1], DimSpan::Full, "{}", s.g.nodes[n.0].name);
            }
            assert!(c.cost.footprint <= 96 * 1024);
        }

        // D1(c): the one external input, x, is staged by two stages (max and
        // sub) and so read twice; the output, div, is stored once.
        let tensor_bytes = 64.0 * 1024.0 * 4.0;
        assert_eq!(cands[0].cost.traffic[0], 3.0 * tensor_bytes);
    }

    #[test]
    fn test_d1a_narrow_column_tiles_are_rejected_and_the_base_tile_jumps_past_them() {
        let s = softmax(64, 1024);
        let n_set = whole(&s);
        with_group(&s.g, &n_set, &TeenygradCaps, 10, |cx| {
            let sr = search(cx, &n_set);
            let names: Vec<KnobName> = sr.knobs.iter().map(|k| k.name).collect();
            assert_eq!(
                names,
                vec![KnobName::Block("BLOCK_C")],
                "rows are a literal 1"
            );

            // Over device and shared memory every fused edge is lockstep, so a
            // column tile narrower than the row fails at exp -> div, before
            // recompute is even checked.
            let exp_div = s.g.nodes[s.div.0].in_edges[0];
            for theta in [[1], [256], [512]] {
                let r = eval_tiling(cx, &sr, &theta);
                assert!(
                    matches!(r, Err(Reject::Lockstep(e)) if e == exp_div),
                    "θ = {theta:?} should be rejected at exp -> div"
                );
            }
            // The minimum θ is illegal, but the base tile jumps BLOCK_C to its
            // maximum, so the search still has somewhere to start.
            assert!(!tile_group(cx, 0, &n_set, &[]).is_empty());
        });
    }

    #[test]
    fn test_two_knobs_are_searched_from_the_base_tile_upward() {
        // A singleton [64, 1024] elementwise op tiled by BLOCK_R and BLOCK_C.
        let mut b = Builder::default();
        let x = b.input("x", &[64, 1024]);
        let t = tensor(2, &[bind(&[0], "BLOCK_R", "R"), bind(&[1], "BLOCK_C", "C")]);
        let n = b.op("n", &[64, 1024], kernel(&[t], t), &[x]);
        let g = b.build();

        let cands = with_group(&g, &[n], &TeenygradCaps, 10, |cx| {
            let n_set = [n];
            let sr = search(cx, &n_set);
            assert_eq!(sr.knobs.len(), 2);
            tile_group(cx, 0, &n_set, &[])
        });
        assert_eq!(cands.len(), 10);
        // One node does the same work per output item at every θ, so the base
        // tile stays at the minimum and the search grows both knobs from it.
        let distinct = |i: usize| {
            cands
                .iter()
                .map(|c| c.knobs[i].1)
                .collect::<std::collections::BTreeSet<u64>>()
                .len()
        };
        assert!(distinct(0) > 1, "BLOCK_R should vary across the candidates");
        assert!(distinct(1) > 1, "BLOCK_C should vary across the candidates");
        assert!(cands.iter().all(|c| c.cost.footprint <= 96 * 1024));
    }

    #[test]
    fn test_tile_group_respects_top_k_and_is_deterministic() {
        let s = softmax(64, 1024);
        let n_set = whole(&s);
        let run = || {
            with_group(&s.g, &n_set, &TeenygradCaps, 3, |cx| {
                tile_group(cx, 0, &n_set, &[])
            })
            .into_iter()
            .map(|c| (c.knobs, c.cost.est.to_bits()))
            .collect::<Vec<_>>()
        };

        let first = run();
        assert!(!first.is_empty() && first.len() <= 3);
        assert_eq!(run(), first);
    }

    #[test]
    fn test_a_singleton_elementwise_op_reads_and_writes_each_byte_once() {
        // One tiling level and no fusion: traffic is the operand plus the
        // output, once a block's run fills a 128-byte read transaction (32
        // f32s). A shorter run wastes the rest of each transaction.
        let s = softmax(64, 1024);
        let cands = with_group(&s.g, &[s.exp], &TeenygradCaps, 10, |cx| {
            tile_group(cx, 0, &[s.exp], &[])
        });

        assert!(!cands.is_empty());
        let tensor_bytes = 64.0 * 1024.0 * 4.0;
        for c in &cands {
            let [(_, block)] = c.knobs.as_slice() else {
                panic!("a flat kernel has one knob, got {:?}", c.knobs);
            };
            let read = tensor_bytes * (32.0 / (*block as f64).min(32.0));
            assert_eq!(c.cost.traffic[0], read + tensor_bytes, "θ = {:?}", c.knobs);
        }
    }

    #[test]
    fn test_s5_a_row_too_wide_for_shared_memory_has_no_tiling() {
        // A reduction reading whole 2^20-element rows: 4 MiB per unit.
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
        let g = b.build();

        let cands = with_group(&g, &[r], &TeenygradCaps, 10, |cx| {
            let n_set = [r];
            let sr = search(cx, &n_set);
            assert!(matches!(eval_tiling(cx, &sr, &[]), Err(Reject::Capacity)));
            tile_group(cx, 0, &[r], &[])
        });
        assert!(cands.is_empty());
    }
}
