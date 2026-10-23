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

//! Region propagation: one node over its `KernelTileSpec` (A6), and a whole
//! group from its sinks back to its inputs (A8).

use std::collections::BTreeMap;

use super::span::{DimSpan, RStep, Region, chunk_to_full, contains, union_region};
use super::{Ctx, Reject, internal_edges, meta};
use crate::graph::optimizer::anduin::tile_graph::{EdgeId, NodeId, OpMeta};

/// A window const's value: a decimal literal stands for itself, anything else
/// is asked of the node's `consts` (as `tile_resolve::window_const`).
fn konst(name: &str, m: &OpMeta) -> Option<u64> {
    match name.parse::<u64>() {
        Ok(v) => Some(v),
        Err(_) => m.consts.get(name).map(|v| v as u64),
    }
}

/// The receptive field: identical to resolve_one's `(block-1)*stride + kernel`
/// for `len`, and it also tracks origin and step.
fn window_span(s: DimSpan, stride: u64, pad: u64, k: u64) -> DimSpan {
    match s {
        DimSpan::Full => DimSpan::Full,
        DimSpan::Chunk { len, split } => DimSpan::Chunk {
            len: (len - 1) * stride + k,
            split,
        },
        DimSpan::Tiled {
            axis,
            step,
            lo,
            len,
        } => DimSpan::Tiled {
            axis,
            step: step * stride,
            lo: lo * stride as i64 - pad as i64,
            len: (len - 1) * stride + k,
        },
    }
}

/// The region of every input of `m`'s spec, given its output region `out`
/// (`resolve_inputs` with regions). An input dim no output name or rstep
/// resolves stays `Full`, as does a window whose consts are unknown (P3).
pub(super) fn propagate_node(
    m: &OpMeta,
    out: &Region,
    rstep: &RStep,
) -> Result<Vec<Region>, Reject> {
    let spec = m.spec();
    let o = spec.outputs.first().ok_or(Reject::Spec)?;
    if out.len() != o.rank {
        return Err(Reject::Spec);
    }
    // free-variable name -> the spans of the output dims it binds (outer..inner)
    let mut named: BTreeMap<&str, Vec<DimSpan>> = BTreeMap::new();
    for b in o.axes {
        let spans = b
            .dims
            .iter()
            .map(|&d| out.get(d).copied().ok_or(Reject::Spec))
            .collect::<Result<_, _>>()?;
        named.insert(b.extent_param, spans);
    }

    let mut res = Vec::with_capacity(spec.inputs.len());
    for t in spec.inputs {
        let mut r: Region = vec![DimSpan::Full; t.rank];
        for b in t.axes {
            if b.dims.iter().any(|&d| d >= t.rank) {
                return Err(Reject::Spec);
            }
            let name = b.window.map_or(b.extent_param, |w| w.output_extent_param);
            if let Some(src) = named.get(name) {
                if src.len() != b.dims.len() {
                    return Err(Reject::Spec); // P7
                }
                for (i, &d) in b.dims.iter().enumerate() {
                    r[d] = match b.window {
                        None => src[i],
                        Some(w) => match (
                            konst(w.stride_const, m),
                            konst(w.kernel_size_const, m),
                            w.pad_const.map_or(Some(0), |p| konst(p, m)),
                        ) {
                            (Some(s), Some(k), Some(pad)) if s >= 1 && k >= 1 => {
                                window_span(src[i], s, pad, k)
                            }
                            _ => DimSpan::Full, // consts unavailable (as resolve_one)
                        },
                    };
                }
            } else if let Some(c) = rstep.get(b.extent_param) {
                // P6
                if b.dims.len() != 1 || b.window.is_some() {
                    return Err(Reject::Spec);
                }
                r[b.dims[0]] = DimSpan::Chunk {
                    len: c.len,
                    split: c.split,
                };
            } // otherwise: unresolved ⇒ stays Full (conservative, as resolve_one)
        }
        res.push(r);
    }
    Ok(res)
}

pub(super) struct Propagated {
    pub region: BTreeMap<NodeId, Region>,
    pub in_region: BTreeMap<EdgeId, Region>,
}

/// Every node's output region and every in-edge's operand region for one
/// unit, from the anchor's `seed` (Welder `_compute_memory_traffic`'s walk).
///
/// Every pure sink is a positional copy of the seed. Nodes are then visited
/// consumers-first, so a node's region is final (the union of what its in-`N`
/// consumers need) by the time it is propagated. The anchor computes exactly
/// its seed, which must contain what the riders after it need.
pub(super) fn propagate_group(
    cx: &Ctx,
    n_set: &[NodeId],
    anchor: NodeId,
    sinks: &[NodeId],
    seed: &Region,
    ext: &BTreeMap<NodeId, Vec<u64>>,
    rstep: &BTreeMap<NodeId, RStep>,
) -> Result<Propagated, Reject> {
    let top = cx.hw.top();
    let mut region: BTreeMap<NodeId, Region> = BTreeMap::new();
    for &s in sinks {
        if ext[&s] != ext[&anchor] {
            return Err(Reject::SinkShape);
        }
        region.insert(s, seed.clone());
    }
    let mut in_region = BTreeMap::new();
    let mut by_pos: Vec<NodeId> = n_set.to_vec();
    by_pos.sort_by_key(|n| std::cmp::Reverse(cx.g.pos[n.0])); // consumers before producers
    let no_rstep = RStep::new();
    for n in by_pos {
        if n == anchor && !sinks.contains(&n) {
            let need = region
                .get(&n)
                .expect("a non-sink anchor has an in-N consumer");
            if !need.iter().zip(seed).all(|(s, b)| contains(*b, *s)) {
                return Err(Reject::Containment);
            }
            region.insert(n, seed.clone());
        }
        // present: n is a pure sink, or has an in-N consumer visited before it
        let r = region
            .get(&n)
            .expect("every node of N is a sink or has a consumer in N")
            .clone();
        let ins = propagate_node(meta(cx.g, n), &r, rstep.get(&n).unwrap_or(&no_rstep))?;
        for &e in &cx.g.nodes[n.0].in_edges {
            let edge = &cx.g.edges[e.0];
            // P5: in-edge port p is spec input p
            let operand = ins.get(edge.dst_port).ok_or(Reject::Spec)?;
            if operand.len() != cx.g.nodes[edge.src.0].shape.len() {
                return Err(Reject::Spec); // P2
            }
            in_region.insert(e, operand.clone());
            if !n_set.contains(&edge.src) {
                continue;
            }
            let need = if cx.plan.level[e.0] == top {
                operand.clone()
            } else {
                chunk_to_full(operand)
            };
            let merged = match region.get(&edge.src) {
                None => need,
                Some(old) => union_region(old, &need),
            };
            region.insert(edge.src, merged);
        }
    }
    // Lockstep (register / rider-splice) edges: both sides cover the same elements per unit.
    for e in internal_edges(cx.g, n_set) {
        if cx.plan.level[e.0] == top && in_region[&e] != region[&cx.g.edges[e.0].src] {
            return Err(Reject::Lockstep(e));
        }
    }
    Ok(Propagated { region, in_region })
}

#[cfg(test)]
mod tests {
    use super::super::testing::{
        Builder, bind, device_and_shared, flat, kernel, softmax, tensor, windowed,
    };
    use super::super::{ConnectPlan, Ctx, SearchParams, TeenygradCaps};
    use super::*;
    use crate::graph::optimizer::anduin::schedule::cost::AnalyticEvaluator;
    use crate::graph::optimizer::anduin::tile_graph::TileGraph;
    use DimSpan::*;

    fn tiled(axis: usize, step: u64, lo: i64, len: u64) -> DimSpan {
        Tiled {
            axis,
            step,
            lo,
            len,
        }
    }

    /// A 1-D op reading its one operand through a `K`-wide window of stride
    /// `S` and pad `P`, with those consts' values given by `consts`
    /// (`None` = unknown).
    fn conv1d(
        b: &mut Builder,
        input: NodeId,
        len: u64,
        consts: Option<(usize, usize, usize)>,
    ) -> NodeId {
        let spec = kernel(
            &[tensor(1, &[windowed(0, "L", "OL", "S", "P", "K")])],
            tensor(1, &[bind(&[0], "BLOCK", "OL")]),
        );
        let lookup = move |name: &str| {
            let (s, p, k) = consts?;
            match name {
                "S" => Some(s),
                "P" => Some(p),
                "K" => Some(k),
                _ => None,
            }
        };
        b.op_with(
            "conv",
            &[len],
            Some(spec),
            false,
            Box::new(lookup),
            &[input],
        )
    }

    fn with_ctx<R>(g: &TileGraph, plan: &[usize], f: impl FnOnce(&Ctx) -> R) -> R {
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
        let plan = ConnectPlan {
            level: plan.to_vec(),
        };
        f(&Ctx {
            g,
            hw: &hw,
            caps: &TeenygradCaps,
            params: &params,
            ev: &AnalyticEvaluator,
            plan: &plan,
        })
    }

    #[test]
    fn test_a_windowed_operand_resolves_to_its_receptive_field() {
        let mut b = Builder::default();
        let x = b.input("x", &[32]);
        let c = conv1d(&mut b, x, 32, Some((1, 1, 3)));
        let g = b.build();

        let ins = propagate_node(
            super::super::meta(&g, c),
            &vec![tiled(0, 8, 0, 8)],
            &RStep::new(),
        )
        .expect("the spec matches the region");
        // (8 - 1)·1 + 3 = 10 wide, starting one before the tile (pad 1).
        assert_eq!(ins, vec![vec![tiled(0, 8, -1, 10)]]);
    }

    #[test]
    fn test_a_window_with_unknown_consts_reads_the_whole_dim() {
        let mut b = Builder::default();
        let x = b.input("x", &[32]);
        let c = conv1d(&mut b, x, 32, None);
        let g = b.build();

        let ins = propagate_node(
            super::super::meta(&g, c),
            &vec![tiled(0, 8, 0, 8)],
            &RStep::new(),
        )
        .expect("the spec matches the region");
        assert_eq!(ins, vec![vec![Full]]);
    }

    #[test]
    fn test_p7_a_name_bound_on_different_numbers_of_dims_is_a_spec_error() {
        // The output binds N on one dim, the input on two.
        let mut b = Builder::default();
        let x = b.input("x", &[4, 8]);
        let spec = kernel(
            &[tensor(2, &[bind(&[0, 1], "BLOCK", "N")])],
            tensor(1, &[bind(&[0], "BLOCK", "N")]),
        );
        let n = b.op("n", &[32], spec, &[x]);
        let g = b.build();

        let r = propagate_node(
            super::super::meta(&g, n),
            &vec![tiled(0, 8, 0, 8)],
            &RStep::new(),
        );
        assert_eq!(r, Err(Reject::Spec));
        // A region of the wrong rank for the output is one too.
        let r = propagate_node(super::super::meta(&g, n), &vec![Full, Full], &RStep::new());
        assert_eq!(r, Err(Reject::Spec));
    }

    #[test]
    fn test_an_rstep_chunks_the_reduce_dim_and_an_unresolved_dim_stays_full() {
        // y[m] = Σ_k x[m, k] · w[k]: K names no output dim.
        let mut b = Builder::default();
        let x = b.input("x", &[64, 128]);
        let w = b.input("w", &[128]);
        let spec = kernel(
            &[
                tensor(2, &[bind(&[0], "BLOCK_M", "M"), bind(&[1], "BLOCK_K", "K")]),
                tensor(1, &[bind(&[0], "BLOCK_K", "K")]),
            ],
            tensor(1, &[bind(&[0], "BLOCK_M", "M")]),
        );
        let y = b.op("y", &[64], spec, &[x, w]);
        let g = b.build();
        let m = super::super::meta(&g, y);

        let ins = propagate_node(m, &vec![tiled(0, 16, 0, 16)], &RStep::new()).expect("valid");
        assert_eq!(ins, vec![vec![tiled(0, 16, 0, 16), Full], vec![Full]]);

        let rstep = RStep::from([("K", super::super::span::RChunk { len: 32, split: 1 })]);
        let ins = propagate_node(m, &vec![tiled(0, 16, 0, 16)], &rstep).expect("valid");
        let k = Chunk { len: 32, split: 1 };
        assert_eq!(ins, vec![vec![tiled(0, 16, 0, 16), k], vec![k]]);
    }

    #[test]
    fn test_d1_a_full_row_tile_propagates_through_softmax() {
        let s = softmax(64, 1024);
        let ext: BTreeMap<NodeId, Vec<u64>> = [s.max, s.sub, s.exp, s.sum, s.div]
            .iter()
            .map(|&n| (n, s.g.nodes[n.0].shape.clone()))
            .collect();
        let n_set = [s.max, s.sub, s.exp, s.sum, s.div];
        let mut plan = vec![1; s.g.edges.len()];
        for e in &s.g.edges {
            if !n_set.contains(&e.src) || !n_set.contains(&e.dst) {
                plan[e.id.0] = 0;
            }
        }
        let rows = vec![tiled(0, 4, 0, 4), Full];

        let pr = with_ctx(&s.g, &plan, |cx| {
            propagate_group(cx, &n_set, s.div, &[s.div], &rows, &ext, &BTreeMap::new())
        })
        .expect("a full-row tile is lockstep throughout");
        for n in n_set {
            assert_eq!(pr.region[&n], rows, "node {}", s.g.nodes[n.0].name);
        }
        let x_reads: Vec<&Region> = s.g.nodes[s.x.0]
            .out_edges
            .iter()
            .map(|e| &pr.in_region[e])
            .collect();
        assert_eq!(x_reads, vec![&rows, &rows]);

        // Narrower column tiles: div reads exp's columns tile by tile, but
        // sum (and so exp) needs whole rows, so the two sides of exp -> div
        // cover different elements.
        let narrow = vec![tiled(0, 4, 0, 4), tiled(1, 256, 0, 256)];
        let r = with_ctx(&s.g, &plan, |cx| {
            propagate_group(cx, &n_set, s.div, &[s.div], &narrow, &ext, &BTreeMap::new())
        });
        let exp_div = s.g.nodes[s.div.0].in_edges[0];
        assert!(matches!(r, Err(Reject::Lockstep(e)) if e == exp_div));
    }

    #[test]
    fn test_r1_r3_a_halo_diamond_cannot_be_lockstep() {
        // p -> conv (3-wide window) -> j, p -> q -> j. conv needs a halo of p
        // that q doesn't, so p's region is the wider union (R1). Over device
        // and shared memory every fused edge is lockstep, and p -> q covers
        // less than p computes (R3).
        let mut b = Builder::default();
        let x = b.input("x", &[32]);
        let p = b.op("p", &[32], flat(1, 1), &[x]);
        let c = conv1d(&mut b, p, 32, Some((1, 1, 3)));
        let q = b.op("q", &[32], flat(1, 1), &[p]);
        let j = b.op("j", &[32], flat(1, 2), &[c, q]);
        let g = b.build();
        let n_set = [p, c, q, j];
        let ext: BTreeMap<NodeId, Vec<u64>> = n_set.iter().map(|&n| (n, vec![32])).collect();
        let plan: Vec<usize> = g
            .edges
            .iter()
            .map(|e| usize::from(n_set.contains(&e.src) && n_set.contains(&e.dst)))
            .collect();

        let r = with_ctx(&g, &plan, |cx| {
            propagate_group(
                cx,
                &n_set,
                j,
                &[j],
                &vec![tiled(0, 8, 0, 8)],
                &ext,
                &BTreeMap::new(),
            )
        });
        let p_q = g.nodes[q.0].in_edges[0];
        assert!(
            matches!(r, Err(Reject::Lockstep(e)) if e == p_q),
            "p computes rows [8u - 1, 8u + 9), q reads [8u, 8u + 8)"
        );
    }

    #[test]
    fn test_a_rider_needing_more_than_the_anchor_computes_is_rejected() {
        // anchor -> windowed rider: the rider's tile reads a halo of the
        // anchor's, which the anchor's own decode doesn't compute.
        let mut b = Builder::default();
        let x = b.input("x", &[32]);
        let a = b.op("a", &[32], flat(1, 1), &[x]);
        let w = conv1d(&mut b, a, 32, Some((1, 1, 3)));
        let g = b.build();
        let ext: BTreeMap<NodeId, Vec<u64>> = [(a, vec![32]), (w, vec![32])].into();
        let plan: Vec<usize> = g
            .edges
            .iter()
            .map(|e| usize::from(e.src == a && e.dst == w))
            .collect();

        let r = with_ctx(&g, &plan, |cx| {
            propagate_group(
                cx,
                &[a, w],
                a,
                &[w],
                &vec![tiled(0, 8, 0, 8)],
                &ext,
                &BTreeMap::new(),
            )
        });
        assert!(matches!(r, Err(Reject::Containment)));
    }

    #[test]
    fn test_d2b_sinks_of_different_shapes_are_rejected() {
        let mut b = Builder::default();
        let x = b.input("x", &[8]);
        let p = b.op("p", &[8], flat(1, 1), &[x]);
        let s1 = b.op("s1", &[8], flat(1, 1), &[p]);
        let s2 = b.op("s2", &[4], flat(1, 1), &[p]);
        let g = b.build();
        let ext: BTreeMap<NodeId, Vec<u64>> = [(p, vec![8]), (s1, vec![8]), (s2, vec![4])].into();
        let plan = vec![0, 1, 1, 0, 0];

        let r = with_ctx(&g, &plan, |cx| {
            propagate_group(
                cx,
                &[p, s1, s2],
                s2,
                &[s1, s2],
                &vec![Full],
                &ext,
                &BTreeMap::new(),
            )
        });
        assert!(matches!(r, Err(Reject::SinkShape)));
    }
}
