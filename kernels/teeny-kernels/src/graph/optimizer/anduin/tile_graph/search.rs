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

//! Welder §4.1's `EnumerateSubtiles`/`get_base_tile`/`DFS_smem_tile`: an
//! expanding search over candidate output-tile shapes for one node.

use std::collections::HashMap;

use crate::errors::Result;

use super::types::{TileDim, TileEdgeShape, TileOp};
use super::{EdgeId, NodeId, TileGraph};

/// Upper bound on tiles visited by [`TileGraph::enumerate_subtiles`]'s
/// expanding search, mirroring Welder's own `DFS_smem_tile` visited-tile
/// cap (2000) — kept smaller here since a power-of-two-only search space
/// is already much smaller than Welder's any-divisor one.
const MAX_ENUMERATED_TILES: usize = 512;

/// `1, 2, 4, ..., extent.next_power_of_two()` — the candidate tile-size
/// ladder for one [`TileDim::Fixed`] axis in
/// [`TileGraph::enumerate_subtiles`]. See that method's doc comment for
/// why powers of two, not arbitrary divisors.
fn power_of_two_ladder(extent: usize) -> Vec<usize> {
    let top = extent.max(1).next_power_of_two();
    let mut ladder = Vec::new();
    let mut step = 1usize;
    loop {
        ladder.push(step);
        if step >= top {
            break;
        }
        step *= 2;
    }
    ladder
}

/// One [`TileGraph::enumerate_subtiles`] search axis: either an ordinary
/// single real dim, or (when `root`'s tile_spec declares a
/// [`TileAxisBinding`](teeny_core::model::TileAxisBinding) whose `dims`
/// spans more than one real axis — teenygrad-1nr.8/.9) a flattened group
/// of them, searched as one combined ladder.
struct SearchAxis {
    /// Real dims this axis spans, outer to inner (mirrors
    /// [`TileAxisBinding::dims`](teeny_core::model::TileAxisBinding::dims)).
    /// Exactly one entry for the ordinary single-dim case.
    dims: Vec<usize>,
    /// Candidate values for the innermost dim. A single, unchanging
    /// entry means "not enumerable" (a [`TileDim::Sym`] axis, or a
    /// flattened group involving one).
    ladder: Vec<TileDim>,
}

impl SearchAxis {
    /// Writes `value` (one of `self.ladder`'s entries) into `candidate`:
    /// the innermost dim gets `value`, every other spanned dim collapses
    /// to `Fixed(1)` — product-preserving, the same convention
    /// [`TileGraph::propagate`] already applies downstream (see
    /// [`TileAxisBinding::dims`](teeny_core::model::TileAxisBinding::dims)'s
    /// doc comment). A no-op on any dim beyond `candidate`'s own length.
    fn write(&self, candidate: &mut TileEdgeShape, value: &TileDim) {
        let Some((&innermost, outer_dims)) = self.dims.split_last() else {
            return;
        };
        if let Some(dim) = candidate.get_mut(innermost) {
            *dim = value.clone();
        }
        for &outer in outer_dims {
            if let Some(dim) = candidate.get_mut(outer) {
                *dim = TileDim::Fixed(1);
            }
        }
    }
}

impl TileGraph {
    /// Propagates `tile` as `root`'s requested output shape (seeded on
    /// `output_edge`) through `nodes`, and scores the result as
    /// [`Self::mem_traffic_with_config`] bytes per output element —
    /// `None` if [`Self::mem_footprint_with_config`] exceeds `capacity`
    /// (an invalid candidate). Lower is better. Shared by
    /// [`Self::enumerate_subtiles`]'s base-tile growth and expanding
    /// search, which both need exactly this.
    fn score_candidate_tile(
        &self,
        nodes: &[NodeId],
        output_edge: EdgeId,
        capacity: u64,
        tile: &TileEdgeShape,
    ) -> Result<Option<f64>> {
        let mut seed = HashMap::new();
        seed.insert(output_edge, tile.clone());
        let config = self.propagate(nodes, &seed)?;
        if self.mem_footprint_with_config(nodes, &config) > capacity {
            return Ok(None);
        }
        let traffic = self.mem_traffic_with_config(nodes, &config);
        let elements: u64 = tile
            .iter()
            .map(|dim| match dim {
                TileDim::Fixed(extent) => *extent as u64,
                TileDim::Sym(_) => 1,
            })
            .product::<u64>()
            .max(1);
        Ok(Some(traffic as f64 / elements as f64))
    }

    /// Welder §4.1's `EnumerateSubtiles`: a Roller-style expanding search
    /// over candidate output-tile shapes for `root` (rank/axes taken from
    /// its own full output shape), for [`Self::sub_graph_tiling`] to
    /// `propagate` and score.
    ///
    /// Candidate sizes per [`TileDim::Fixed`] axis are restricted to
    /// powers of two — `1, 2, 4, ..., extent.next_power_of_two()` — a
    /// hard requirement of this codebase's Triton backend (`tl.arange`
    /// and friends need power-of-two extents; see e.g.
    /// `next_power_of_two` at `TritonLowering`'s softmax/reduction
    /// lowering sites, and the `BLOCK_SIZE`-must-be-a-power-of-two doc
    /// comments across `nn::attention::flash_attn2`, `nn::norm::groupnorm`,
    /// `nn::activation::softmax`, `nn::loss::{embedding,nll,ranking}`).
    /// This is *preferred*, not literally universal: when an axis's
    /// extent isn't itself a power of two, a chosen block size can still
    /// leave a partial/masked last tile when a grid steps across that
    /// axis — this search never needs to compute or represent that
    /// remainder tile itself, only the candidate block size, exactly like
    /// those existing kernels already handle it. A [`TileDim::Sym`]
    /// (dynamic) axis isn't enumerable this way and is left unchanged in
    /// every candidate.
    ///
    /// When `root`'s own `tile_spec` declares a usable output
    /// [`TensorTileSpec`](teeny_core::model::TensorTileSpec) (its `rank`
    /// matching `root`'s real output rank), the search is driven by that
    /// spec's [`TileAxisBinding`](teeny_core::model::TileAxisBinding)s
    /// instead of raw per-real-dim ladders (teenygrad-1nr.9): each
    /// binding becomes one [`SearchAxis`], whose ladder ranges over the
    /// *combined* extent of every real dim it spans (the product, for a
    /// flattened multi-dim binding — teenygrad-1nr.8's `dims`), written
    /// with the same innermost-gets-the-value/other-spanned-dims-collapse-
    /// to-`Fixed(1)` convention [`Self::propagate`] already applies
    /// downstream — so every candidate this search returns is one
    /// `propagate` (and the real kernel) can actually realize. A real dim
    /// with no axis binding at all (untiled) is never varied, staying at
    /// its full extent in every candidate, matching `propagate`'s own
    /// fallback. Falls back to the previous one-axis-per-real-dim search
    /// when `root` has no usable tile_spec — still the overwhelming
    /// majority of nodes (coverage is opt-in) — a strict,
    /// behavior-preserving extension for every currently-covered
    /// single-dim-axes spec (`Relu`/`MatMul`).
    ///
    /// Unlike Welder's own `DFS_smem_tile` (`../Welder/python/welder/policy/default.py`),
    /// whose candidate steps are *any* divisor of the axis extent (with a
    /// handful of powers of two spliced in only for large primes), this
    /// search's candidate steps are the power-of-two ladder above —
    /// actually a *smaller* search space (`O(log extent)` per axis
    /// instead of `O(divisor count)`).
    ///
    /// Algorithm, adapted from `get_base_tile` + `DFS_smem_tile`:
    /// 1. **Base tile**: starting from all-`Fixed(1)`, grow one axis at a
    ///    time through its ladder while [`Self::score_candidate_tile`]
    ///    (traffic per output element) keeps improving, stopping at the
    ///    first non-improving step — Welder's own `get_base_tile` does the
    ///    same per-axis greedy growth (a "workload per item" metric there;
    ///    `MemTraffic` here, since that's what this search is ultimately
    ///    ranking candidates by anyway).
    /// 2. **Expanding search**: from the base tile, repeatedly take the
    ///    best-scoring visited tile not yet expanded and try bumping each
    ///    axis to its next ladder step, scoring and recording every new
    ///    tile — Welder's own priority-queue neighbor expansion, capped at
    ///    [`MAX_ENUMERATED_TILES`] visited tiles (same idea as Welder's own
    ///    2000-tile cap, just a smaller bound given this search space is
    ///    already much smaller).
    ///
    /// Returns every valid (footprint ≤ `capacity`) visited tile, sorted by
    /// ascending score (best first).
    pub fn enumerate_subtiles(
        &self,
        nodes: &[NodeId],
        root: NodeId,
        capacity: u64,
    ) -> Result<Vec<TileEdgeShape>> {
        let full_shape = self.node_output_shape(root).clone();
        let Some(output_edge) = self
            .output_edge_id(root)
            .or_else(|| self.children(root).first().map(|&(_, id)| id))
        else {
            return Ok(Vec::new());
        };

        let (search_axes, mut base) = Self::search_axes_for(&full_shape, self.node(root));

        let mut visited: HashMap<TileEdgeShape, Option<f64>> = HashMap::new();
        let mut queue: Vec<(f64, TileEdgeShape)> = Vec::new();

        // Base tile: grow each axis independently while the score keeps
        // improving. Every candidate tried (not just the axis's final
        // choice) is recorded via `visit_candidate`, so it's also a
        // returnable result, not just a stepping stone.
        for axis in &search_axes {
            if axis.ladder.len() <= 1 {
                continue;
            }
            let mut best_idx = 0usize;
            let mut best_score = self.visit_candidate(
                nodes,
                output_edge,
                capacity,
                base.clone(),
                &mut visited,
                &mut queue,
            )?;
            for (idx, step) in axis.ladder.iter().enumerate().skip(1) {
                let mut candidate = base.clone();
                axis.write(&mut candidate, step);
                let candidate_score = self.visit_candidate(
                    nodes,
                    output_edge,
                    capacity,
                    candidate,
                    &mut visited,
                    &mut queue,
                )?;
                let Some(candidate_score) = candidate_score else {
                    continue;
                };
                let improved = match best_score {
                    Some(best) => candidate_score < best,
                    None => true,
                };
                if !improved {
                    break;
                }
                best_score = Some(candidate_score);
                best_idx = idx;
            }
            axis.write(&mut base, &axis.ladder[best_idx]);
        }

        // Expanding neighbor search: repeatedly take the best-scoring
        // visited-but-not-yet-expanded tile and try bumping each axis to
        // its next ladder step.
        while !queue.is_empty() && visited.len() < MAX_ENUMERATED_TILES {
            // Manual min-index scan, not `.iter().min_by(..).expect(..)`:
            // `queue` is non-empty (the loop guard above), so `min_idx`
            // starting at 0 is always a valid index -- no `Option` to
            // unwrap.
            let mut min_idx = 0;
            for idx in 1..queue.len() {
                if queue[idx].0.total_cmp(&queue[min_idx].0).is_lt() {
                    min_idx = idx;
                }
            }
            let (_, tile) = queue.remove(min_idx);

            for axis in &search_axes {
                let Some(&innermost) = axis.dims.last() else {
                    continue;
                };
                let Some(current) = tile.get(innermost) else {
                    continue;
                };
                let Some(idx) = axis.ladder.iter().position(|dim| dim == current) else {
                    continue;
                };
                if idx + 1 >= axis.ladder.len() {
                    continue;
                }
                let mut neighbor = tile.clone();
                axis.write(&mut neighbor, &axis.ladder[idx + 1]);
                self.visit_candidate(
                    nodes,
                    output_edge,
                    capacity,
                    neighbor,
                    &mut visited,
                    &mut queue,
                )?;
            }
        }

        let mut results: Vec<(f64, TileEdgeShape)> = visited
            .into_iter()
            .filter_map(|(tile, score)| score.map(|score| (score, tile)))
            .collect();
        results.sort_by(|a, b| a.0.total_cmp(&b.0));
        Ok(results.into_iter().map(|(_, tile)| tile).collect())
    }

    /// Builds [`Self::enumerate_subtiles`]'s search axes and initial base
    /// tile for `full_shape`, driven by `node`'s own declared output
    /// [`TensorTileSpec`](teeny_core::model::TensorTileSpec) when it's
    /// usable (its `rank` matches `full_shape.len()`) — one [`SearchAxis`]
    /// per [`TileAxisBinding`](teeny_core::model::TileAxisBinding),
    /// spanning a flattened group of real dims when `dims` names more
    /// than one, with every uncovered real dim left untiled (never
    /// varied, always its own full extent). Falls back to one ordinary
    /// per-real-dim axis each when `node` has no usable tile_spec.
    fn search_axes_for(
        full_shape: &TileEdgeShape,
        node: &TileOp,
    ) -> (Vec<SearchAxis>, TileEdgeShape) {
        let output_spec = node
            .tile_spec
            .as_ref()
            .and_then(|spec| spec.outputs.first())
            .filter(|output_spec| output_spec.rank == full_shape.len());

        let Some(output_spec) = output_spec else {
            let base: TileEdgeShape = full_shape
                .iter()
                .map(|dim| match dim {
                    TileDim::Fixed(_) => TileDim::Fixed(1),
                    TileDim::Sym(name) => TileDim::Sym(name.clone()),
                })
                .collect();
            let axes = (0..full_shape.len())
                .map(|d| SearchAxis {
                    dims: vec![d],
                    ladder: match &full_shape[d] {
                        TileDim::Fixed(extent) => power_of_two_ladder(*extent)
                            .into_iter()
                            .map(TileDim::Fixed)
                            .collect(),
                        TileDim::Sym(name) => vec![TileDim::Sym(name.clone())],
                    },
                })
                .collect();
            return (axes, base);
        };

        // Untiled dims (no axis binding at all) keep their full extent,
        // never varied -- start from a full-shape clone and only
        // overwrite the dims an axis binding actually spans.
        let mut base = full_shape.clone();
        let mut axes = Vec::new();
        for axis in output_spec.axes.iter() {
            let Some((&innermost, _)) = axis.dims.split_last() else {
                continue; // empty dims: nothing to bind (spec-authoring bug)
            };
            let all_fixed = axis
                .dims
                .iter()
                .all(|&d| matches!(full_shape.get(d), Some(TileDim::Fixed(_))));
            let ladder = if all_fixed {
                let combined_extent: usize = axis
                    .dims
                    .iter()
                    .filter_map(|&d| match full_shape.get(d) {
                        Some(TileDim::Fixed(extent)) => Some(*extent),
                        _ => None,
                    })
                    .product();
                power_of_two_ladder(combined_extent)
                    .into_iter()
                    .map(TileDim::Fixed)
                    .collect()
            } else {
                // A Sym-typed dim is in this group: not enumerable,
                // mirrors the ordinary per-dim Sym case (a single
                // unchanging entry).
                vec![
                    full_shape
                        .get(innermost)
                        .cloned()
                        .unwrap_or(TileDim::Fixed(1)),
                ]
            };
            for &d in axis.dims.iter() {
                if let Some(dim @ TileDim::Fixed(_)) = base.get_mut(d) {
                    *dim = TileDim::Fixed(1);
                }
            }
            axes.push(SearchAxis {
                dims: axis.dims.to_vec(),
                ladder,
            });
        }
        (axes, base)
    }

    /// Scores `tile` via [`Self::score_candidate_tile`] and records it in
    /// `visited`/`queue`, unless it's already been visited (returns the
    /// cached score in that case, without rescoring or re-queuing). Shared
    /// by [`Self::enumerate_subtiles`]'s base-tile growth and expanding
    /// search, so every candidate either phase tries ends up in the
    /// returned result set, not just each axis's final choice.
    fn visit_candidate(
        &self,
        nodes: &[NodeId],
        output_edge: EdgeId,
        capacity: u64,
        tile: TileEdgeShape,
        visited: &mut HashMap<TileEdgeShape, Option<f64>>,
        queue: &mut Vec<(f64, TileEdgeShape)>,
    ) -> Result<Option<f64>> {
        if let Some(&existing) = visited.get(&tile) {
            return Ok(existing);
        }
        let score = self.score_candidate_tile(nodes, output_edge, capacity, &tile)?;
        visited.insert(tile.clone(), score);
        if let Some(score) = score {
            queue.push((score, tile));
        }
        Ok(score)
    }
}

#[cfg(test)]
mod tests {
    use teeny_core::model::ExecutableOp;
    use teeny_core::utils::dag::Dag;

    use super::super::testing::{batchnorm2d_shaped_spec, op, op_with_tile_spec};
    use super::super::{TileDim, TileEdgeShape, TileGraph};

    #[test]
    fn enumerate_subtiles_only_returns_power_of_two_extents() {
        // 100 isn't a power of two -- every candidate extent must still be.
        let mut dag: Dag<Box<dyn ExecutableOp>> = Dag::new();
        let a = dag.add_node(op("a", vec![Some(100)], true));

        let tile_graph = TileGraph::from_dag(&dag);
        let results = tile_graph.enumerate_subtiles(&[a], a, u64::MAX).unwrap();

        assert!(!results.is_empty());
        for shape in &results {
            let TileDim::Fixed(extent) = shape[0] else {
                panic!("expected a Fixed axis");
            };
            assert!(extent.is_power_of_two(), "{extent} is not a power of two");
        }
    }

    #[test]
    fn enumerate_subtiles_never_exceeds_capacity() {
        // a is isolated: an F32 [64] input boundary edge (256B, fixed,
        // unaffected by the candidate) and an output boundary edge (the
        // seeded candidate). Capacity 320 = 256 + 64B admits candidate
        // extents up to 16 (16 * 4B = 64B) but not 32 (128B -> 384B total).
        let mut dag: Dag<Box<dyn ExecutableOp>> = Dag::new();
        let a = dag.add_node(op("a", vec![Some(64)], true));

        let tile_graph = TileGraph::from_dag(&dag);
        let results = tile_graph.enumerate_subtiles(&[a], a, 320).unwrap();

        assert!(!results.is_empty());
        for shape in &results {
            let TileDim::Fixed(extent) = shape[0] else {
                panic!("expected a Fixed axis");
            };
            assert!(
                extent <= 16,
                "extent {extent} should have exceeded capacity"
            );
        }
    }

    #[test]
    fn enumerate_subtiles_treats_a_dynamic_axis_as_unenumerable() {
        let mut dag: Dag<Box<dyn ExecutableOp>> = Dag::new();
        let a = dag.add_node(op("a", vec![None, Some(8)], true));

        let tile_graph = TileGraph::from_dag(&dag);
        let dynamic_axis = tile_graph.output_edge(a).unwrap().shape[0].clone();
        let results = tile_graph.enumerate_subtiles(&[a], a, u64::MAX).unwrap();

        assert!(!results.is_empty());
        for shape in &results {
            assert_eq!(shape[0], dynamic_axis);
            assert!(matches!(shape[1], TileDim::Fixed(extent) if extent.is_power_of_two()));
        }
    }

    #[test]
    fn enumerate_subtiles_is_sorted_by_ascending_score() {
        let mut dag: Dag<Box<dyn ExecutableOp>> = Dag::new();
        let a = dag.add_node(op("a", vec![Some(64)], true));

        let tile_graph = TileGraph::from_dag(&dag);
        let output_edge = tile_graph.output_edge_id(a).unwrap();
        let results = tile_graph.enumerate_subtiles(&[a], a, u64::MAX).unwrap();

        assert!(
            results.len() > 1,
            "need at least 2 results to check ordering"
        );
        let scores: Vec<f64> = results
            .iter()
            .map(|tile| {
                tile_graph
                    .score_candidate_tile(&[a], output_edge, u64::MAX, tile)
                    .unwrap()
                    .expect("every returned tile should still score as valid")
            })
            .collect();
        for pair in scores.windows(2) {
            assert!(
                pair[0] <= pair[1],
                "results are not sorted ascending: {scores:?}"
            );
        }
    }

    #[test]
    fn enumerate_subtiles_ignores_flattened_multi_dim_axes() {
        // Regression test for teenygrad-1nr.9: b declares
        // batchnorm2d_shaped_spec (H, W flattened into one BLOCK_HW-style
        // axis via `dims: &[2, 3]`), but enumerate_subtiles builds its
        // search space purely from b's real, per-axis full shape -- with
        // no notion that dims 2 and 3 are jointly driven by one flattened
        // tile_spec axis. It grows H (dim 2) and W (dim 3) independently,
        // producing candidates like H=2, W=4 that no real
        // batch_norm_2d_nchw_forward_inference kernel configuration could
        // ever realize -- the kernel can only pick one flat BLOCK_HW count
        // over the combined H*W range (see TileAxisBinding::dims's doc
        // comment, and propagate's own convention -- teenygrad-1nr.8 -- of
        // collapsing every outer flattened dim to Fixed(1)).
        //
        // A flattened-axis-aware search should never grow H independently:
        // every candidate's dim-2 entry should stay at Fixed(1), exactly
        // like propagate already collapses it on the input side. This
        // currently FAILS -- enumerate_subtiles has no awareness of
        // tile_spec at all.
        let full_shape = vec![Some(2), Some(4), Some(8), Some(8)]; // [B, C, H, W]
        let mut dag: Dag<Box<dyn ExecutableOp>> = Dag::new();
        let b = dag.add_node(op_with_tile_spec(
            "b",
            full_shape,
            false,
            batchnorm2d_shaped_spec(),
        ));

        let tile_graph = TileGraph::from_dag(&dag);
        let results = tile_graph.enumerate_subtiles(&[b], b, u64::MAX).unwrap();

        assert!(!results.is_empty());
        let independently_varied_h: Vec<&TileEdgeShape> = results
            .iter()
            .filter(|shape| !matches!(shape[2], TileDim::Fixed(1)))
            .collect();
        assert!(
            independently_varied_h.is_empty(),
            "expected every candidate to keep H (dim 2, the outer half of \
             the flattened HW axis) at Fixed(1) -- enumerate_subtiles has \
             no notion that H and W are jointly driven by one tile_spec \
             axis, so it grows them independently. Candidates with \
             H != 1: {independently_varied_h:?}"
        );
    }
}
