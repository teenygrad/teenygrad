/*
 * Copyright (c) 2026 teenygrad (https://teenygrad.org).
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

//! Welder §3.2's `Propagate` (Fig. 6): back-propagating a target output
//! tile through a node set to resolve the tile shape required on every
//! reachable edge.

use std::collections::{HashMap, HashSet};

use super::types::{TileConfig, TileDim, TileEdgeShape};
use super::{EdgeId, NodeId, TileGraph};

/// Applies a [`TileAxisBinding::divide_by`](teeny_core::model::TileAxisBinding::divide_by)
/// divisor to `dim`, if both are present (teenygrad-1nr.10) -- a
/// [`TileDim::Sym`] (dynamic) axis, or `divide_by: None`, passes through
/// unchanged (`.clone()`d, since the caller always wants an owned value
/// back here, not a borrow of `dim`).
fn apply_divide_by(dim: &TileDim, divide_by: Option<usize>) -> TileDim {
    match (dim, divide_by) {
        (TileDim::Fixed(extent), Some(divisor)) if divisor > 0 => TileDim::Fixed(extent / divisor),
        _ => dim.clone(),
    }
}

impl TileGraph {
    /// Welder §3.2's `Propagate` (Fig. 6): back-propagates a target output
    /// tile through `nodes`, resolving the tile shape required on every
    /// reachable edge. `output_tiles` seeds the tile shape requested on
    /// `nodes`'s own boundary output edges (typically from
    /// [`Self::boundary_edges`]/[`Self::output_edge_id`]) — this is our
    /// analogue of Welder's `Map<Axis, Dim> config`.
    ///
    /// Unlike Welder's own reference implementation (which walks a symbolic
    /// tensor-expression IR we don't have — see
    /// `TILE_GRAPH_SCHEDULING_PLAN.md`), this resolves axes by declared
    /// name: each node's [`super::types::TileOp::tile_spec`] names its axes
    /// with `extent_param` strings, and any two axes sharing a name
    /// (anywhere, input or output, any tensor) are the same free variable.
    /// Seeding a node's output tile resolves every axis on its parent
    /// edges that shares one of those names; an axis whose name never
    /// appears in the output (e.g. a reduction axis) is *correctly* left
    /// unresolved — its tile size isn't derivable from the output alone,
    /// so this falls back to that axis's full, untiled extent (never
    /// smaller than needed, same optimistic-not-pessimistic philosophy as
    /// [`super::types::TileEdge::byte_size`]). The same fallback covers a
    /// dim with no
    /// [`TileAxisBinding`](teeny_core::model::TileAxisBinding) at all —
    /// [`TensorTileSpec::untiled_dims`](teeny_core::model::TensorTileSpec::untiled_dims)'s
    /// case — since every input/output tile is built at that tensor's
    /// full [`TensorTileSpec::rank`](teeny_core::model::TensorTileSpec::rank),
    /// not `axes.len()`, with only the dims `axes` actually names
    /// overwritten.
    ///
    /// Degrades to a hard boundary — stops there, doesn't guess — at any
    /// node with no declared `tile_spec` (the overwhelming majority; see
    /// [`teeny_core::model::KernelTileSpec`]'s doc comment on coverage
    /// being opt-in), whose output tile's axis count doesn't match its
    /// spec's declared output rank, or whose parent-edge count doesn't
    /// match its spec's declared input count (the same positional
    /// producer-to-spec correspondence limitation the design this revives
    /// already had — see that module's doc comment).
    pub fn propagate(
        &self,
        nodes: &[NodeId],
        output_tiles: &HashMap<EdgeId, TileEdgeShape>,
    ) -> TileConfig {
        let included: HashSet<NodeId> = nodes.iter().copied().collect();
        let mut tiles: HashMap<EdgeId, TileEdgeShape> = output_tiles.clone();

        for node in self.topological_sort().into_iter().rev() {
            if !included.contains(&node) {
                continue;
            }
            let Some(spec) = &self.node(node).tile_spec else {
                continue; // hard boundary: no declared spec
            };
            let Some(output_spec) = spec.outputs.first() else {
                continue; // hard boundary: spec declares no output
            };

            // This node's own required output tile: any outgoing edge
            // already resolved (a boundary output edge, or an internal
            // edge to an already-processed in-set consumer), falling back
            // to the full untiled shape if nothing downstream constrained
            // it yet.
            let output_tile: TileEdgeShape = self
                .children(node)
                .iter()
                .map(|&(_, id)| id)
                .chain(self.output_edge_id(node))
                .find_map(|id| tiles.get(&id).cloned())
                .unwrap_or_else(|| self.node_output_shape(node).clone());

            if output_tile.len() != output_spec.rank {
                continue; // hard boundary: declared rank doesn't match reality
            }

            // Seed resolved extent_param -> TileDim from the output tile,
            // indexed by each axis's declared innermost `dims` entry --
            // not position, since `axes` may cover fewer dims than `rank`
            // (see `untiled_dims`) and needn't list them in tensor-dim
            // order. A flattened axis's block-sized value lives on its
            // innermost real dim -- see `TileAxisBinding::dims`'s doc
            // comment. `divide_by` (teenygrad-1nr.10) applies uniformly
            // right here, so every later consumer of this name already
            // sees the divided value -- owned, not borrowed, since a
            // divided value is freshly computed, not a view into
            // `output_tile`.
            let mut resolved: HashMap<&'static str, TileDim> = output_spec
                .axes
                .iter()
                .filter_map(|axis| {
                    let &innermost = axis.dims.last()?;
                    let dim = output_tile.get(innermost)?;
                    Some((axis.extent_param, apply_divide_by(dim, axis.divide_by)))
                })
                .collect();

            // Additional declared outputs (teenygrad-1nr.11): each one
            // whose synthesized boundary edge the caller has seeded
            // contributes more resolved extent_param -> TileDim entries,
            // on top of the primary output's above. Unlike the primary
            // output, there's no full-shape fallback for these -- an
            // unseeded or rank-mismatched secondary output simply
            // contributes nothing, rather than hard-boundarying the
            // whole node the way a bad primary output does above.
            //
            // Collected into owned storage up front (mirroring
            // `output_tile` above) rather than read from `tiles` inline
            // below: `tiles` itself is mutated later in this same node's
            // processing (`tiles.insert` for each parent edge).
            let secondary_tiles: Vec<Option<TileEdgeShape>> = spec
                .outputs
                .iter()
                .enumerate()
                .skip(1)
                .map(|(secondary_index, secondary_spec)| {
                    let &edge_id = self.secondary_output_edges(node).get(secondary_index - 1)?;
                    let tile = tiles.get(&edge_id)?.clone();
                    (tile.len() == secondary_spec.rank).then_some(tile)
                })
                .collect();
            for (secondary_spec, tile) in spec.outputs.iter().skip(1).zip(secondary_tiles.iter()) {
                let Some(tile) = tile else {
                    continue; // not seeded, or declared rank doesn't match reality
                };
                for axis in secondary_spec.axes.iter() {
                    let Some(&innermost) = axis.dims.last() else {
                        continue;
                    };
                    if let Some(dim) = tile.get(innermost) {
                        resolved.insert(axis.extent_param, apply_divide_by(dim, axis.divide_by));
                    }
                }
            }

            let parents = self.parent_edges(node);
            if parents.len() != spec.inputs.len() {
                continue; // hard boundary: positional correspondence unsafe
            }
            for ((_, edge_id), input_spec) in parents.iter().zip(spec.inputs.iter()) {
                let full_shape = &self.edge(*edge_id).shape;
                if full_shape.len() != input_spec.rank {
                    continue; // this input's declared rank doesn't match reality
                }
                // Full rank, not `axes.len()`: a dim with no `TileAxisBinding`
                // (an untiled dim, or a reduction axis with no output-side
                // counterpart) keeps its full extent via this fallback,
                // rather than being dropped from the result entirely.
                let mut input_tile: TileEdgeShape = full_shape.clone();
                for axis in input_spec.axes.iter() {
                    let Some((&innermost, outer_dims)) = axis.dims.split_last() else {
                        continue; // empty dims: nothing to bind (spec-authoring bug)
                    };
                    match resolved.get(axis.extent_param) {
                        Some(resolved_dim) => {
                            if let Some(dim) = input_tile.get_mut(innermost) {
                                *dim = resolved_dim.clone();
                            }
                            // Flattened-away outer dims collapse to 1 --
                            // product-preserving, see
                            // `TileAxisBinding::dims`'s doc comment.
                            for &outer in outer_dims {
                                if let Some(dim) = input_tile.get_mut(outer) {
                                    *dim = TileDim::Fixed(1);
                                }
                            }
                        }
                        None => {
                            // Unresolved: every spanned dim keeps its
                            // full-shape fallback -- except the
                            // innermost, which `divide_by` (if declared)
                            // adjusts from the raw full extent to this
                            // axis's real, usable extent (teenygrad-1nr.10,
                            // e.g. GroupNorm's channels_per_group = C / G).
                            if axis.divide_by.is_some()
                                && let Some(dim) = input_tile.get_mut(innermost)
                            {
                                *dim = apply_divide_by(dim, axis.divide_by);
                            }
                        }
                    }
                }
                tiles.insert(*edge_id, input_tile);
            }
        }

        TileConfig { tiles }
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use teeny_core::graph::{DtypeRepr, Graph, Op};
    use teeny_core::model::LoweringMode;

    use crate::graph::TritonLowering;

    use super::super::testing::edge_between;
    use super::super::{TileDim, TileGraph};

    #[test]
    fn propagate_pointwise_ops() {
        // input -> relu -> silu, lowered through the real TritonLowering
        // rather than a dummy ExecutableOp: both `Op::Relu` and `Op::Silu`
        // get a real `flat_elementwise_tile_spec` from `graph::mod`'s
        // lowering arms (input and output share one "n_elements" axis
        // name), so a tile requested on silu's output should thread back
        // unchanged through both nodes, all the way to relu's input --
        // same shape as `flat_unary_spec`'s synthetic version, but against
        // the real kernel specs.
        let shape = vec![Some(64)];
        let mut graph = Graph::new();
        let input = graph.add_node(Op::Input, vec![], DtypeRepr::F32, shape.clone());
        let relu = graph.add_node(Op::Relu, vec![input], DtypeRepr::F32, shape.clone());
        let silu = graph.add_node(Op::Silu, vec![relu], DtypeRepr::F32, shape);

        let lowering = TritonLowering::default();
        let (dag, mapping, _) = lowering
            .lower_with_mapping(&graph, LoweringMode::Inference)
            .expect("lowering should not fail here");

        let (relu, silu) = (mapping[relu], mapping[silu]);
        let tile_graph = TileGraph::from_dag(&dag);

        let silu_output = tile_graph
            .output_edge_id(silu)
            .expect("silu is childless, so it has a boundary output edge");
        let requested_tile = vec![TileDim::Fixed(16)];
        let output_tiles = HashMap::from([(silu_output, requested_tile.clone())]);

        let config = tile_graph.propagate(&[mapping[input], relu, silu], &output_tiles);

        let relu_to_silu = edge_between(&tile_graph, relu, silu);
        let x_to_relu = edge_between(&tile_graph, mapping[input], relu);

        assert_eq!(config.get(relu_to_silu), Some(&requested_tile));
        assert_eq!(config.get(x_to_relu), Some(&requested_tile));
    }
}
