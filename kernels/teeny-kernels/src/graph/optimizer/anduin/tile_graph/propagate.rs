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

use crate::errors::{Error, Result};

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
    /// Degrades to a hard boundary — stops there, doesn't guess, and does
    /// *not* error — at any node with no declared `tile_spec` (the
    /// overwhelming majority; see [`teeny_core::model::KernelTileSpec`]'s
    /// doc comment on coverage being opt-in), or whose parent-edge count
    /// doesn't match its spec's declared input count (the same positional
    /// producer-to-spec correspondence limitation the design this revives
    /// already had — see that module's doc comment; e.g. `Add(x, x)`
    /// legitimately collapsing two operand slots into one deduped parent
    /// edge, see [`Self::parents`]'s doc comment).
    ///
    /// Everything else a `tile_spec` asserts about the graph that turns
    /// out false — a declared rank that disagrees with the real shape, an
    /// axis with an empty `dims` list, an output spec with no outputs at
    /// all — is a spec-authoring bug, not a shape this function can fall
    /// back from, so those return `Err` instead of silently skipping the
    /// node.
    pub fn propagate(
        &self,
        nodes: &[NodeId],
        output_tiles: &HashMap<EdgeId, TileEdgeShape>,
    ) -> Result<TileConfig> {
        let included: HashSet<NodeId> = nodes.iter().copied().collect();
        let mut tiles: HashMap<EdgeId, TileEdgeShape> = output_tiles.clone();

        for node in self.topological_sort().into_iter().rev() {
            if !included.contains(&node) {
                continue;
            }
            let Some(spec) = &self.node(node).tile_spec else {
                continue; // hard boundary: no declared spec
            };
            let op_name = || self.node(node).name.clone();
            let Some(output_spec) = spec.outputs.first() else {
                return Err(Error::TileSpecMissingOutput {
                    node,
                    op_name: op_name(),
                }
                .into());
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
                return Err(Error::OutputRankMismatch {
                    node,
                    op_name: op_name(),
                    output_index: 0,
                    expected: output_spec.rank,
                    actual: output_tile.len(),
                }
                .into());
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
            let mut resolved: HashMap<&'static str, TileDim> = HashMap::new();
            for axis in output_spec.axes.iter() {
                let Some(&innermost) = axis.dims.last() else {
                    return Err(Error::EmptyAxisDims {
                        node,
                        op_name: op_name(),
                        extent_param: axis.extent_param,
                    }
                    .into());
                };
                if let Some(dim) = output_tile.get(innermost) {
                    resolved.insert(axis.extent_param, apply_divide_by(dim, axis.divide_by));
                }
            }

            // Additional declared outputs (teenygrad-1nr.11): each one
            // whose synthesized boundary edge the caller has seeded
            // contributes more resolved extent_param -> TileDim entries,
            // on top of the primary output's above. Unlike the primary
            // output, there's no full-shape fallback for these -- an
            // *unseeded* secondary output simply contributes nothing,
            // rather than hard-boundarying the whole node the way a bad
            // primary output does above. A *seeded* one whose rank
            // disagrees with the spec, though, is the same kind of
            // spec-authoring bug as the primary output's rank check, so
            // it errors the same way.
            for (secondary_index, secondary_spec) in spec.outputs.iter().enumerate().skip(1) {
                let Some(&edge_id) = self.secondary_output_edges(node).get(secondary_index - 1)
                else {
                    continue; // no synthesized boundary edge for this declared output
                };
                let Some(tile) = tiles.get(&edge_id).cloned() else {
                    continue; // not seeded by the caller -- the common case
                };
                if tile.len() != secondary_spec.rank {
                    return Err(Error::OutputRankMismatch {
                        node,
                        op_name: op_name(),
                        output_index: secondary_index,
                        expected: secondary_spec.rank,
                        actual: tile.len(),
                    }
                    .into());
                }
                for axis in secondary_spec.axes.iter() {
                    let Some(&innermost) = axis.dims.last() else {
                        return Err(Error::EmptyAxisDims {
                            node,
                            op_name: op_name(),
                            extent_param: axis.extent_param,
                        }
                        .into());
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
            for (input_index, ((_, edge_id), input_spec)) in
                parents.iter().zip(spec.inputs.iter()).enumerate()
            {
                let full_shape = &self.edge(*edge_id).shape;
                if full_shape.len() != input_spec.rank {
                    return Err(Error::InputRankMismatch {
                        node,
                        op_name: op_name(),
                        input_index,
                        expected: input_spec.rank,
                        actual: full_shape.len(),
                    }
                    .into());
                }
                // Full rank, not `axes.len()`: a dim with no `TileAxisBinding`
                // (an untiled dim, or a reduction axis with no output-side
                // counterpart) keeps its full extent via this fallback,
                // rather than being dropped from the result entirely.
                let mut input_tile: TileEdgeShape = full_shape.clone();
                for axis in input_spec.axes.iter() {
                    let Some((&innermost, outer_dims)) = axis.dims.split_last() else {
                        return Err(Error::EmptyAxisDims {
                            node,
                            op_name: op_name(),
                            extent_param: axis.extent_param,
                        }
                        .into());
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

        Ok(TileConfig { tiles })
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use teeny_core::graph::{DtypeRepr, Graph, Op};
    use teeny_core::model::{
        ExecutableOp, KernelTileSpec, LoweringMode, TensorTileSpec, TileAxisBinding,
    };
    use teeny_core::utils::dag::Dag;

    use crate::errors::Error;
    use crate::graph::TritonLowering;

    use super::super::testing::{edge_between, op, op_with_tile_spec};
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

        let config = tile_graph
            .propagate(&[mapping[input], relu, silu], &output_tiles)
            .expect("relu and silu both have real, well-formed tile_specs");

        let relu_to_silu = edge_between(&tile_graph, relu, silu);
        let x_to_relu = edge_between(&tile_graph, mapping[input], relu);

        assert_eq!(config.get(relu_to_silu), Some(&requested_tile));
        assert_eq!(config.get(x_to_relu), Some(&requested_tile));
    }

    #[test]
    fn propagate_errors_when_output_rank_disagrees_with_the_real_shape() {
        // b's tile_spec claims a rank-2 output, but b has no consumer, so
        // its output_tile falls back to its real (rank-1) shape --
        // disagreeing with the spec's declared rank is a spec-authoring
        // bug, not a shape `propagate` can guess its way around.
        const BAD_OUTPUT: TensorTileSpec = TensorTileSpec {
            param: "y_ptr",
            rank: 2,
            axes: &[],
            reduction_axis: None,
            untiled_dims: &[],
        };
        let bad_spec = KernelTileSpec {
            inputs: &[],
            outputs: &[BAD_OUTPUT],
        };

        let mut dag: Dag<Box<dyn ExecutableOp>> = Dag::new();
        let b = dag.add_node(op_with_tile_spec("b", vec![Some(64)], true, bad_spec));
        let tile_graph = TileGraph::from_dag(&dag);

        let err = tile_graph
            .propagate(&[b], &HashMap::new())
            .expect_err("b's declared output rank (2) disagrees with its real shape (rank 1)");

        assert!(matches!(
            err.downcast_ref::<Error>(),
            Some(Error::OutputRankMismatch {
                node,
                output_index: 0,
                expected: 2,
                actual: 1,
                ..
            }) if *node == b
        ));
    }

    #[test]
    fn propagate_errors_when_input_rank_disagrees_with_the_real_shape() {
        // b's tile_spec claims a rank-2 input, but the real a -> b edge
        // (a's own output shape) is rank 1.
        const OUT: TensorTileSpec = TensorTileSpec {
            param: "y_ptr",
            rank: 1,
            axes: &[],
            reduction_axis: None,
            untiled_dims: &[],
        };
        const BAD_INPUT: TensorTileSpec = TensorTileSpec {
            param: "x_ptr",
            rank: 2,
            axes: &[],
            reduction_axis: None,
            untiled_dims: &[],
        };
        let bad_spec = KernelTileSpec {
            inputs: &[BAD_INPUT],
            outputs: &[OUT],
        };

        let mut dag: Dag<Box<dyn ExecutableOp>> = Dag::new();
        let a = dag.add_node(op("a", vec![Some(64)], true));
        let b = dag.add_node(op_with_tile_spec("b", vec![Some(64)], false, bad_spec));
        dag.add_edge(a, b);
        let tile_graph = TileGraph::from_dag(&dag);

        let err = tile_graph
            .propagate(&[a, b], &HashMap::new())
            .expect_err("b's declared input rank (2) disagrees with the a -> b edge's real rank (1)");

        assert!(matches!(
            err.downcast_ref::<Error>(),
            Some(Error::InputRankMismatch {
                node,
                input_index: 0,
                expected: 2,
                actual: 1,
                ..
            }) if *node == b
        ));
    }

    #[test]
    fn propagate_errors_when_an_axis_declares_empty_dims() {
        // TileAxisBinding::dims's own doc comment calls an empty `dims`
        // list a spec-authoring bug -- there's no real tensor axis for
        // the binding to resolve.
        const BAD_AXIS: TileAxisBinding = TileAxisBinding {
            dims: &[],
            block_const: "BLOCK_SIZE",
            extent_param: "n_elements",
            window: None,
            divide_by: None,
        };
        const OUT: TensorTileSpec = TensorTileSpec {
            param: "y_ptr",
            rank: 1,
            axes: &[BAD_AXIS],
            reduction_axis: None,
            untiled_dims: &[],
        };
        let bad_spec = KernelTileSpec {
            inputs: &[],
            outputs: &[OUT],
        };

        let mut dag: Dag<Box<dyn ExecutableOp>> = Dag::new();
        let b = dag.add_node(op_with_tile_spec("b", vec![Some(64)], true, bad_spec));
        let tile_graph = TileGraph::from_dag(&dag);

        let err = tile_graph
            .propagate(&[b], &HashMap::new())
            .expect_err("b's \"n_elements\" axis declares an empty dims list");

        assert!(matches!(
            err.downcast_ref::<Error>(),
            Some(Error::EmptyAxisDims {
                node,
                extent_param: "n_elements",
                ..
            }) if *node == b
        ));
    }
}
