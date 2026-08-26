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

    use teeny_core::model::{ExecutableOp, KernelTileSpec, TensorTileSpec, TileAxisBinding};
    use teeny_core::utils::dag::Dag;

    use super::super::testing::{batchnorm2d_shaped_spec, flat_unary_spec, op, op_with_tile_spec};
    use super::super::{TileDim, TileGraph};

    /// GEMM-shaped: `a_ptr: [M, K]`, `b_ptr: [K, N]`, `c_ptr: [M, N]` —
    /// mirrors the real `MATMUL_TILE_SPEC` in `graph/mod.rs`.
    fn gemm_shaped_spec() -> KernelTileSpec {
        const A: TensorTileSpec = TensorTileSpec {
            param: "a_ptr",
            rank: 2,
            axes: &[
                TileAxisBinding {
                    dims: &[0],
                    block_const: "BLOCK_M",
                    extent_param: "M",
                    window: None,
                    divide_by: None,
                },
                TileAxisBinding {
                    dims: &[1],
                    block_const: "BLOCK_K",
                    extent_param: "K",
                    window: None,
                    divide_by: None,
                },
            ],
            reduction_axis: Some(1),
            untiled_dims: &[],
        };
        const B: TensorTileSpec = TensorTileSpec {
            param: "b_ptr",
            rank: 2,
            axes: &[
                TileAxisBinding {
                    dims: &[0],
                    block_const: "BLOCK_K",
                    extent_param: "K",
                    window: None,
                    divide_by: None,
                },
                TileAxisBinding {
                    dims: &[1],
                    block_const: "BLOCK_N",
                    extent_param: "N",
                    window: None,
                    divide_by: None,
                },
            ],
            reduction_axis: Some(0),
            untiled_dims: &[],
        };
        const C: TensorTileSpec = TensorTileSpec {
            param: "c_ptr",
            rank: 2,
            axes: &[
                TileAxisBinding {
                    dims: &[0],
                    block_const: "BLOCK_M",
                    extent_param: "M",
                    window: None,
                    divide_by: None,
                },
                TileAxisBinding {
                    dims: &[1],
                    block_const: "BLOCK_N",
                    extent_param: "N",
                    window: None,
                    divide_by: None,
                },
            ],
            reduction_axis: None,
            untiled_dims: &[],
        };
        KernelTileSpec {
            inputs: &[A, B],
            outputs: &[C],
        }
    }

    #[test]
    fn propagate_resolves_flat_elementwise_identity() {
        // input(a) -> relu(b), b declares flat_unary_spec. Seeding b's
        // output tile at 500 (smaller than its full 1000) must resolve the
        // a -> b edge to that same 500, via the shared "n_elements" name.
        let full_shape = vec![Some(1000)];
        let mut dag: Dag<Box<dyn ExecutableOp>> = Dag::new();
        let a = dag.add_node(op("a", full_shape.clone(), true));
        let b = dag.add_node(op_with_tile_spec("b", full_shape, false, flat_unary_spec()));
        dag.add_edge(a, b);

        let tile_graph = TileGraph::from_dag(&dag);
        let ab_edge = tile_graph.children(a)[0].1;
        let b_output_edge = tile_graph
            .output_edge_id(b)
            .expect("b has no consumer in dag");

        let mut seed = HashMap::new();
        seed.insert(b_output_edge, vec![TileDim::Fixed(500)]);
        let config = tile_graph.propagate(&[a, b], &seed);

        assert_eq!(config.get(ab_edge), Some(&vec![TileDim::Fixed(500)]));
    }

    #[test]
    fn propagate_leaves_the_reduction_axis_at_its_full_extent() {
        // a: [M=256, K=96] -> c; b: [K=96, N=128] -> c; c = matmul(a, b).
        // Seeding c's output tile at [M=64, N=32] must resolve M on a and N
        // on b, while K (no output-side counterpart) falls back to its own
        // full extent (96) on both — not derived from M, N, or each other.
        let mut dag: Dag<Box<dyn ExecutableOp>> = Dag::new();
        let a = dag.add_node(op("a", vec![Some(256), Some(96)], true));
        let b = dag.add_node(op("b", vec![Some(96), Some(128)], true));
        let c = dag.add_node(op_with_tile_spec(
            "c",
            vec![Some(256), Some(128)],
            false,
            gemm_shaped_spec(),
        ));
        dag.add_edge(a, c);
        dag.add_edge(b, c);

        let tile_graph = TileGraph::from_dag(&dag);
        let ac_edge = tile_graph.children(a)[0].1;
        let bc_edge = tile_graph.children(b)[0].1;
        let c_output_edge = tile_graph
            .output_edge_id(c)
            .expect("c has no consumer in dag");

        let mut seed = HashMap::new();
        seed.insert(c_output_edge, vec![TileDim::Fixed(64), TileDim::Fixed(32)]);
        let config = tile_graph.propagate(&[a, b, c], &seed);

        assert_eq!(
            config.get(ac_edge),
            Some(&vec![TileDim::Fixed(64), TileDim::Fixed(96)])
        );
        assert_eq!(
            config.get(bc_edge),
            Some(&vec![TileDim::Fixed(96), TileDim::Fixed(32)])
        );
    }

    #[test]
    fn propagate_stops_at_a_node_with_no_declared_tile_spec() {
        // a -> b, but b has no tile_spec: propagate must not guess a's
        // required tile shape from b's seeded output tile.
        let shape = vec![Some(1000)];
        let mut dag: Dag<Box<dyn ExecutableOp>> = Dag::new();
        let a = dag.add_node(op("a", shape.clone(), true));
        let b = dag.add_node(op("b", shape, false));
        dag.add_edge(a, b);

        let tile_graph = TileGraph::from_dag(&dag);
        let ab_edge = tile_graph.children(a)[0].1;
        let b_output_edge = tile_graph.output_edge_id(b).unwrap();

        let mut seed = HashMap::new();
        seed.insert(b_output_edge, vec![TileDim::Fixed(500)]);
        let config = tile_graph.propagate(&[a, b], &seed);

        assert_eq!(config.get(b_output_edge), Some(&vec![TileDim::Fixed(500)]));
        assert_eq!(config.get(ab_edge), None);
    }

    /// A conv2d-style spec: only the width axis (dim 0 here) is genuinely
    /// block-tiled (`BLOCK_OW`/`"OW"`, mirroring the real
    /// `conv2d_forward` kernel's only per-axis block const); the other
    /// real dimension (channels) is grid-driven with no block-size
    /// generic of its own, so it's named in `untiled_dims` instead of
    /// getting a `TileAxisBinding` -- exactly the case `untiled_dims`'s
    /// own doc comment in `teeny_core::model::tile_spec` describes as its
    /// reason to exist. `rank` (2) still reflects the tensor's real,
    /// full rank; only `axes` (1 entry) is partial.
    fn partially_tiled_spec() -> KernelTileSpec {
        const WIDTH: TileAxisBinding = TileAxisBinding {
            dims: &[0],
            block_const: "BLOCK_OW",
            extent_param: "OW",
            window: None,
            divide_by: None,
        };
        const X: TensorTileSpec = TensorTileSpec {
            param: "x_ptr",
            rank: 2,
            axes: &[WIDTH],
            reduction_axis: None,
            untiled_dims: &["C"],
        };
        const Y: TensorTileSpec = TensorTileSpec {
            param: "y_ptr",
            ..X
        };
        KernelTileSpec {
            inputs: &[X],
            outputs: &[Y],
        }
    }

    #[test]
    fn propagate_of_a_partially_tiled_spec_should_not_drop_the_untiled_dims() {
        // a -> b, b declares partially_tiled_spec: axes covers only its
        // width dim (dim 0); channels (dim 1) is named in untiled_dims
        // instead, per that field's documented purpose. propagate must
        // resolve a full rank-2 tile for the a->b edge -- channels at its
        // real full extent (8), width at whatever b's seeded output tile
        // requests (16) -- not silently drop the untiled channels
        // dimension. Regression test for teenygrad-1nr.7: propagate used
        // to build `resolved`/`input_tile` by positionally zipping only
        // `axes` (len 1) against the real rank-2 output tile/edge shape,
        // producing a length-1 TileEdgeShape for a rank-2 edge.
        let full_shape = vec![Some(64), Some(8)]; // [OW, C]
        let mut dag: Dag<Box<dyn ExecutableOp>> = Dag::new();
        let a = dag.add_node(op("a", full_shape.clone(), true));
        let b = dag.add_node(op_with_tile_spec(
            "b",
            full_shape.clone(),
            false,
            partially_tiled_spec(),
        ));
        dag.add_edge(a, b);

        let tile_graph = TileGraph::from_dag(&dag);
        let ab_edge = tile_graph.children(a)[0].1;
        let b_output_edge = tile_graph
            .output_edge_id(b)
            .expect("b has no consumer in dag");

        let mut seed = HashMap::new();
        seed.insert(b_output_edge, vec![TileDim::Fixed(16), TileDim::Fixed(8)]);
        let config = tile_graph.propagate(&[a, b], &seed);

        let resolved = config
            .get(ab_edge)
            .expect("propagate should have resolved the a->b edge");
        assert_eq!(
            resolved.len(),
            full_shape.len(),
            "expected a rank-{}-consistent tile for the a->b edge (width \
             16, channels at its full extent 8), got a rank-{} shape \
             instead: {resolved:?} -- propagate silently dropped \
             untiled_dims' channels dimension",
            full_shape.len(),
            resolved.len(),
        );
        assert_eq!(
            resolved,
            &vec![TileDim::Fixed(16), TileDim::Fixed(8)],
            "width should resolve to b's seeded 16, channels should fall \
             back to its full extent 8 -- got {resolved:?}"
        );
    }

    #[test]
    fn propagate_of_a_flattened_multi_dim_axis_collapses_outer_dims_to_one() {
        // a -> b, b declares batchnorm2d_shaped_spec. Seeding b's output
        // tile with W (the innermost of the flattened [H, W] pair) at 24
        // must resolve the a -> b edge to B/C at their full extent (2, 4
        // -- untiled), H collapsed to 1 (the flattened-away outer dim),
        // W at the resolved 24 -- not literal H*W=24 spread back across
        // both axes (there's no way to invert a flat block size into
        // separate per-axis extents in general), but product-preserving:
        // 2*4*1*24 matches what a real BLOCK_HW=24 tile's element count
        // would be, times B/C's own untiled extents.
        let full_shape = vec![Some(2), Some(4), Some(16), Some(32)]; // [B, C, H, W]
        let mut dag: Dag<Box<dyn ExecutableOp>> = Dag::new();
        let a = dag.add_node(op("a", full_shape.clone(), true));
        let b = dag.add_node(op_with_tile_spec(
            "b",
            full_shape.clone(),
            false,
            batchnorm2d_shaped_spec(),
        ));
        dag.add_edge(a, b);

        let tile_graph = TileGraph::from_dag(&dag);
        let ab_edge = tile_graph.children(a)[0].1;
        let b_output_edge = tile_graph
            .output_edge_id(b)
            .expect("b has no consumer in dag");

        let mut seed = HashMap::new();
        seed.insert(
            b_output_edge,
            vec![
                TileDim::Fixed(2),
                TileDim::Fixed(4),
                TileDim::Fixed(16),
                TileDim::Fixed(24),
            ],
        );
        let config = tile_graph.propagate(&[a, b], &seed);

        let resolved = config
            .get(ab_edge)
            .expect("propagate should have resolved the a->b edge");
        assert_eq!(
            resolved,
            &vec![
                TileDim::Fixed(2),  // B: untiled, full extent
                TileDim::Fixed(4),  // C: untiled, full extent
                TileDim::Fixed(1),  // H: flattened-away outer dim
                TileDim::Fixed(24), // W: the resolved HW value
            ],
            "expected B/C at full extent, H collapsed to 1, W at the \
             resolved HW value -- got {resolved:?}"
        );
    }

    /// A groupnorm-style spec: mirrors the real `group_norm_forward`
    /// kernel's tiling shape (grid `[N*G]`, one CTA per (sample, group),
    /// iterating `BLOCK_NL`-wide tiles over `channels_per_group * L`
    /// where `channels_per_group = C / G`) -- teenygrad-1nr.10. Only `L`
    /// (dim 2) gets a `TileAxisBinding`; the channel axis (dim 1) is
    /// deliberately left out of `axes` entirely -- the best available
    /// authoring choice, since `TensorTileSpec` has no way to say "this
    /// axis's real per-tile extent is `C` divided by the compile-time
    /// constant `G`."
    fn groupnorm_shaped_spec() -> KernelTileSpec {
        const L_AXIS: TileAxisBinding = TileAxisBinding {
            dims: &[2],
            block_const: "BLOCK_NL",
            extent_param: "group_size",
            window: None,
            divide_by: None,
        };
        // channels_per_group = C / G (teenygrad-1nr.10): its own axis,
        // never resolved via name-matching (nothing else names
        // "channels_per_group"), so its value always comes from the
        // divide_by fallback -- both here (from the seeded output tile)
        // and on the input side.
        const C_AXIS: TileAxisBinding = TileAxisBinding {
            dims: &[1],
            block_const: "",
            extent_param: "channels_per_group",
            window: None,
            divide_by: Some(2), // G
        };
        const X: TensorTileSpec = TensorTileSpec {
            param: "x_ptr",
            rank: 3,
            axes: &[L_AXIS, C_AXIS],
            reduction_axis: None,
            untiled_dims: &["N"],
        };
        const Y: TensorTileSpec = TensorTileSpec {
            param: "y_ptr",
            ..X
        };
        KernelTileSpec {
            inputs: &[X],
            outputs: &[Y],
        }
    }

    #[test]
    fn propagate_resolves_a_dim_subdivided_by_a_grid_constant() {
        // Fixed for teenygrad-1nr.10: group_norm_forward's real per-CTA
        // tile spans channels_per_group (= C/G) * L -- only a fraction
        // of the channel axis, not the whole of it. `TileAxisBinding::divide_by`
        // now lets a spec say so directly: with N=2, C=8, L=16, G=2
        // (channels_per_group=4), the channel axis must resolve to 4,
        // not C's full extent 8.
        let full_shape = vec![Some(2), Some(8), Some(16)]; // [N, C, L]
        let mut dag: Dag<Box<dyn ExecutableOp>> = Dag::new();
        let a = dag.add_node(op("a", full_shape.clone(), true));
        let b = dag.add_node(op_with_tile_spec(
            "b",
            full_shape.clone(),
            false,
            groupnorm_shaped_spec(),
        ));
        dag.add_edge(a, b);

        let tile_graph = TileGraph::from_dag(&dag);
        let ab_edge = tile_graph.children(a)[0].1;
        let b_output_edge = tile_graph
            .output_edge_id(b)
            .expect("b has no consumer in dag");

        let mut seed = HashMap::new();
        seed.insert(
            b_output_edge,
            vec![TileDim::Fixed(2), TileDim::Fixed(8), TileDim::Fixed(16)],
        );
        let config = tile_graph.propagate(&[a, b], &seed);

        let resolved = config
            .get(ab_edge)
            .expect("propagate should have resolved the a->b edge");
        const CHANNELS_PER_GROUP: usize = 8 / 2; // C / G
        assert_eq!(
            resolved[1],
            TileDim::Fixed(CHANNELS_PER_GROUP),
            "expected the channel axis to resolve to channels_per_group \
             ({CHANNELS_PER_GROUP}), matching group_norm_forward's real \
             per-CTA tile -- got {:?}",
            resolved[1]
        );
    }

    /// Mirrors `groupnorm_shaped_spec`, except the output side never
    /// declares the channels axis at all (only `L` is tiled on `y_ptr`).
    /// Used to exercise `propagate`'s *input-side* `divide_by` fallback
    /// directly (teenygrad-1nr.10): `groupnorm_shaped_spec`'s `Y` happens
    /// to also declare `C_AXIS` (via `..X`), so its own test resolves
    /// `"channels_per_group"` on the output side first, never actually
    /// reaching the input-side fallback branch.
    fn groupnorm_input_only_divide_by_spec() -> KernelTileSpec {
        const L_AXIS: TileAxisBinding = TileAxisBinding {
            dims: &[2],
            block_const: "BLOCK_NL",
            extent_param: "group_size",
            window: None,
            divide_by: None,
        };
        const C_AXIS: TileAxisBinding = TileAxisBinding {
            dims: &[1],
            block_const: "",
            extent_param: "channels_per_group",
            window: None,
            divide_by: Some(2), // G
        };
        const X: TensorTileSpec = TensorTileSpec {
            param: "x_ptr",
            rank: 3,
            axes: &[L_AXIS, C_AXIS],
            reduction_axis: None,
            untiled_dims: &["N"],
        };
        const Y: TensorTileSpec = TensorTileSpec {
            param: "y_ptr",
            rank: 3,
            axes: &[L_AXIS], // channels not declared here at all
            reduction_axis: None,
            untiled_dims: &["N", "C"],
        };
        KernelTileSpec {
            inputs: &[X],
            outputs: &[Y],
        }
    }

    #[test]
    fn propagate_applies_divide_by_on_the_input_side_fallback_when_never_resolved_via_output() {
        // Companion to propagate_resolves_a_dim_subdivided_by_a_grid_constant:
        // that test's spec happens to also declare the divide_by axis on
        // the output side, so "channels_per_group" gets resolved (and
        // divided) there first. This spec's output never declares it at
        // all, so the input-side fallback branch is what has to apply
        // divide_by directly to the raw full extent.
        let full_shape = vec![Some(2), Some(8), Some(16)]; // [N, C, L]
        let mut dag: Dag<Box<dyn ExecutableOp>> = Dag::new();
        let a = dag.add_node(op("a", full_shape.clone(), true));
        let b = dag.add_node(op_with_tile_spec(
            "b",
            full_shape.clone(),
            false,
            groupnorm_input_only_divide_by_spec(),
        ));
        dag.add_edge(a, b);

        let tile_graph = TileGraph::from_dag(&dag);
        let ab_edge = tile_graph.children(a)[0].1;
        let b_output_edge = tile_graph
            .output_edge_id(b)
            .expect("b has no consumer in dag");

        let mut seed = HashMap::new();
        seed.insert(
            b_output_edge,
            vec![TileDim::Fixed(2), TileDim::Fixed(8), TileDim::Fixed(16)],
        );
        let config = tile_graph.propagate(&[a, b], &seed);

        let resolved = config
            .get(ab_edge)
            .expect("propagate should have resolved the a->b edge");
        assert_eq!(
            resolved[1],
            TileDim::Fixed(4), // C / G = 8 / 2
            "expected the channel axis to fall back to channels_per_group \
             via divide_by, since nothing resolves it via name-matching \
             -- got {:?}",
            resolved[1]
        );
    }

    /// Simulates a two-output kernel (mirrors `flash_attn2`'s real
    /// `o_ptr` + `l_ptr` outputs, and `group_norm_forward`'s three --
    /// teenygrad-1nr.11): `outputs[0]` (`Y1`) names one `extent_param`
    /// (`"M"`); `outputs[1]` (`Y2`) names a *different* one (`"aux"`)
    /// that only appears on one input axis.
    fn two_output_spec() -> KernelTileSpec {
        const X_AXIS: TileAxisBinding = TileAxisBinding {
            dims: &[0],
            block_const: "BLOCK_AUX",
            extent_param: "aux", // only Y2 names this -- Y1 never does
            window: None,
            divide_by: None,
        };
        const X: TensorTileSpec = TensorTileSpec {
            param: "x_ptr",
            rank: 1,
            axes: &[X_AXIS],
            reduction_axis: None,
            untiled_dims: &[],
        };
        const Y1_AXIS: TileAxisBinding = TileAxisBinding {
            dims: &[0],
            block_const: "BLOCK_M",
            extent_param: "M",
            window: None,
            divide_by: None,
        };
        const Y1: TensorTileSpec = TensorTileSpec {
            param: "y_ptr",
            rank: 1,
            axes: &[Y1_AXIS],
            reduction_axis: None,
            untiled_dims: &[],
        };
        const Y2_AXIS: TileAxisBinding = TileAxisBinding {
            dims: &[0],
            block_const: "BLOCK_AUX",
            extent_param: "aux",
            window: None,
            divide_by: None,
        };
        const Y2: TensorTileSpec = TensorTileSpec {
            param: "l_ptr",
            rank: 1,
            axes: &[Y2_AXIS],
            reduction_axis: None,
            untiled_dims: &[],
        };
        KernelTileSpec {
            inputs: &[X],
            outputs: &[Y1, Y2],
        }
    }

    #[test]
    fn propagate_resolves_a_second_declared_output_when_its_edge_is_seeded() {
        // Fixed for teenygrad-1nr.11: KernelTileSpec.outputs can declare
        // more than one real output tensor, motivated by flash_attn2's
        // real o_ptr + l_ptr (logsumexp) outputs and
        // group_norm_forward's y_ptr/mean_ptr/rstd_ptr. propagate used to
        // only ever read spec.outputs.first() -- an extent_param that
        // appears only on a second/later output was never seeded into
        // `resolved`, so an input axis bound to it always fell back to
        // its full extent, no matter what was seeded.
        //
        // TileGraph::from_dag now synthesizes one extra boundary edge per
        // additional declared output (TileGraph::secondary_output_edges)
        // -- seeding *that* edge (not b's primary output edge) lets
        // propagate resolve "aux" for real.
        let full_shape = vec![Some(1000)];
        let mut dag: Dag<Box<dyn ExecutableOp>> = Dag::new();
        let a = dag.add_node(op("a", full_shape.clone(), true));
        let b = dag.add_node(op_with_tile_spec("b", full_shape, false, two_output_spec()));
        dag.add_edge(a, b);

        let tile_graph = TileGraph::from_dag(&dag);
        let ab_edge = tile_graph.children(a)[0].1;
        let b_output_edge = tile_graph
            .output_edge_id(b)
            .expect("b has no consumer in dag");
        let b_aux_edge = *tile_graph
            .secondary_output_edges(b)
            .first()
            .expect("b's two-output spec should have synthesized one secondary output edge");

        let mut seed = HashMap::new();
        seed.insert(b_output_edge, vec![TileDim::Fixed(50)]); // Y1's "M"
        seed.insert(b_aux_edge, vec![TileDim::Fixed(20)]); // Y2's "aux"
        let config = tile_graph.propagate(&[a, b], &seed);

        let resolved = config
            .get(ab_edge)
            .expect("propagate should have resolved the a->b edge");
        assert_eq!(
            resolved,
            &vec![TileDim::Fixed(20)],
            "expected a's tile to resolve via Y2's seeded \"aux\" value \
             (20), not Y1's \"M\" (50, which a's axis isn't bound to) or \
             the full-extent fallback (1000) -- got {resolved:?}"
        );
    }

    #[test]
    fn propagate_leaves_an_unseeded_second_output_without_a_full_shape_fallback() {
        // Companion to the test above: unlike the primary output (which
        // falls back to the node's own full shape when nothing seeds
        // it), a second declared output has no ground-truth shape to
        // fall back to at all (ExecutableOp::output_shape is singular).
        // Leaving it unseeded must simply contribute nothing -- b's
        // primary-output resolution (and the whole node) still proceeds
        // normally, and a's "aux"-bound axis falls back to its own full
        // extent, exactly like an ordinary unresolved axis (e.g. a
        // reduction axis) always has.
        let full_shape = vec![Some(1000)];
        let mut dag: Dag<Box<dyn ExecutableOp>> = Dag::new();
        let a = dag.add_node(op("a", full_shape.clone(), true));
        let b = dag.add_node(op_with_tile_spec("b", full_shape, false, two_output_spec()));
        dag.add_edge(a, b);

        let tile_graph = TileGraph::from_dag(&dag);
        let ab_edge = tile_graph.children(a)[0].1;
        let b_output_edge = tile_graph
            .output_edge_id(b)
            .expect("b has no consumer in dag");

        let mut seed = HashMap::new();
        seed.insert(b_output_edge, vec![TileDim::Fixed(50)]); // Y1's "M" only
        let config = tile_graph.propagate(&[a, b], &seed);

        let resolved = config
            .get(ab_edge)
            .expect("propagate should have resolved the a->b edge");
        assert_eq!(
            resolved,
            &vec![TileDim::Fixed(1000)],
            "expected a's \"aux\"-bound axis to fall back to its own full \
             extent (1000) when Y2 is never seeded -- got {resolved:?}"
        );
    }

    #[test]
    fn propagate_resolves_a_fan_out_producers_two_edges_independently() {
        // a feeds both b and c, each with their own flat_unary_spec and
        // their own independently-seeded output tile. EdgeId-keying means
        // a's two outgoing edges must resolve to two different tile
        // shapes, with no merging between them.
        let full_shape = vec![Some(1000)];
        let mut dag: Dag<Box<dyn ExecutableOp>> = Dag::new();
        let a = dag.add_node(op("a", full_shape.clone(), true));
        let b = dag.add_node(op_with_tile_spec(
            "b",
            full_shape.clone(),
            false,
            flat_unary_spec(),
        ));
        dag.add_edge(a, b);
        let c = dag.add_node(op_with_tile_spec("c", full_shape, false, flat_unary_spec()));
        dag.add_edge(a, c);

        let tile_graph = TileGraph::from_dag(&dag);
        let a_children = tile_graph.children(a);
        let ab_edge = a_children.iter().find(|&&(n, _)| n == b).unwrap().1;
        let ac_edge = a_children.iter().find(|&&(n, _)| n == c).unwrap().1;
        let b_output_edge = tile_graph.output_edge_id(b).unwrap();
        let c_output_edge = tile_graph.output_edge_id(c).unwrap();

        let mut seed = HashMap::new();
        seed.insert(b_output_edge, vec![TileDim::Fixed(10)]);
        seed.insert(c_output_edge, vec![TileDim::Fixed(20)]);
        let config = tile_graph.propagate(&[a, b, c], &seed);

        assert_eq!(config.get(ab_edge), Some(&vec![TileDim::Fixed(10)]));
        assert_eq!(config.get(ac_edge), Some(&vec![TileDim::Fixed(20)]));
    }

    #[test]
    fn propagate_of_an_empty_node_set_returns_the_seed_unchanged() {
        let tile_graph = TileGraph::default();
        let dummy_edge = {
            // Build a throwaway single-node graph just to mint a valid
            // EdgeId to seed with -- propagate on an empty node set must
            // not touch it either way.
            let mut dag: Dag<Box<dyn ExecutableOp>> = Dag::new();
            dag.add_node(op("a", vec![Some(4)], true));
            let g = TileGraph::from_dag(&dag);
            g.output_edge_id(0).unwrap()
        };

        let mut seed = HashMap::new();
        seed.insert(dummy_edge, vec![TileDim::Fixed(4)]);
        let config = tile_graph.propagate(&[], &seed);

        assert_eq!(config.len(), 1);
        assert_eq!(config.get(dummy_edge), Some(&vec![TileDim::Fixed(4)]));
    }
}
