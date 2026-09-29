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

//! Tile-shape vocabulary for the v2 `TileGraph` (teenygrad-1nr.25).
//!
//! Ported from the v1 submodule deleted at `9b9f58b28` (`tile_graph/types.rs`),
//! with one deliberate change forced by v2's node/edge model.
//!
//! ## How a tile is keyed, and why it changed
//!
//! In v1 an `EdgeId` identified a *global* edge, held in an arena alongside its
//! producer and consumer, so a tile config could be `HashMap<EdgeId, _>`.
//!
//! In v2 there is no arena: edges live inside the nodes they touch
//! ([`super::node::Node::in_edges`]/`out_edges`), and an `EdgeId` is a *port
//! index* on one node rather than an identity of its own. `EdgeId(0)` means
//! something different on every node.
//!
//! So a tile is keyed by [`ValueId`] -- the producing node plus the output port
//! it came out of -- which is what actually identifies a value in v2. That is
//! also the pair v2 already uses to name a producer when wiring inputs
//! (`add_ir_node(name, &[(node, port)])`), so this introduces no new notion of
//! identity; it just names the one already in use.
//!
//! Keeping v1's arena alongside v2's embedded edges was the alternative, and is
//! rejected on purpose: two representations of the same wiring is exactly the
//! sort of duplicated state that has to be kept in sync by hand.

use std::collections::HashMap;

use teeny_core::device::hardware::MemoryLevelKind;
use teeny_core::graph::{DtypeRepr, Shape};

use super::node::{EdgeId, NodeId};

/// One dimension of a tile: either a known extent, or a symbol standing for an
/// extent the graph does not know yet (a dynamic batch, say).
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum TileDim {
    Fixed(usize),
    Sym(String),
}

/// A tile's shape, one [`TileDim`] per dimension.
pub type TileShape = Vec<TileDim>;

/// Identifies a value in the graph: the node that produced it, and which of
/// that node's output ports it left by. See the module doc for why tiles are
/// keyed by this rather than by an `EdgeId` alone.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ValueId {
    pub node: NodeId,
    pub port: EdgeId,
}

impl ValueId {
    pub fn new(node: NodeId, port: EdgeId) -> Self {
        Self { node, port }
    }
}

impl From<(NodeId, EdgeId)> for ValueId {
    fn from((node, port): (NodeId, EdgeId)) -> Self {
        Self::new(node, port)
    }
}

/// Turns a graph shape into a tile shape, naming each unknown dimension after
/// the node and axis it belongs to so that two unknowns from different places
/// are never accidentally treated as the same free variable.
pub(super) fn to_tile_shape(node: NodeId, shape: &Shape) -> TileShape {
    shape
        .iter()
        .enumerate()
        .map(|(axis, dim)| match dim {
            Some(extent) => TileDim::Fixed(*extent),
            None => TileDim::Sym(format!("n{}d{axis}", node.0)),
        })
        .collect()
}

/// Bytes one tile of `shape` occupies at `dtype`.
///
/// A symbolic dimension counts as 1, exactly as in v1: an unresolved extent
/// contributes no known elements, so this is a lower bound rather than a
/// guess. `teenygrad-1nr.27` is where that under-count is addressed, and
/// `teenygrad-1nr.26` is where a `Sym` gains an optional maximum -- both
/// deliberately out of scope for this revival, which is a port.
pub fn shape_byte_size(shape: &TileShape, dtype: DtypeRepr) -> u64 {
    let elements: u64 = shape
        .iter()
        .map(|dim| match dim {
            TileDim::Fixed(extent) => *extent as u64,
            TileDim::Sym(_) => 1,
        })
        .product();
    elements * dtype_bytes(dtype)
}

fn dtype_bytes(dtype: DtypeRepr) -> u64 {
    match dtype {
        DtypeRepr::Bool | DtypeRepr::I8 | DtypeRepr::U8 => 1,
        DtypeRepr::I16 | DtypeRepr::U16 | DtypeRepr::F16 | DtypeRepr::BF16 => 2,
        DtypeRepr::I32 | DtypeRepr::U32 | DtypeRepr::F32 => 4,
        DtypeRepr::I64 | DtypeRepr::U64 | DtypeRepr::F64 => 8,
    }
}

/// A tile assignment: which shape each value carries, and at which level of the
/// memory hierarchy it is held.
///
/// v1 kept the memory level on the edge record itself. Here it sits beside the
/// shape in the config: v2's [`super::node::Edge`] is a `Copy` 4-tuple used as a
/// map key and carries no payload, and a level is a property of a *scheduling
/// decision* rather than of the wiring, so it belongs with the decision.
#[derive(Debug, Clone, Default)]
pub struct TileConfig {
    tiles: HashMap<ValueId, TileShape>,
    levels: HashMap<ValueId, MemoryLevelKind>,
}

impl TileConfig {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn get(&self, value: ValueId) -> Option<&TileShape> {
        self.tiles.get(&value)
    }

    pub fn level(&self, value: ValueId) -> Option<MemoryLevelKind> {
        self.levels.get(&value).copied()
    }

    /// Records `shape` for `value`, returning the shape it replaced.
    pub fn set(&mut self, value: ValueId, shape: TileShape) -> Option<TileShape> {
        self.tiles.insert(value, shape)
    }

    pub fn set_level(&mut self, value: ValueId, level: MemoryLevelKind) {
        self.levels.insert(value, level);
    }

    pub fn len(&self) -> usize {
        self.tiles.len()
    }

    pub fn is_empty(&self) -> bool {
        self.tiles.is_empty()
    }

    /// Every assigned value, in a deterministic order so that callers and tests
    /// do not depend on `HashMap` iteration order.
    pub fn values(&self) -> Vec<ValueId> {
        let mut out: Vec<ValueId> = self.tiles.keys().copied().collect();
        out.sort();
        out
    }
}

/// One node group and the tiling chosen for it, plus the groups nested inside
/// it at the next memory level down.
#[derive(Debug, Clone)]
pub struct SubGraphTilingResult {
    pub nodes: Vec<NodeId>,
    pub config: TileConfig,
    pub children: Vec<SubGraphTilingResult>,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn v(node: usize, port: usize) -> ValueId {
        ValueId::new(NodeId(node), EdgeId(port))
    }

    /// The point of keying by `ValueId`: `EdgeId(0)` names a different value on
    /// every node, so a config keyed by port alone would collide across nodes.
    #[test]
    fn test_same_port_on_different_nodes_is_a_different_value() {
        let mut config = TileConfig::new();
        config.set(v(0, 0), vec![TileDim::Fixed(8)]);
        config.set(v(1, 0), vec![TileDim::Fixed(16)]);

        assert_eq!(config.len(), 2, "port 0 of node 0 is not port 0 of node 1");
        assert_eq!(config.get(v(0, 0)), Some(&vec![TileDim::Fixed(8)]));
        assert_eq!(config.get(v(1, 0)), Some(&vec![TileDim::Fixed(16)]));
        assert_eq!(config.get(v(2, 0)), None);
    }

    #[test]
    fn test_set_replaces_and_reports_the_previous_shape() {
        let mut config = TileConfig::new();
        assert_eq!(config.set(v(0, 0), vec![TileDim::Fixed(8)]), None);
        assert_eq!(
            config.set(v(0, 0), vec![TileDim::Fixed(16)]),
            Some(vec![TileDim::Fixed(8)]),
            "re-resolving a value reports what it displaced"
        );
        assert_eq!(config.len(), 1);
    }

    #[test]
    fn test_values_are_ordered_so_callers_do_not_see_hashmap_order() {
        let mut config = TileConfig::new();
        for (node, port) in [(2, 1), (0, 0), (1, 0), (0, 1)] {
            config.set(v(node, port), vec![TileDim::Fixed(1)]);
        }
        assert_eq!(
            config.values(),
            vec![v(0, 0), v(0, 1), v(1, 0), v(2, 1)],
            "sorted by node then port"
        );
    }

    #[test]
    fn test_a_level_is_recorded_separately_from_a_shape() {
        let mut config = TileConfig::new();
        let value = v(3, 0);
        assert_eq!(config.level(value), None);
        config.set_level(value, MemoryLevelKind::SharedMemory);
        assert_eq!(config.level(value), Some(MemoryLevelKind::SharedMemory));
        assert!(
            config.is_empty(),
            "a level alone is not a tile assignment, so len() still counts shapes"
        );
    }

    #[test]
    fn test_to_tile_shape_names_unknown_dims_per_node_and_axis() {
        let shape: Shape = vec![None, Some(32), None];
        assert_eq!(
            to_tile_shape(NodeId(7), &shape),
            vec![
                TileDim::Sym("n7d0".to_string()),
                TileDim::Fixed(32),
                TileDim::Sym("n7d2".to_string()),
            ]
        );
        assert_ne!(
            to_tile_shape(NodeId(7), &shape),
            to_tile_shape(NodeId(8), &shape),
            "two nodes' unknown dims are different free variables"
        );
    }

    #[test]
    fn test_byte_size_counts_a_symbolic_dim_as_one_element() {
        let known = vec![TileDim::Fixed(4), TileDim::Fixed(8)];
        assert_eq!(shape_byte_size(&known, DtypeRepr::F32), 4 * 8 * 4);

        let dynamic = vec![TileDim::Sym("n0d0".to_string()), TileDim::Fixed(8)];
        assert_eq!(
            shape_byte_size(&dynamic, DtypeRepr::F32),
            8 * 4,
            "an unresolved extent contributes one element, making this a lower bound \
             (see teenygrad-1nr.27)"
        );

        assert_eq!(shape_byte_size(&known, DtypeRepr::F64), 4 * 8 * 8);
        assert_eq!(shape_byte_size(&known, DtypeRepr::I8), 4 * 8);
        assert_eq!(shape_byte_size(&vec![], DtypeRepr::F32), 4, "a scalar tile");
    }
}
