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

//! Welder Fig. 7's `SubGraphTiling`: recursing the tile-config search up
//! through the memory hierarchy.

use std::collections::HashMap;

use teeny_core::device::hardware::{HardwareProfile, MemoryLevelKind};

use crate::errors::Result;

use super::types::{SubGraphTilingResult, TileConfig};
use super::{NodeId, TileGraph};

impl TileGraph {
    /// Welder Fig. 7's `SubGraphTiling(g, level, c)`: enumerates candidate
    /// output tiles for `root` via [`Self::enumerate_subtiles`] (bounded by
    /// `level`'s capacity in `hardware`), `propagate`s each into a
    /// [`TileConfig`], keeps the `top_k` lowest-`MemTraffic` ones, and
    /// recurses one memory level up on the (deduplicated) subgraphs
    /// [`Self::extract_subgraph`] finds there.
    ///
    /// Recursion terminates at the top of `hardware`'s declared memory
    /// hierarchy (Fig. 7's "return empty sub-graph at top level to exit
    /// recursion") — bounded by the number of distinct
    /// [`MemoryLevelKind`]s `hardware` declares, so this always halts.
    ///
    /// Deviates from the paper in one respect, noted in
    /// `TILE_GRAPH_SCHEDULING_PLAN.md`/teenygrad-1nr.4: this always
    /// re-derives each level's own candidates fresh from that level's own
    /// root, rather than threading the paper's `c` (the config chosen one
    /// level down) into the next level's candidate search — the paper's
    /// pseudocode doesn't specify enough about how `c` constrains
    /// `EnumerateSubtiles` to port literally, and even Welder's own
    /// shipped implementation (`policy/default.py`'s `emit_config`) calls
    /// its search once from a single base tile rather than implementing
    /// this literal recursive threading.
    pub fn sub_graph_tiling(
        &self,
        nodes: &[NodeId],
        root: NodeId,
        level: Option<MemoryLevelKind>,
        hardware: &HardwareProfile,
        top_k: usize,
    ) -> Result<Vec<SubGraphTilingResult>> {
        let capacity = level
            .and_then(|level| hardware.level(level))
            .map(|memory_level| memory_level.capacity)
            .unwrap_or(u64::MAX);

        let Some(output_edge) = self
            .output_edge_id(root)
            .or_else(|| self.children(root).first().map(|&(_, id)| id))
        else {
            return Ok(Vec::new());
        };

        let mut scored: Vec<(TileConfig, u64)> = Vec::new();
        for subtile in self.enumerate_subtiles(nodes, root, capacity)? {
            let mut seed = HashMap::new();
            seed.insert(output_edge, subtile);
            let config = self.propagate(nodes, &seed)?;
            if self.mem_footprint_with_config(nodes, &config) > capacity {
                continue;
            }
            let traffic = self.mem_traffic_with_config(nodes, &config);
            scored.push((config, traffic));
        }
        scored.sort_by_key(|&(_, traffic)| traffic);
        scored.truncate(top_k.max(1));

        let next_level = hardware.next_memory_level(level);

        scored
            .into_iter()
            .map(|(config, _)| -> Result<SubGraphTilingResult> {
                let children = match next_level {
                    None => Vec::new(),
                    Some(next_level) => {
                        let mut seen: Vec<Vec<NodeId>> = Vec::new();
                        let mut children = Vec::new();
                        for &node in nodes {
                            let subgraph = self.extract_subgraph(node, Some(next_level));
                            if seen.contains(&subgraph) {
                                continue;
                            }
                            seen.push(subgraph.clone());
                            children.extend(self.sub_graph_tiling(
                                &subgraph,
                                node,
                                Some(next_level),
                                hardware,
                                top_k,
                            )?);
                        }
                        children
                    }
                };
                Ok(SubGraphTilingResult {
                    nodes: nodes.to_vec(),
                    config,
                    children,
                })
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use teeny_core::device::hardware::{HardwareProfile, MemoryLevel, MemoryLevelKind};
    use teeny_core::model::ExecutableOp;
    use teeny_core::utils::dag::Dag;

    use super::super::testing::{flat_unary_spec, op, op_with_tile_spec};
    use super::super::{NodeId, SubGraphTilingResult, TileConfig, TileGraph};

    fn two_level_hardware(register_capacity: u64, device_capacity: u64) -> HardwareProfile {
        HardwareProfile {
            name: "test-device".to_string(),
            compute_units: 1,
            memory_levels: vec![
                MemoryLevel {
                    kind: MemoryLevelKind::Register,
                    capacity: register_capacity,
                    bandwidth: None,
                    latency: None,
                },
                MemoryLevel {
                    kind: MemoryLevelKind::DeviceMemory,
                    capacity: device_capacity,
                    bandwidth: None,
                    latency: None,
                },
            ],
            execution: None,
        }
    }

    #[test]
    fn sub_graph_tiling_returns_configs_that_fit_the_level_capacity() {
        let full_shape = vec![Some(64)];
        let mut dag: Dag<Box<dyn ExecutableOp>> = Dag::new();
        let a = dag.add_node(op("a", full_shape.clone(), true));
        let b = dag.add_node(op_with_tile_spec("b", full_shape, false, flat_unary_spec()));
        dag.add_edge(a, b);

        let tile_graph = TileGraph::from_dag(&dag);
        // a has no tile_spec, so a's own input boundary edge (64 * 4B =
        // 256B) is a fixed floor on every candidate's footprint,
        // regardless of the chosen tile -- capacity must clear it.
        let hardware = two_level_hardware(2000, u64::MAX);

        let results =
            tile_graph.sub_graph_tiling(&[a, b], b, Some(MemoryLevelKind::Register), &hardware, 5).unwrap();

        assert!(!results.is_empty());
        for result in &results {
            let footprint = tile_graph.mem_footprint_with_config(&[a, b], &result.config);
            assert!(
                footprint <= 2000,
                "footprint {footprint} exceeds capacity 2000"
            );
        }
    }

    #[test]
    fn sub_graph_tiling_has_no_children_at_the_top_of_the_hierarchy() {
        let mut dag: Dag<Box<dyn ExecutableOp>> = Dag::new();
        let a = dag.add_node(op("a", vec![Some(64)], true));

        let tile_graph = TileGraph::from_dag(&dag);
        let hardware = HardwareProfile {
            name: "single-level".to_string(),
            compute_units: 1,
            memory_levels: vec![MemoryLevel {
                kind: MemoryLevelKind::DeviceMemory,
                capacity: u64::MAX,
                bandwidth: None,
                latency: None,
            }],
            execution: None,
        };

        let results =
            tile_graph.sub_graph_tiling(&[a], a, Some(MemoryLevelKind::DeviceMemory), &hardware, 3).unwrap();

        assert!(!results.is_empty());
        for result in &results {
            assert!(result.children.is_empty());
        }
    }

    #[test]
    fn sub_graph_tiling_recurses_one_level_when_a_higher_level_exists() {
        let mut dag: Dag<Box<dyn ExecutableOp>> = Dag::new();
        let a = dag.add_node(op("a", vec![Some(64)], true));

        let tile_graph = TileGraph::from_dag(&dag);
        let hardware = two_level_hardware(u64::MAX, u64::MAX);

        let results =
            tile_graph.sub_graph_tiling(&[a], a, Some(MemoryLevelKind::Register), &hardware, 3).unwrap();

        assert!(!results.is_empty());
        for result in &results {
            assert!(
                !result.children.is_empty(),
                "expected recursion into DeviceMemory"
            );
            for child in &result.children {
                assert!(
                    child.children.is_empty(),
                    "DeviceMemory is the top declared level, recursion should stop there"
                );
            }
        }
    }

    #[test]
    fn sub_graph_tiling_of_none_recurses_into_the_hardwares_lowest_declared_level() {
        // Unlike `two_level_hardware` (which declares Register as a real
        // level, exercised above), real Triton hardware profiles never
        // declare Register at all -- only SharedMemory/DeviceMemory
        // (e.g. `orin_nano_hardware_profile`). `level: None` ("nothing
        // decided yet") must still correctly recurse into that lowest
        // *declared* level as its first real child, not skip it or
        // require a Register entry to exist.
        let mut dag: Dag<Box<dyn ExecutableOp>> = Dag::new();
        let a = dag.add_node(op("a", vec![Some(64)], true));

        let tile_graph = TileGraph::from_dag(&dag);
        let hardware = HardwareProfile {
            name: "shared-and-device".to_string(),
            compute_units: 1,
            memory_levels: vec![
                MemoryLevel {
                    kind: MemoryLevelKind::SharedMemory,
                    capacity: u64::MAX,
                    bandwidth: None,
                    latency: None,
                },
                MemoryLevel {
                    kind: MemoryLevelKind::DeviceMemory,
                    capacity: u64::MAX,
                    bandwidth: None,
                    latency: None,
                },
            ],
            execution: None,
        };

        let results = tile_graph.sub_graph_tiling(&[a], a, None, &hardware, 3).unwrap();

        assert!(!results.is_empty());
        for result in &results {
            assert!(
                !result.children.is_empty(),
                "expected recursion into SharedMemory, the hardware's lowest declared level"
            );
            for shared_memory_child in &result.children {
                assert!(
                    !shared_memory_child.children.is_empty(),
                    "expected recursion from SharedMemory into DeviceMemory"
                );
                for device_memory_child in &shared_memory_child.children {
                    assert!(
                        device_memory_child.children.is_empty(),
                        "DeviceMemory is the top declared level, recursion should stop there"
                    );
                }
            }
        }
    }

    #[test]
    fn sub_graph_tiling_respects_top_k() {
        let mut dag: Dag<Box<dyn ExecutableOp>> = Dag::new();
        let a = dag.add_node(op("a", vec![Some(64)], true));

        let tile_graph = TileGraph::from_dag(&dag);
        let hardware = two_level_hardware(u64::MAX, u64::MAX);

        let top_1 =
            tile_graph.sub_graph_tiling(&[a], a, Some(MemoryLevelKind::Register), &hardware, 1).unwrap();
        let top_3 =
            tile_graph.sub_graph_tiling(&[a], a, Some(MemoryLevelKind::Register), &hardware, 3).unwrap();

        assert_eq!(top_1.len(), 1);
        assert_eq!(top_3.len(), 3);
    }

    #[test]
    fn sub_graph_tiling_results_carry_the_node_set_they_cover() {
        let mut dag: Dag<Box<dyn ExecutableOp>> = Dag::new();
        let a = dag.add_node(op("a", vec![Some(64)], true));
        let b = dag.add_node(op("b", vec![Some(64)], false));
        dag.add_edge(a, b);

        let tile_graph = TileGraph::from_dag(&dag);
        let hardware = two_level_hardware(u64::MAX, u64::MAX);

        let results =
            tile_graph.sub_graph_tiling(&[a, b], b, Some(MemoryLevelKind::Register), &hardware, 1).unwrap();

        assert!(!results.is_empty());
        for result in &results {
            assert_eq!(result.nodes, vec![a, b]);
            // a -> b is left at from_dag's default (DeviceMemory), so
            // extract_subgraph(_, DeviceMemory)'s strict "> level" test
            // doesn't qualify it -- each child stays a singleton, one per
            // node, rather than merging into one [a, b] child.
            let mut all_child_nodes: Vec<NodeId> = result
                .children
                .iter()
                .flat_map(|c| c.nodes.clone())
                .collect();
            all_child_nodes.sort_unstable();
            assert_eq!(all_child_nodes, vec![a, b]);
            for child in &result.children {
                assert_eq!(child.nodes.len(), 1);
            }
        }
    }

    #[test]
    fn resolved_tiling_round_trips_through_record_and_get() {
        let mut dag: Dag<Box<dyn ExecutableOp>> = Dag::new();
        let a = dag.add_node(op("a", vec![Some(4)], true));
        let b = dag.add_node(op("b", vec![Some(4)], false));
        dag.add_edge(a, b);

        let mut tile_graph = TileGraph::from_dag(&dag);
        let edge_id = tile_graph.children(a)[0].1;
        assert!(tile_graph.resolved_tiling(edge_id).is_none());

        let result = SubGraphTilingResult {
            nodes: vec![a, b],
            config: TileConfig::default(),
            children: Vec::new(),
        };
        tile_graph.record_resolved_tiling(edge_id, result);

        let recorded = tile_graph
            .resolved_tiling(edge_id)
            .expect("just recorded a result for this edge");
        assert_eq!(recorded.nodes, vec![a, b]);
    }
}
