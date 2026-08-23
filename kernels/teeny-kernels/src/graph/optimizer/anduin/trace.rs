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

//! [`Trace::trace_graph`] — Welder §3.3's `ExecuteGraph` (Fig. 8), renamed
//! and made an associated function of [`Trace`]: what this walk actually
//! does, in this codebase, is build a *trace* — the one and only way a
//! [`Trace`] gets created — not execute anything for real, so a free
//! `execute_graph`/`trace_graph` taking a generic
//! [`ExecuteDevice`](super::codegen::ExecuteDevice) claimed more genericity
//! than the walk ever used: nothing has ever driven it with anything but a
//! [`Trace`].
//!
//! `Trace::trace_graph` recurses through the memory hierarchy: allocate a
//! workspace *at this level*, load input tiles into it, then for each node
//! either compute it directly (if this is the top memory level) or recurse
//! into that node's own subgraph one level up — finally store results back
//! down. [`SubGraphTilingResult`] (teenygrad-1nr.4) is already exactly this
//! recursive shape (a config at this level plus recursively-tiled
//! children), so `trace_graph` just walks it; there's no separate
//! "execution plan" type to build.
//!
//! The resulting trace is deliberately the intended output — not just a
//! test double. It's the input [`codegen`](super::codegen::codegen)
//! replays through a *different* [`ExecuteDevice`](super::codegen::ExecuteDevice)
//! — [`DagCodegen`](super::codegen::DagCodegen), still a stub — to
//! actually build a `Dag` of custom ops. `codegen`'s replay direction is
//! where genericity over `ExecuteDevice` actually matters (any device can
//! consume a trace); building one only ever produces a `Trace`. See
//! `codegen`'s module doc comment for the full picture.

use teeny_core::device::hardware::{HardwareProfile, MemoryLevelKind};

use super::codegen::ExecuteDevice;
use super::tile_graph::{NodeId, SubGraphTilingResult, TileGraph};

/// One call [`Trace`] recorded.
#[derive(Debug, Clone, PartialEq)]
pub enum TraceEvent {
    /// Welder §3.1/Fig. 5's *virtual node*: the original `NodeId`s
    /// consolidated into one fused unit as viewed from `level` — see
    /// [`ExecuteDevice::virtual_node`](super::codegen::ExecuteDevice::virtual_node).
    VirtualNode {
        nodes: Vec<NodeId>,
        level: MemoryLevelKind,
    },
    Allocate {
        footprint: u64,
        level: MemoryLevelKind,
    },
    LoadTiles {
        nodes: Vec<NodeId>,
        level: MemoryLevelKind,
    },
    ComputeTile {
        node: NodeId,
    },
    StoreTiles {
        nodes: Vec<NodeId>,
        level: MemoryLevelKind,
    },
}

/// A recorded trace of [`ExecuteDevice`] calls — a structural stand-in for
/// actually executing a scheduled tile-graph. The only way to build one is
/// [`Trace::trace_graph`]; until a real device exists, a `Trace` is also
/// the handoff point to a future codegen pass (`codegen`'s module doc
/// comment).
#[derive(Debug, Default)]
pub struct Trace {
    pub events: Vec<TraceEvent>,
}

impl Trace {
    /// Welder §3.3's `ExecuteGraph` (Fig. 8), renamed and made an
    /// associated function — see the module doc comment for why. Builds a
    /// fresh `Trace` by recursively walking `result` (a
    /// [`TileGraph::sub_graph_tiling`]/[`TileGraph::resolved_tiling`]
    /// result) at `level`.
    ///
    /// Deviates from the paper's literal `for n : g.nodes()` loop in one
    /// respect, not specified precisely enough by the pseudocode to port
    /// verbatim: `SubGraphTilingResult::children` can (via
    /// `sub_graph_tiling`'s own deduplication) cover *several* of
    /// `result`'s nodes with a single child, when their subgraphs one
    /// level up turn out identical (a fused group becomes one virtual
    /// node at the next level — see §3.1/Fig. 5). So rather than
    /// dispatching once per node unconditionally, this dispatches once
    /// per *child* (covering every node that child's `nodes` contains)
    /// and only falls through to `compute_tile` directly for a node no
    /// child covers (e.g. one with no declared `tile_spec`, a hard
    /// boundary `propagate` never resolved past).
    pub fn trace_graph(
        tile_graph: &TileGraph,
        result: &SubGraphTilingResult,
        level: MemoryLevelKind,
        hardware: &HardwareProfile,
    ) -> Self {
        let mut trace = Self::default();
        trace.virtual_node(&result.nodes, level);

        let footprint = tile_graph.mem_footprint_with_config(&result.nodes, &result.config);
        trace.allocate(footprint, level);
        trace.load_tiles(&result.nodes, level);

        let next_level = hardware.next_memory_level(Some(level));
        let mut executed = std::collections::HashSet::new();

        for &node in &result.nodes {
            if executed.contains(&node) {
                continue;
            }
            match next_level {
                None => {
                    trace.compute_tile(node);
                    executed.insert(node);
                }
                Some(next_level) => {
                    if let Some(child) = result.children.iter().find(|c| c.nodes.contains(&node)) {
                        let child_trace =
                            Self::trace_graph(tile_graph, child, next_level, hardware);
                        trace.events.extend(child_trace.events);
                        executed.extend(child.nodes.iter().copied());
                    } else {
                        trace.compute_tile(node);
                        executed.insert(node);
                    }
                }
            }
        }

        trace.store_tiles(&result.nodes, level);
        trace
    }
}

impl ExecuteDevice for Trace {
    fn virtual_node(&mut self, nodes: &[NodeId], level: MemoryLevelKind) {
        self.events.push(TraceEvent::VirtualNode {
            nodes: nodes.to_vec(),
            level,
        });
    }

    fn allocate(&mut self, footprint: u64, level: MemoryLevelKind) {
        self.events.push(TraceEvent::Allocate { footprint, level });
    }

    fn load_tiles(&mut self, nodes: &[NodeId], level: MemoryLevelKind) {
        self.events.push(TraceEvent::LoadTiles {
            nodes: nodes.to_vec(),
            level,
        });
    }

    fn compute_tile(&mut self, node: NodeId) {
        self.events.push(TraceEvent::ComputeTile { node });
    }

    fn store_tiles(&mut self, nodes: &[NodeId], level: MemoryLevelKind) {
        self.events.push(TraceEvent::StoreTiles {
            nodes: nodes.to_vec(),
            level,
        });
    }
}

#[cfg(test)]
mod tests {
    use teeny_core::device::hardware::MemoryLevel;
    use teeny_core::graph::{DtypeRepr, Shape};
    use teeny_core::model::ExecutableOp;
    use teeny_core::utils::dag::Dag;

    use super::*;
    use crate::graph::optimizer::anduin::TileConfig;

    struct TestOp {
        name: &'static str,
        dtype: DtypeRepr,
        shape: Shape,
        is_input: bool,
    }

    impl ExecutableOp for TestOp {
        fn name(&self) -> &str {
            self.name
        }

        fn is_input(&self) -> bool {
            self.is_input
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
            self.dtype
        }

        fn as_any(&self) -> &dyn std::any::Any {
            self
        }
    }

    fn op(name: &'static str, shape: Shape, is_input: bool) -> Box<dyn ExecutableOp> {
        Box::new(TestOp {
            name,
            dtype: DtypeRepr::F32,
            shape,
            is_input,
        })
    }

    fn single_level_hardware(kind: MemoryLevelKind) -> HardwareProfile {
        HardwareProfile {
            name: "single-level".to_string(),
            compute_units: 1,
            memory_levels: vec![MemoryLevel {
                kind,
                capacity: u64::MAX,
                bandwidth: None,
                latency: None,
            }],
        }
    }

    fn two_level_hardware() -> HardwareProfile {
        HardwareProfile {
            name: "two-level".to_string(),
            compute_units: 1,
            memory_levels: vec![
                MemoryLevel {
                    kind: MemoryLevelKind::Register,
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
        }
    }

    #[test]
    fn trace_graph_computes_directly_at_the_top_level() {
        let mut dag: Dag<Box<dyn ExecutableOp>> = Dag::new();
        let a = dag.add_node(op("a", vec![Some(4)], true));

        let tile_graph = TileGraph::from_dag(&dag);
        let hardware = single_level_hardware(MemoryLevelKind::DeviceMemory);
        let footprint = tile_graph.mem_footprint(&[a]);

        let result = SubGraphTilingResult {
            nodes: vec![a],
            config: TileConfig::default(),
            children: Vec::new(),
        };

        let device = Trace::trace_graph(
            &tile_graph,
            &result,
            MemoryLevelKind::DeviceMemory,
            &hardware,
        );

        assert_eq!(
            device.events,
            vec![
                // A singleton group still gets a VirtualNode -- a node not
                // fused with anything is still "viewed from this level" as
                // itself (see ExecuteDevice::virtual_node's doc comment).
                TraceEvent::VirtualNode {
                    nodes: vec![a],
                    level: MemoryLevelKind::DeviceMemory,
                },
                TraceEvent::Allocate {
                    footprint,
                    level: MemoryLevelKind::DeviceMemory,
                },
                TraceEvent::LoadTiles {
                    nodes: vec![a],
                    level: MemoryLevelKind::DeviceMemory,
                },
                TraceEvent::ComputeTile { node: a },
                TraceEvent::StoreTiles {
                    nodes: vec![a],
                    level: MemoryLevelKind::DeviceMemory,
                },
            ]
        );
    }

    #[test]
    fn trace_graph_recurses_into_a_child_and_does_not_double_execute_its_nodes() {
        let shape = vec![Some(4)];
        let mut dag: Dag<Box<dyn ExecutableOp>> = Dag::new();
        let a = dag.add_node(op("a", shape.clone(), true));
        let b = dag.add_node(op("b", shape, false));
        dag.add_edge(a, b);

        let tile_graph = TileGraph::from_dag(&dag);
        let hardware = two_level_hardware();

        let child_footprint = tile_graph.mem_footprint(&[a, b]);
        let child = SubGraphTilingResult {
            nodes: vec![a, b],
            config: TileConfig::default(),
            children: Vec::new(),
        };
        let top_footprint = tile_graph.mem_footprint(&[a, b]);
        let top = SubGraphTilingResult {
            nodes: vec![a, b],
            config: TileConfig::default(),
            children: vec![child],
        };

        let device = Trace::trace_graph(&tile_graph, &top, MemoryLevelKind::Register, &hardware);

        // b is covered by the same child as a (child.nodes contains both),
        // so the outer loop must dispatch the recursive call once (when it
        // reaches a) and skip b afterwards -- not compute or recurse twice.
        assert_eq!(
            device.events,
            vec![
                // Two VirtualNode events for the same {a, b} pair, one per
                // level -- each is that level's own view of the group
                // (mirrors Fig. 5's Conv+ReLU virtual node existing at L0
                // and again, consolidated further, at L1).
                TraceEvent::VirtualNode {
                    nodes: vec![a, b],
                    level: MemoryLevelKind::Register,
                },
                TraceEvent::Allocate {
                    footprint: top_footprint,
                    level: MemoryLevelKind::Register,
                },
                TraceEvent::LoadTiles {
                    nodes: vec![a, b],
                    level: MemoryLevelKind::Register,
                },
                TraceEvent::VirtualNode {
                    nodes: vec![a, b],
                    level: MemoryLevelKind::DeviceMemory,
                },
                TraceEvent::Allocate {
                    footprint: child_footprint,
                    level: MemoryLevelKind::DeviceMemory,
                },
                TraceEvent::LoadTiles {
                    nodes: vec![a, b],
                    level: MemoryLevelKind::DeviceMemory,
                },
                TraceEvent::ComputeTile { node: a },
                TraceEvent::ComputeTile { node: b },
                TraceEvent::StoreTiles {
                    nodes: vec![a, b],
                    level: MemoryLevelKind::DeviceMemory,
                },
                TraceEvent::StoreTiles {
                    nodes: vec![a, b],
                    level: MemoryLevelKind::Register,
                },
            ]
        );
    }

    #[test]
    fn trace_graph_falls_back_to_compute_tile_when_no_child_covers_a_node() {
        // Two independent nodes; only a is covered by a child (simulating
        // e.g. a hard boundary that kept b from being fused further) -- b
        // must still be computed, directly at the current level, not
        // silently dropped.
        let mut dag: Dag<Box<dyn ExecutableOp>> = Dag::new();
        let a = dag.add_node(op("a", vec![Some(4)], true));
        let b = dag.add_node(op("b", vec![Some(4)], true));

        let tile_graph = TileGraph::from_dag(&dag);
        let hardware = two_level_hardware();

        let child_a = SubGraphTilingResult {
            nodes: vec![a],
            config: TileConfig::default(),
            children: Vec::new(),
        };
        let top = SubGraphTilingResult {
            nodes: vec![a, b],
            config: TileConfig::default(),
            children: vec![child_a],
        };

        let device = Trace::trace_graph(&tile_graph, &top, MemoryLevelKind::Register, &hardware);

        // b's ComputeTile must appear directly in the trace (not nested
        // inside a second Allocate/LoadTiles/StoreTiles block), and exactly
        // once.
        let compute_b_count = device
            .events
            .iter()
            .filter(|event| matches!(event, TraceEvent::ComputeTile { node } if *node == b))
            .count();
        assert_eq!(compute_b_count, 1);
        assert!(device.events.contains(&TraceEvent::ComputeTile { node: a }));
    }
}
