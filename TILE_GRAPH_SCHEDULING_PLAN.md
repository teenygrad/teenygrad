# Tile-graph scheduling: interface plan (Welder §3.2)

Source: *Welder: Scheduling Deep Learning Memory Access via Tile-graph*
(OSDI '23, Shi et al.), `osdi23-shi.pdf` in the repo root. This plan covers
§3.1 (tile-graph model), §3.2 (tile-graph scheduling), and the boundary with
§3.3/§4 (hardware mapping / hardware-aligned tile search), read against our
current `kernels/teeny-kernels/src/graph/optimizer/anduin/tile_graph/mod.rs`.

Tracking: `teenygrad-1nr`.

## What §3.2 specifies

Four primitives on the tile-graph (Fig. 6), all we currently lack except a
partial structural base:

```
void       SetConnect(Edge *edge, MemLevel level);
TileConfig Propagate(TileGraph g, Map<Axis, Dim> config);
size_t     MemFootprint(TileGraph g);
size_t     MemTraffic(TileGraph g);
```

...and a two-step scheduler built on them (Fig. 7):

- **`GraphConnecting`** — walks nodes in topological order, and for each
  out-edge tries every memory level via `SetConnect`, extracts the resulting
  fused sub-graph (`ExtractSubgraph`), scores it (`SubGraphTiling` plus a
  hardware profiling call), and keeps the best level.
- **`SubGraphTiling`** — recursive: enumerate candidate output-tile shapes
  (`EnumerateSubtiles`, a Roller-style expanding search — that's §4.1, a
  layer above the tile-graph API itself), `Propagate` each into a full
  `TileConfig`, reject candidates that blow `MemFootprint`, rank the rest by
  `MemTraffic`, keep the top-K, then recurse one memory level up on the
  sub-graphs extracted at that level.
- **`ExtractSubgraph(node, level)`** — walks both in- and out-edges from
  `node`, transitively pulling in any neighbor whose edge's connect-level is
  *above* `level`. It materializes "the fused group at this level" as its
  own sub-`TileGraph`.

## Gap analysis against our `TileGraph`

Our current type (`tile_graph/mod.rs`) has topology (`parents`/`children`),
boundary edges, and *one* shape per node — the full, untiled shape carried
over from `ExecutableOp::output_shape()`. Everything the scheduler needs is
layered on top of that.

| Paper concept | Status |
|---|---|
| Edge identity / mutation | **Done.** `TileGraph` now stores every edge once in an arena, addressed by an opaque `EdgeId` — internal and boundary edges alike, since a boundary edge is just an edge with one endpoint `None`. `EdgeId` replaces the `EdgeRef` enum originally sketched below: no separate edge-kind type is needed. |
| `SetConnect` | **Done** — `TileGraph::set_connect(EdgeId, MemoryLevelKind)`, O(1) via the arena, visible from both the producer's and consumer's adjacency list since both hold the same id. `connect_level`/`edge` are the read side. |
| Memory-level ordering (`ExtractSubgraph`/`GraphConnecting`'s "above this level") | **Done.** `MemoryLevelKind` now derives `PartialOrd`/`Ord`, ranked fast→slow by declaration order (`Register < ... < HostMemory`). |
| `ExtractSubgraph` | **Done.** `TileGraph::extract_subgraph(node, level)` floods `outgoing`/`incoming` adjacency in both directions along edges with `memory_level > level` and returns the reachable node set (`Vec<NodeId>`) — matching the paper's own `SubGraph(nodes)`, not a rebuilt/remapped graph. |
| `MemFootprint` | **Done**, as a peak-live-set estimate rather than a literal best-fit simulation — see below. |
| `MemTraffic` | **Done**, as `Σ(boundary edge bytes)` with no further multiplier — see below; exact only in the degenerate case where the full shape *is* the tile shape. |
| `TileConfig` (candidate per-node *tile* shape, distinct from the full shape) | **Missing as a type.** This is what `Propagate` produces — it is *not* the same as `TileEdge.shape`, which holds the un-tiled shape today. |
| `Propagate` | **Missing, and the hard part.** Needs per-op backward shape inference from an output tile ("the dependent region of the input tensor can be accurately determined by analyzing its tensor expression and output tile size" — §3.1). Since the last refactor, `TileOp` deliberately no longer carries the source `Op` enum — a lowered `ExecutableOp` has no tensor-expression semantics exposed at all. This is the "additional information from the graph — add it to the custom op" case: it needs a new `ExecutableOp` method, not a reach back into `Graph`. |
| `EnumerateSubtiles`, hardware-aligned parallelism, `d.Profile` | Explicitly a later layer (§4.1 / real-hardware validation) — the module doc already anticipates this ("pruning candidates ... before validating the winner on real hardware"). Not blocking for a first `TileGraph` API. |

## Current surface (`set_connect`/`connect_level`/`extract_subgraph` shipped)

`TileGraph` now stores edges in an arena addressed by `EdgeId` (an internal
`Vec<TileEdgeRecord>`, each record's `producer`/`consumer` node index `None`
exactly on the boundary side), with each node keeping `outgoing`/`incoming`
lists of `EdgeId`s into that arena. This replaced the originally-sketched
`EdgeRef` enum entirely — `EdgeId` already addresses internal and boundary
edges uniformly, since a boundary edge is just an edge with one endpoint
`None`, so there's no need for a separate `Internal`/`Input`/`Output` variant
type.

`memory_level` stays per-edge (on `TileEdge`), not per-node: Welder §3.1
connects two adjacent operator-tiles through a reuse-tile "along each
adjacent edge," so a producer with several consumers can legitimately be
connected to each at a different level (e.g. fused into one consumer's
kernel while a different consumer reads a separately materialized copy).
That's intentional — it's exactly the per-edge decision `GraphConnecting`
searches over — not something to collapse to a single node-level value.

Shipped:

```rust
pub type NodeId = usize; // alias, not a newtype -- interchangeable with the source Dag's own indices

impl TileGraph {
    pub fn edge(&self, id: EdgeId) -> &TileEdge;
    pub fn connect_level(&self, id: EdgeId) -> MemoryLevelKind;
    pub fn set_connect(&mut self, id: EdgeId, level: MemoryLevelKind);

    pub fn children(&self, index: NodeId) -> Vec<(NodeId, EdgeId)>;
    pub fn input_edge_id(&self, index: NodeId) -> Option<EdgeId>;
    pub fn output_edge_id(&self, index: NodeId) -> Option<EdgeId>;

    /// The node set reachable from `node` through edges whose connect level
    /// is above `level` (Fig. 7's ExtractSubgraph / `SubGraph(nodes)`).
    pub fn extract_subgraph(&self, node: NodeId, level: MemoryLevelKind) -> Vec<NodeId>;
}
```

`extract_subgraph` returns node ids into *this* graph, not a rebuilt one —
matching the paper's own `SubGraph(nodes)` (a node set, not a fresh graph
object). An earlier version rebuilt a fully independent `TileGraph` with
remapped `0..k` indices and synthesized boundary edges for every cut point;
that's substantially more machinery than the paper needs and nothing
downstream requires it, so it was simplified away. Future cost-model code
(`MemFootprint`/`MemTraffic`) can test an edge's endpoints against the
returned set directly.

Also shipped, in `anduin::profiler` (its own module): a `Profiler` trait —
Welder's `Profile` device interface (`Min(d.Profile(configs))` in
`GraphConnecting`, Fig. 7) — and `SimpleProfiler`, a structural stand-in
implementation:

```rust
pub trait Profiler {
    fn profile(&self, tile_graph: &TileGraph, nodes: &[NodeId], hardware: &HardwareProfile) -> f64;
}

pub struct SimpleProfiler;
impl Profiler for SimpleProfiler { /* boundary-traffic / bandwidth estimate, see below */ }
```

A *real* `Profile` would be built from Table 1's abstracted device
interfaces (`Allocate`/`LoadTiles`/`ComputeTile`/`StoreTiles`) — actually
running (or simulating) the candidate on hardware and timing it. That needs
`TileConfig`/`Propagate`, still blocked on the open fidelity decision below.
`SimpleProfiler` sums, over every edge in `TileGraph::boundary_edges(nodes)`
(an edge with exactly one endpoint in `nodes` — an internal edge, both
endpoints inside, contributes nothing), `bytes / that edge's own
connect-level bandwidth`.

Also shipped, directly on `TileGraph` (§3.1's other two interfaces,
Fig. 6):

```rust
impl TileGraph {
    pub fn mem_traffic(&self, nodes: &[NodeId]) -> u64;
    pub fn mem_footprint(&self, nodes: &[NodeId]) -> u64;
}
```

Both use the full untiled shape `from_dag` captured — no distinct tile
shape exists yet — via a new `TileEdge::byte_size(dtype) -> u64` (element
count × dtype size; a dynamic `TileDim::Sym` axis counts as extent 1, same
simplification `SimpleProfiler` already had, and it's now shared: both
`mem_traffic` and `SimpleProfiler` sum over the exact same
`TileGraph::boundary_edges(nodes)` set). `mem_traffic` is that sum
directly — exactly correct only in the degenerate case where the whole
tensor *is* the tile; otherwise it's what the paper's formula would compute
before multiplying by the number of tile-graphs needed to cover the full
output. `mem_footprint` is a peak-live-set simulation over a topological
walk of `nodes`: a node's own output becomes live when produced, freed once
its last *in-set* consumer has run — unless it also crosses the boundary of
`nodes` (an output edge, or an excluded consumer), in which case it's kept
live for the rest of the walk, since an external reader could need it at
any point. That's an upper bound, not the paper's literal best-fit
allocation (which can pack tighter by reusing freed space below the peak),
and it assumes one materialization per node regardless of per-edge
connect-level divergence — a node whose edges end up at genuinely different
levels could in principle need more than one buffer, unmodeled for now.

Still to add:

```rust
impl TileGraph {
    pub fn propagate(&self, target: TileShapeTarget) -> Result<TileConfig>;
}

pub struct TileConfig {
    /* per-node tile shape, distinct from TileOp's full shape */
}
```

`propagate` is the one that forces a decision on `ExecutableOp`, since every
kernel in the codebase implements that trait. `GraphConnecting` itself (Fig.
7's outer loop) is blocked on `propagate` plus the §3.3/§4 hardware-`Profile`
interface — deliberately deferred rather than scaffolded with a stub, per
the decision to keep each piece (`extract_subgraph`, `Profiler`,
`mem_traffic`/`mem_footprint`) as its own self-contained increment instead
of half-wiring `GraphConnecting` around missing pieces.

## Open decision: `Propagate`'s fidelity

Three options, roughly in increasing effort / increasing faithfulness to the
paper:

1. **Defer `Propagate` for now.** Implement the structural pieces first
   (`EdgeRef`, `SetConnect`, `ExtractSubgraph`) with no `ExecutableOp`
   changes; leave `Propagate`/`MemFootprint`/`MemTraffic` as a follow-up once
   the structural layer is in and tested.

2. **Small fixed category set.** Add one
   `ExecutableOp::tile_propagation_kind() -> TilePropagationKind` method with
   an `Opaque` default:

   ```rust
   enum TilePropagationKind {
       Pointwise,
       Reduction { axes: Vec<usize> },
       SlidingWindow { kernel: [usize; 2], stride: [usize; 2], padding: [usize; 2] },
       Opaque, // default: no fusion across this edge
   }
   ```

   One generic propagation rule per category (four-ish), not per kernel.
   Faster to ship a working scheduler; `Opaque` ops just don't fuse across
   that edge until they get a real category.

3. **Full per-kernel propagation.** New required
   `ExecutableOp::propagate_input_tiles(output_tile) -> Vec<TileEdgeShape>`,
   implemented on every concrete kernel struct (~100 types). Exactly matches
   the paper's per-operator tensor-expression inference; most correct
   long-term, much larger surface to implement up front.

## Suggested implementation order

1. ~~`EdgeId` + `set_connect`/`connect_level` + `extract_subgraph`~~ — done.
2. `TileConfig` type + `propagate` at whichever fidelity is chosen above +
   `mem_footprint` + `mem_traffic`.
3. `EnumerateSubtiles` / hardware-aligned search (§4.1) as its own module,
   consuming the above.
4. `GraphConnecting`/`SubGraphTiling` scheduler, replacing the `todo!()` in
   `Anduin::optimize`.
