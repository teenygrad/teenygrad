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
| `TileConfig` (per-*edge* tile shape, distinct from `TileEdge.shape`'s full shape) | **Done** — `HashMap<EdgeId, TileEdgeShape>`, see "`Propagate` implementation plan" below. |
| `Propagate` | **Done** (teenygrad-1nr.2), via a revived `KernelTileSpec` (name-matching, not per-category formulas — see below), not a port of Welder's own TE-based implementation. |
| `EnumerateSubtiles` | **Done** (teenygrad-1nr.3), power-of-two candidates only — see below. |
| `SubGraphTiling` / `GraphConnecting` | **Done** (teenygrad-1nr.4/.5) as `sub_graph_tiling`/`schedule_graph` — see below. Still doesn't rewrite the DAG into fused kernels (blocked on teenygrad-1nr.1) or do real-hardware validation (`d.Profile` is still `SimpleProfiler`'s structural stand-in). |

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

Also shipped, in `anduin::profile` (its own module): a `Profiler` trait —
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

## `Propagate` (shipped, teenygrad-1nr.2)

### How Welder's own implementation actually works

Read against the reference implementation
(`../Welder/python/welder/{graph.py,bestfit.py,shape_inference/{te.py,common.py}}`)
rather than just the paper text. Welder represents every op as a TVM
**tensor expression (TE)** — an array comprehension (`compute(shape, lambda
i, j: expr(i, j))`). `_extract_dependent_region`
(`shape_inference/te.py:174`) walks each op's TE body and records, per input
tensor, the *symbolic index expression* used to read it (e.g. for
`conv[i,j] = sum_k(x[i*stride+k], w[k])`, the dependent region of `x` is
literally `i*stride+k`). `Propagate`
(`IRNode.propogate` → `InputShapeInference.infer`,
`shape_inference/common.py:21`/`te.py:92`) then runs backward **interval-bound
propagation**: given an output tile as a `ConstIntBound` per axis, it
evaluates those recorded index expressions through TVM's arithmetic analyzer
in reverse topological order to get the tightest bound each input needs.
Reduction axes get their own bound (full extent, or a partial `rstep` for
split reductions) via the same mechanism.

This is genuinely generic — it treats pointwise, reduction, and conv
identically because it's walking the same symbolic form for all of them.
It only works because Welder's frontend lowers every op into that symbolic
TE representation first. **We don't have anything like that**: our kernels
are opaque Triton/CUDA source strings behind `KernelExecutable` (see the
gap analysis above), not a symbolic array-comprehension IR we can walk.
Building one would essentially mean adding a TE-equivalent layer to
teenygrad — a materially bigger project than tile-graph scheduling itself,
and out of scope here.

One thing the reference implementation *does* resolve, usefully: reduction
axes aren't dynamically discovered either — by the time `Propagate` runs,
`self.raxis` is just read off the already-built TE compute op's
`reduce_axis` property (`graph.py:171`). Welder resolves "which axes are
reduced" one layer earlier (Relay/TE lowering), not inside `Propagate`
itself. Our equivalent is to resolve `Reduce*`'s axes at our own
`TritonLowering` time, the same distance upstream.

**Shipped instead: revived `KernelTileSpec`** (teenygrad-1nr.2), not the
`TilePropagationKind`-enum sketch this section originally proposed. While
investigating that enum's implementation, `git log` turned up a fuller,
already-tested prior attempt at this exact feature
(`kernels/teeny-triton/src/tile.rs` /
`kernels/teeny-kernels/src/graph/optimizer/propagate.rs`, deleted at
`84ca6eedf^`, built incrementally under the `teenygrad-3w0` epic). Its
actual algorithm isn't per-category formulas at all — it's **name
matching**: every axis declares an `extent_param` name, and any two axes
anywhere (any tensor, any op) sharing a name are the same free variable.
Seeding an output tile's axis values resolves every axis sharing those
names for free — Pointwise and MatMul-shaped ops need *no* formula at all
this way (e.g. `a_ptr`/`c_ptr` both declaring an axis named `"M"`).

It was deleted for two reasons, neither of which indicts this design: the
hand-coded Anduin fusion strategies it fed were removed as non-Welder
pattern-matching (`431230501`; `propagate.rs` was explicitly *kept* at that
point — "real algorithm infrastructure, not a hand-coded pattern"), and
then the *separate* `#[tile(...)]` attribute macro that auto-generated
index arithmetic into the compiled kernel body was removed because that
codegen broke composability for kernel-calling-kernel fusion (`84ca6eedf`;
contributing factor: no consumer existed yet, since Anduin wasn't wired up
either). This revival keeps only the metadata + name-matching half —
`KernelTileSpec` is pure `const` data, consumed for scheduling analysis,
never driving what's generated into a kernel's source — and does not bring
back the attribute macro or its codegen.

### Types (`core/teeny-core/src/model/tile_spec.rs`)

```rust
pub struct TileWindow { pub stride_const: &'static str, pub pad_const: &'static str, pub kernel_size_const: &'static str }
pub struct TileAxisBinding { pub dim: usize, pub block_const: &'static str, pub extent_param: &'static str, pub window: Option<TileWindow> }
pub struct TensorTileSpec { pub param: &'static str, pub rank: usize, pub axes: &'static [TileAxisBinding], pub reduction_axis: Option<usize>, pub untiled_dims: &'static [&'static str] }
pub struct KernelTileSpec { pub inputs: &'static [TensorTileSpec], pub outputs: &'static [TensorTileSpec] }
```

`ExecutableOp::tile_spec(&self) -> Option<KernelTileSpec>` is a new default
method (`None`) — coverage is opt-in per kernel, same as the original.
`KernelExecutable` (`kernels/teeny-kernels/src/graph/mod.rs`) carries a
`tile_spec: Option<KernelTileSpec>` field, set at `TritonLowering`
construction time (no `RuntimeOp::as_any` needed — `&Op`'s static fields
are already in scope at each match arm before erasure into
`KernelExecutable`). Two ops are tagged as a first proof-of-concept slice:
`Op::Relu` (`RELU_TILE_SPEC` — flat single-axis identity) and
`Op::MatMul`/`Op::Gemm` (`MATMUL_TILE_SPEC` — `M`/`N` resolved, `K`
correctly left unresolved). `TileOp` carries the same field, populated in
`from_dag` from `node.value.tile_spec()`.

`TileWindow` is carried over for fidelity but **not yet consumed by
`propagate`** — the original never actually wired it into
`propagate_within_kernel` either (only the separate `mem_traffic`
estimator read it, given an already-resolved value from the caller).
Teaching `propagate` to invert a windowed axis's extent from its driving
output axis is a real follow-up.

### `TileConfig` and `TileGraph::propagate`

```rust
pub struct TileConfig { tiles: HashMap<EdgeId, TileEdgeShape> } // pub fn get/len/is_empty

pub fn propagate(&self, nodes: &[NodeId], output_tiles: &HashMap<EdgeId, TileEdgeShape>) -> TileConfig
```

Keyed by `EdgeId`, not `NodeId` — preserves the same per-edge flexibility
`TileEdge::memory_level` already has (two consumers of one producer can
resolve to two different tile shapes on their own distinct edges, with no
merge/reconciliation needed, since each incoming edge has exactly one
consumer by construction). `output_tiles` seeds the tile shape requested on
`nodes`'s own boundary output edges — our analogue of Welder's
`Map<Axis, Dim> config`. Walking `nodes` in reverse topological order, at
each node with a declared `tile_spec`: resolve its own output tile (from
whichever outgoing edge is already known, falling back to its full untiled
shape), seed an `extent_param -> TileDim` map from that tile against its
spec's declared output axes, then for each parent edge resolve that
input's axes by name lookup (falling back to that axis's own full extent
when the name is unresolved — e.g. a reduction axis with no output-side
counterpart, which is *correct*, not a gap: Welder's own model treats a
reduction axis's tile size as the search's decision, not `Propagate`'s).
Degrades to a hard boundary (stops, doesn't guess) at any node with no
`tile_spec`, a mismatched declared output rank, or a parent-edge count
that doesn't match the spec's declared input count — the same positional
producer-to-spec correspondence limitation the original `propagate_graph`
had.

### Explicit non-goals

- **Extents only, no offsets/intervals** — fine for byte-count purposes,
  not sufficient to emit a correct tiled kernel (§3.3, still out of scope).
- **`TileWindow` (sliding-window/conv) not yet consumed by `propagate`** —
  see above.
- **Coverage is opt-in and currently just `Relu`/`MatMul`** — the other
  ~40+ `TritonLowering` match arms still return `tile_spec: None`
  (mechanically threaded through so the field is never a compile error);
  extending coverage is incremental, ordinary follow-up work, not a design
  gap.
- **`Op::Custom`/`CustomOp`** has no backward-shape contract, so custom
  kernels are `None`/hard-boundary by construction.

### What this unblocks next

`SubGraphTiling` (Fig. 7) can now call `propagate` per candidate
output-tile shape and rank the result by `MemFootprint`/`MemTraffic` — the
last missing primitive before `GraphConnecting`/`SubGraphTiling` itself.
(Since shipped — see below.)

## `EnumerateSubtiles` / `sub_graph_tiling` / `schedule_graph` (shipped, teenygrad-1nr.3/.4/.5)

The rest of Fig. 7's scheduler, in `tile_graph/mod.rs` and the new
`anduin/scheduler.rs`:

- **`TileGraph::mem_footprint_with_config`/`mem_traffic_with_config`** —
  the config-aware footprint/traffic these three all needed (the plan
  above originally left this as a non-goal; implementing `EnumerateSubtiles`
  made it a hard prerequisite, not optional, since scoring a candidate tile
  is meaningless against the full untiled shape). `TileEdge::byte_size`'s
  core was factored into a shared `shape_byte_size(shape, dtype)` so both
  the existing full-shape methods and these new config-aware ones share one
  implementation.
- **`TileGraph::enumerate_subtiles(nodes, root, capacity)`** — Welder
  §4.1's `EnumerateSubtiles`, restricted to power-of-two candidate steps
  per axis (`1, 2, ..., extent.next_power_of_two()`), not Welder's own
  any-divisor set — see the method's doc comment for the concrete Triton
  evidence (`next_power_of_two` call sites, `BLOCK_SIZE`-must-be-a-power-
  of-two doc comments across several real kernels) this restriction is
  grounded in. Base-tile growth + expanding neighbor search, scored by
  `MemTraffic`-per-output-element via `propagate` + `mem_traffic_with_config`,
  pruned by `mem_footprint_with_config` against a capacity. `TileWindow`
  (sliding-window/conv) is *not* consumed here yet — same as `propagate`,
  see above.
- **`TileGraph::sub_graph_tiling(nodes, root, level, hardware, top_k)`** —
  Welder's `SubGraphTiling`. One documented deviation: always re-derives
  each recursion level's candidates fresh from that level's own root,
  rather than threading the paper's `c` down from the level below — the
  paper doesn't specify enough to port that literally, and even Welder's
  own shipped policy (`policy/default.py`'s `emit_config`) searches once
  from a single base tile rather than implementing this literal recursion.
  Returns a `SubGraphTilingResult` tree; recursion terminates at the top of
  `HardwareProfile`'s declared memory hierarchy.
- **`schedule_graph`** (`anduin/schedule.rs`) — Welder's `GraphConnecting`,
  renamed (`GraphConnecting` reads like a type, not an action). One
  documented deviation: `Profiler` doesn't yet score a specific
  `TileConfig` (see its own doc comment), so this asks `sub_graph_tiling`
  for only its single best-ranked candidate per level and profiles the
  *structural* cost of the extracted subgraph instead — still responds
  correctly to which level is tried, just coarser than the paper's
  per-candidate `Min(d.Profile(configs))`. Mutates a `TileGraph`'s
  `connect_level`s in place; does **not** rewrite the DAG into fused
  kernels — that's still blocked on teenygrad-1nr.1 (see `Anduin::optimize`'s
  own module doc comment).

## §3.3 — `execute_graph` + `codegen` scaffold (shipped, teenygrad-1nr.6)

`schedule_graph` originally computed a `SubGraphTilingResult` per candidate
memory level and kept only `connect_level`, discarding the actual chosen
tile shapes even for the winning level. Fixed: `TileGraph` gained
`resolved_tiling: HashMap<EdgeId, SubGraphTilingResult>` (read via
`TileGraph::resolved_tiling`, written via `TileGraph::record_resolved_tiling`),
and `schedule_graph` now caches the winning level's already-computed result
instead of throwing it away.

What's worth persisting turned out to be a *tree*, not a flat per-edge
shape — Fig. 8's `ExecuteGraph` recurses through the memory hierarchy,
needing each level's own workspace/config, which is exactly
`SubGraphTilingResult`'s existing shape (config at this level + recursively
tiled `children`). One gap found while wiring this up: `SubGraphTilingResult`
had no record of *which* nodes each `children` entry covers, so the tree
wasn't walkable — fixed by adding a `nodes: Vec<NodeId>` field, populated
from `sub_graph_tiling`'s own `extract_subgraph` calls (already computed,
previously discarded after the dedup check).

Three small modules now hold this, split by concern:

- `anduin/codegen.rs`: `ExecuteDevice` (`allocate`/`load_tiles`/
  `compute_tile`/`store_tiles`, Table 1) and `execute_graph`, the
  structural walk implementing Fig. 8's recursion over a `TileGraph`, a
  `SubGraphTilingResult`, and an `ExecuteDevice`, terminating at the top of
  `HardwareProfile`'s declared hierarchy (via the same `next_memory_level`
  helper `sub_graph_tiling` uses). One documented deviation: a fused group
  can become one child covering several nodes (`SubGraphTilingResult::children`'s
  existing dedup-by-subgraph behavior), so `execute_graph` dispatches once
  per *child* rather than unconditionally once per node, falling back to
  `compute_tile` directly only for a node no child covers.
- `anduin/trace.rs`: `TraceEvent` and `TraceDevice`, the `ExecuteDevice`
  implementation `execute_graph` is driven with today — it records events
  rather than doing anything real.
- Back in `anduin/codegen.rs`: `codegen`, which runs the *other* direction
  through the same `ExecuteDevice` interface — given an already-recorded
  trace (typically `TraceDevice::events`), it replays each event through an
  `ExecuteDevice` again, the same interface driven from a static list
  instead of a live walk. `DagCodegen` is the intended `ExecuteDevice` for
  this direction: it's meant to build a real `Dag<Box<dyn ExecutableOp>>`
  of custom (fused) ops as it replays — matching `GraphOptimizer::optimize`'s
  own `(Dag, Vec<usize>)` contract, the way the original hand-coded Anduin
  fusion strategies did before removal — but every method is currently a
  dummy `todo!()` stub. Building it for real is §4.2's scope (register-level
  `compute_inline`-style fusion, shared-memory load/store rewriting,
  block/thread index remapping, a best-fit shared-memory allocator) and
  overlaps heavily with the still-open teenygrad-1nr.1.

## Suggested implementation order

1. ~~`EdgeId` + `set_connect`/`connect_level` + `extract_subgraph`~~ — done.
2. ~~`mem_footprint` + `mem_traffic` + `Profiler`/`SimpleProfiler`~~ — done.
3. ~~`TileConfig` + `propagate`~~ — done (teenygrad-1nr.2).
4. ~~`EnumerateSubtiles`~~ — done (teenygrad-1nr.3).
5. ~~`sub_graph_tiling` + `schedule_graph`~~ — done (teenygrad-1nr.4/.5),
   replacing Fig. 7's `SubGraphTiling`/`GraphConnecting`.
6. ~~Persist the resolved schedule + structural `execute_graph` + `codegen`
   replay scaffold~~ — done (teenygrad-1nr.6), Welder §3.3's `ExecuteGraph`
   (Fig. 8). `DagCodegen`, the `ExecuteDevice` meant to turn a replayed
   trace into a real `Dag` of custom ops, is a dummy stub (`todo!()` per
   method) — implementing it for real is the next step here.
7. §4.1's hardware-aligned refinements (reviving the removed, real-hardware-
   calibrated `teenygrad-3w0` `CostModel`) and §4.2's real code generation
   (kernel composition, best-fit shared-memory allocator) — both still
   deferred; §4.2 overlaps heavily with teenygrad-1nr.1's still-open
   `Tile<D>` composition rework, which is the harder blocker for actually
   materializing fused kernels from a schedule.
