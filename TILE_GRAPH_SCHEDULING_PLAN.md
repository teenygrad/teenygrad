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
pub struct TileAxisBinding { pub dims: &'static [usize], pub block_const: &'static str, pub extent_param: &'static str, pub window: Option<TileWindow> }
pub struct TensorTileSpec { pub param: &'static str, pub rank: usize, pub axes: &'static [TileAxisBinding], pub reduction_axis: Option<usize>, pub untiled_dims: &'static [&'static str] }
pub struct KernelTileSpec { pub inputs: &'static [TensorTileSpec], pub outputs: &'static [TensorTileSpec] }
```

`ExecutableOp::tile_spec(&self) -> Option<KernelTileSpec>` is a new default
method (`None`) — coverage is opt-in per kernel, same as the original.
`KernelExecutable` (`kernels/teeny-kernels/src/graph/mod.rs`) carries a
`tile_spec: Option<KernelTileSpec>` field, set at `TritonLowering`
construction time (no `RuntimeOp::as_any` needed — `&Op`'s static fields
are already in scope at each match arm before erasure into
`KernelExecutable`). Three ops are tagged so far: `Op::Relu`
(`RELU_TILE_SPEC` — flat single-axis identity), `Op::MatMul`/`Op::Gemm`
(`MATMUL_TILE_SPEC` — `M`/`N` resolved, `K` correctly left unresolved),
and `Op::BatchNorm2d` (`BATCHNORM2D_TILE_SPEC` — see `dims` below).
`TileOp` carries the same field, populated in `from_dag` from
`node.value.tile_spec()`.

**`TileAxisBinding.dims` (renamed from `dim: usize`, fixed
teenygrad-1nr.8).** Almost always one entry (`&[N]`, the ordinary
one-block-const-per-real-axis case), but a kernel that flattens several
real dims into one iterated range with a single block const — NCHW
batchnorm2d's `H*W` loop, one `BLOCK_HW` — needs more than one:
`dims: &[h_dim, w_dim]`, outermost to innermost. `propagate` resolves the
actual block-sized value onto the *last* (innermost) entry and collapses
every other entry to a bare `TileDim::Fixed(1)` — product-preserving
(matches the real per-tile element count) even though it isn't a literal
axis-aligned subregion once the block size doesn't evenly divide the
innermost axis's extent, the same kind of simplification
`enumerate_subtiles`'s own doc comment already accepts for masked/partial
last tiles. `BATCHNORM2D_TILE_SPEC` (`graph/mod.rs`) is the first real
spec using this: `dims: &[2, 3]` (H, W) for its one `HW` axis, batch/
channels (dims 0/1) untiled. Known limitation, not solved here:
`enumerate_subtiles`'s candidate search still varies every axis's ladder
independently, with no notion that H and W are jointly driven by one
flattened binding — teaching the search itself to respect that is a
separate follow-up (`propagate` resolving a *given* output tile correctly
is what this covers).

`TileWindow` is carried over for fidelity but **not yet consumed by
`propagate`** — the original never actually wired it into
`propagate_within_kernel` either (only the separate `mem_traffic`
estimator read it, given an already-resolved value from the caller).
Teaching `propagate` to invert a windowed axis's extent from its driving
output axis is a real follow-up.

**`untiled_dims`, by contrast, is already effectively honored (fixed
teenygrad-1nr.7).** Originally `propagate` built each input/output tile by
positionally zipping only `axes` against the real shape, so a spec with
`axes.len() < rank` (e.g. a conv2d-style kernel that only block-tiles one
axis, with the rest genuinely relying on `untiled_dims`) got a
`TileEdgeShape` truncated to `axes.len()`, silently dropping the untiled
dims instead of falling back to their full extent — reproduced by
`tile_graph/mod.rs::tests::propagate_of_a_partially_tiled_spec_should_not_drop_the_untiled_dims`.
Fixed by indexing by each `TileAxisBinding`'s declared `dim` (not
position) and building every tile at full `rank`, overwriting only the
dims `axes` actually names — the untiled-dim string list itself still
isn't read, but the *effect* it describes now holds structurally. This
unblocks writing real `Conv2d`/`BatchNorm2d` tile specs (partial-axis,
relying on the fallback) as a follow-up.

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

The rest of Fig. 7's scheduler, in `tile_graph/mod.rs` and `anduin/schedule.rs`:

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

## §3.3 — `trace_graph` + `codegen` scaffold (shipped, teenygrad-1nr.6)

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

Two modules hold this, split by concern — the walk that *produces* a trace
lives with the trace types it produces, not with the `ExecuteDevice`
interface definition:

- `anduin/codegen.rs`: `ExecuteDevice` — Table 1's `allocate`/`load_tiles`/
  `compute_tile`/`store_tiles`, plus one addition of our own,
  `virtual_node(nodes, level)` (not in Table 1) — plus `codegen` and
  `DagCodegen` (below).

  `virtual_node` makes Welder §3.1/Fig. 5's *virtual node* — the original
  `NodeId`s consolidated into one fused unit as viewed from a level — an
  explicit event, called first (before `allocate`) on every `trace_graph`
  invocation, i.e. once per recursion frame, with that frame's own
  `result.nodes` (including singleton groups — a node not fused with
  anything is still "viewed from this level" as itself). Unlike the paper,
  where a virtual node can exist at *any* level including deep
  register-level sub-fusion, this codebase's real `HardwareProfile`s only
  ever declare `SharedMemory`/`DeviceMemory` as levels a scheduling
  decision can target — Triton gives no explicit control over registers,
  L1, or L2, which stay hardware-managed within a single kernel body. So
  every virtual node this reports is a genuine candidate kernel boundary:
  it's the exact grouping `DagCodegen` needs to decide "these nodes become
  one compiled kernel," without needing to reconstruct that grouping by
  pattern-matching `Allocate`/`StoreTiles` nesting out of the flatter
  four-event trace after the fact.

  `codegen(trace, device)` runs the *replay* direction through the same
  `ExecuteDevice` interface — given an already-recorded trace (typically
  `Trace::events`), it replays each event through an `ExecuteDevice`
  again, the same interface driven from a static list instead of a live
  walk. `DagCodegen` is the intended `ExecuteDevice` for this direction:
  it's meant to build a real `Dag<Box<dyn ExecutableOp>>` of custom
  (fused) ops as it replays — matching `GraphOptimizer::optimize`'s own
  `(Dag, Vec<usize>)` contract, the way the original hand-coded Anduin
  fusion strategies did before removal — but every method is currently a
  dummy `todo!()` stub. Building it for real is §4.2's scope (register-level
  `compute_inline`-style fusion, shared-memory load/store rewriting,
  block/thread index remapping, a best-fit shared-memory allocator) and
  overlaps heavily with the still-open teenygrad-1nr.1.
- `anduin/trace.rs`: `TraceEvent`/`Trace` (the `ExecuteDevice`
  implementation that just records events), and `Trace::trace_graph` —
  Welder §3.3's `ExecuteGraph` (Fig. 8), renamed *and* made an associated
  function of `Trace`: what this walk actually does in this codebase is
  build a trace — the only way a `Trace` gets created — not execute
  anything for real, so a free `execute_graph`/`trace_graph` taking a
  generic `ExecuteDevice` claimed more genericity than the walk ever used
  (nothing has ever driven it with anything but a `Trace`). Structural
  walk implementing Fig. 8's recursion over a `TileGraph` and a
  `SubGraphTilingResult`, terminating at the top of `HardwareProfile`'s
  declared hierarchy (via the same `HardwareProfile::next_memory_level`
  method `sub_graph_tiling` uses). Single self-recursive function: the
  recursive case builds a child `Trace` via `Self::trace_graph(...)` and
  merges its `events` into the parent's, rather than threading a
  `&mut Self` accumulator through a separate helper. One documented
  deviation: a fused group can
  become one child covering several nodes (`SubGraphTilingResult::children`'s
  existing dedup-by-subgraph behavior), so `trace_graph` dispatches once
  per *child* rather than unconditionally once per node, falling back to
  `compute_tile` directly only for a node no child covers. `codegen`'s
  replay direction is where genericity over `ExecuteDevice` actually
  matters (any device can consume a trace); building one only ever
  produces a `Trace`.

## Suggested implementation order

1. ~~`EdgeId` + `set_connect`/`connect_level` + `extract_subgraph`~~ — done.
2. ~~`mem_footprint` + `mem_traffic` + `Profiler`/`SimpleProfiler`~~ — done.
3. ~~`TileConfig` + `propagate`~~ — done (teenygrad-1nr.2).
4. ~~`EnumerateSubtiles`~~ — done (teenygrad-1nr.3).
5. ~~`sub_graph_tiling` + `schedule_graph`~~ — done (teenygrad-1nr.4/.5),
   replacing Fig. 7's `SubGraphTiling`/`GraphConnecting`.
6. ~~Persist the resolved schedule + structural `Trace::trace_graph` +
   `codegen` replay scaffold~~ — done (teenygrad-1nr.6), Welder §3.3's
   `ExecuteGraph` (Fig. 8), renamed `trace_graph` and made an associated
   function of `Trace`. `DagCodegen`, the `ExecuteDevice` meant
   to turn a replayed trace into a real `Dag` of custom ops, is a dummy
   stub (`todo!()` per method) — implementing it for real is the next step
   here.
7. ~~Wire `schedule_graph`/`Trace::trace_graph`/`codegen` into
   `Anduin::optimize`~~ — done. `optimize` now runs `schedule_graph`, then
   for every topologically-ordered node's outgoing edges with a
   `resolved_tiling`, builds a `Trace` (rooted at `MemoryLevelKind::Register`
   — the level `schedule_graph`'s `sub_graph_tiling` call always tiles
   from, regardless of which level wins `connect_level`) and replays it
   through a single shared `DagCodegen` via `codegen`, skipping an edge
   once every node it covers has already been traced by an earlier edge
   (sibling edges off the same node can cover overlapping node sets).
   `DagCodegen` is still the `todo!()` stub from step 6, so running
   `optimize` now panics inside `DagCodegen::virtual_node` instead of at a
   blanket `todo!()` at the top — the pipeline is wired end-to-end, but the
   last stage remains deliberately unimplemented pending teenygrad-1nr.1.
   The `Vec<usize>` `optimize` returns is `DagCodegen::into_dag`'s own
   (currently also `todo!()`) mapping, unchecked against the `mapping`
   parameter `optimize` receives — composing the two into the graph-node-
   idx -> output-dag-node-idx mapping the `GraphOptimizer` trait promises
   is deferred until `DagCodegen` defines what its mapping actually means.

   Split into two `Anduin` associated functions so the schedule/fusion
   decision can be tested without hitting `DagCodegen`'s stubs:
   `Anduin::schedule(&dag, hardware) -> (TileGraph, Vec<Trace>)` runs
   §3.1-§3.3 up to (not including) codegen, and `Anduin::codegen(dag,
   &traces) -> (Dag, Vec<usize>)` is the finalization step `optimize` now
   just composes them through. Two tests exercise `Anduin::schedule`
   directly against real lowered ops (not the toy `TestOp`s the
   `tile_graph`/`trace` unit tests use):
   - `conv2d_batchnorm_silu_schedule_fuses_the_three_compute_nodes_apart_from_input`
     — none of `Conv2d`/`BatchNorm2d`/`Silu` has a declared `KernelTileSpec`
     (only `Relu`/`MatMul` do), yet `schedule_graph`/`sub_graph_tiling`
     still group conv+batchnorm+silu into one SharedMemory-level virtual
     node, separate from the input load — confirms the fusion *decision*
     is driven by `mem_footprint`/`mem_traffic` structurally, not gated on
     a tile_spec being present.
   - `relu_reduce_sum_relu_schedule_treats_the_whole_chain_as_one_flat_group`
     — a `[2048, 4096]` F32 reduction sandwiched between two `Relu`s (which
     *do* have a tile_spec) never recurses past Register at all: every
     node computes directly in one flat top-level virtual node. Documented
     as current behavior, not asserted as correct — worth another look
     once `sub_graph_tiling`'s config search is better exercised.
8. §4.1's hardware-aligned refinements (reviving the removed, real-hardware-
   calibrated `teenygrad-3w0` `CostModel`) and §4.2's real code generation
   (kernel composition, best-fit shared-memory allocator) — both still
   deferred; §4.2 overlaps heavily with teenygrad-1nr.1's still-open
   `Tile<D>` composition rework, which is the harder blocker for actually
   materializing fused kernels from a schedule.
