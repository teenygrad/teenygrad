# Fusion case 1: pure elementwise chains (single input/output)

Status: **decided** (teenygrad-1bf.1)  
Parent: teenygrad-1bf (`TritonFusionBackend` spec)  
Prior art: spinorml-1fj.1 Phase 1 (graph `Op::Fused` + allowlist; commit `efc1a4e5d`)

This note answers the three acceptance questions for case 1 only. Broader
cases (fan-in, reduction, epilogue, …) stay on their own child issues.

## Scope reminder

Case 1 is a **linear chain** of single-input / single-output elementwise ops,
each shape-preserving, each with exactly one consumer on interior nodes.
Already modeled by:

- `Op::Fused { members: Vec<Op> }`
- `is_fusable_elementwise()` + `Graph::fuse_elementwise_chains()`
- Triton lowering stub that still errors on `Op::Fused`

---

## Decision 1 — fusability gate: keep the allowlist (with a hard contract)

**v1 keeps and extends `is_fusable_elementwise`'s hand-verified allowlist.**
It does **not** replace graph-time fusion with a live `RuntimeOp::grid()`
equality check against neighbouring ops.

### Why

`Graph::fuse_elementwise_chains()` runs on the `teeny-core::Op` graph, before
any `RuntimeOp` exists. A real `grid()` comparison is only available inside
`TritonLowering`, after per-op runtime ops are constructed. Moving the
*decision* entirely into lowering would either:

- duplicate the chain-finding rewrite after RuntimeOps exist, or
- leave `Op::Fused` production in the graph layer with no authoritative gate

For case 1, the allowlist **is** the equality check, compiled offline: every
listed op is an equivalence class of kernels that share one program → element
mapping when lowered the normal way.

### Contract the allowlist must satisfy

An op may be added to `is_fusable_elementwise` only when, under
`TritonLowering`'s standard construction for that op, all of the following
match every other allowlisted member:

1. Kernel `BLOCK_SIZE` const used in the `pid * BLOCK_SIZE + arange` mapping
2. `RuntimeOp::grid(output_shape)` for any shape (same formula)
3. Pointer-in / pointer-out forward signature shape (so members chain)

CUDA thread counts are **not** part of this contract: they come from PTX
metadata (`.reqntid` / `num_warps`) after compile, not from `RuntimeOp`.
`RuntimeOp::block()` / `backward_block()` were removed for that reason.

Structural graph rules stay as today: shape-preserving, one input, single
consumer on interior nodes. Ops that fail any of (1)–(3) stay off the list
even if they are "elementwise" in the math sense (e.g. Softmax — different
grid formula; norms — not this shape class).

### Safety net at lower time (not a second decision)

When Phase-2 lowering synthesizes a fused kernel, it **must assert** that
every member's constructed `RuntimeOp` agrees on `grid(shape)` (and the
construction uses the same `BLOCK_SIZE`) before emitting. That catches
allowlist drift; it does not replace the graph-time allowlist for case 1.

### Parent open question (when is fusion decided?)

For case 1 specifically: **keep the allowlist**, extended op-by-op under the
contract above. Hybrid later is fine for other cases: graph identifies
candidate regions; lowering may refuse or re-shape. Case 1 does not need to
move the decision into `TritonLowering` to ship.

---

## Decision 2 — kernel-body composition: scratch-buffer threading for v1

**v1 keeps spinorml-1fj.1's scratch-buffer / ping-pong approach**, not
value-level register-resident composition.

### Mechanics

1. Concatenate each member's generic `#[kernel]` fn body into one compilation
   unit (dedupe shared imports).
2. Synthesize one entry point that calls members in order.
3. Thread intermediates via a small ping-pong pair of buffers:
   member *N*'s output pointer = member *N+1*'s input pointer.
4. External input feeds member 0; final member writes the fused node's output.

### Why not register-resident yet

Existing activation kernels are pointer-in / pointer-out. True register
fusion needs each body split into "load / pure compute / store" so fused
members compose on values. That is a real refactor across every allowlisted
op (and every future add). Scratch-buffer fusion already collapses **N
launches → 1**, which is the bulk of the measured glue-launch win from
spinorml-1fj, without that refactor. MLIR inline + DCE may still erase some
store/load pairs when aliasing is provable; treat that as opportunistic, not
required for correctness of v1.

**Follow-up (out of case-1 AC):** value-level composition as an explicit
later upgrade once case-1 lowering works end-to-end.

---

## Decision 3 — `FusedRuntimeOp` may delegate `grid()` to any member, under precondition

For case 1 only, a synthesized `FusedRuntimeOp` may implement `grid()`,
`pack_args()` (adjusted for the fused entry's buffer wiring), etc. by
**delegating to any one member's** `RuntimeOp`.

This is legitimate **only because case 1 is program→element-shape-identical by
construction** via Decision 1's allowlist contract — not because fused
regions in general may pick an arbitrary member.

Document that precondition at:

- `Op::Fused` / `is_fusable_elementwise` docs (graph side)
- the `Op::Fused` lowering arm / `FusedRuntimeOp` (kernels side)

If the equality assert at lower time fails, refuse to lower (hard error), do
not silently pick a member.

Cases with mismatched natural shapes (reduction-terminated, multi-reduction,
GEMM epilogues, …) must **not** use this delegation rule; they synthesize a
region-level launch config by their own case rules.

---

## Case-1 v1 implementation checklist (not part of this decision issue)

Tracked for the implementation follow-up after teenygrad-1bf agrees:

1. Implement `Op::Fused` lowering: source concat + ping-pong entry +
   `FusedRuntimeOp` with `grid()` equality assert.
2. Keep `fuse_elementwise_chains()` opt-in until (1) lands; then consider
   wiring into the fusion backend's `plan_fusions`, still not blindly into
   `Graph::optimise()` for every backend.
3. Extend allowlist op-by-op only under Decision 1's contract (Gelu, etc.
   are candidates once verified).
4. Add a regression test that allowlisted RuntimeOps agree on `grid` for a
   sample shape under the same construction path TritonLowering uses.

## Non-goals for case 1

- Fan-in / broadcast / residual-add (case 2)
- Fan-out materialize-vs-recompute (case 3)
- Any reduction or retiling (cases 4–6)
- Training fused backward policy (case 12) — inference-first for v1 lowering
  is acceptable; training may refuse `Op::Fused` until specified
