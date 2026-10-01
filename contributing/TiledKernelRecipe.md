# Converting a forward kernel to `#[tiled_kernel]`

One recipe, for every rung of `teenygrad-1tl`. The goal of a conversion is not
just that the kernel still compiles — it is that the kernel **declares its tile
axes**, so `TileGraph::propagate` can schedule across it instead of treating it
as a hard boundary.

## The rule

A forward kernel with `In<Tile<..>>`/`Out<Tile<..>>` parameters must carry
`#[tile(...)]` on **every** one of them.

`tests/test_tile_declarations.rs` enforces this by parsing the crate's own
source. It has to read the source rather than call a method, because the
failure is invisible at the type level: with no attributes the macro silently
falls back to the implicit `BLOCK_SIZE`/`n_elements` convention, the kernel
compiles, its numeric tests pass, and `tile_spec()` is simply *not generated*.
There is no method to assert against.

Out of scope, and correctly ignored by that test:

- **Backward kernels.** `elu_backward` and `selu_backward` have `Tile`
  parameters and are deliberately left alone.
- **Kernels still on raw pointers.** Unconverted is the normal state until the
  relevant rung lands. `conv2d_forward` is the one such kernel, waiting on
  `teenygrad-1tl.7`.
- **`#[tiled_kernel]` used only for dtype dispatch.** `matmul_forward` and
  `flash_attention2_forward` carry the attribute with plain
  `In<T::Pointer<D>>` parameters and no tiling meaning. Keying the rule off the
  attribute instead of off `Tile` parameters would flag them for a non-defect.

## The flat single-axis case

`relu_forward` is the worked example. From raw pointers:

```rust
#[tiled_kernel]
pub fn relu_forward<T: Triton, D: Num, const BLOCK_SIZE: i32>(
    x_ptr: In<T::Pointer<D>>,
    y_ptr: Out<T::Pointer<D>>,
    n_elements: i32,
) {
    let pid = T::program_id(Axis::X);
    let block_start = pid * BLOCK_SIZE;
    let offsets = T::arange(0, BLOCK_SIZE) + block_start;
    let in_bounds = offsets.lt(n_elements);
    let x = T::load(x_ptr.add_offsets(offsets), Some(in_bounds), ...);
    let relu = T::maximum(x, T::zeros_like(x));
    T::store(y_ptr.add_offsets(offsets), relu, Some(in_bounds), ...);
}
```

to the target form:

```rust
#[tiled_kernel]
pub fn relu_forward<T: Triton, D: Num, const BLOCK_SIZE: i32>(
    #[tile(block = BLOCK_SIZE, extent = n_elements)] x: In<Tile<T, D>>,
    #[tile(block = BLOCK_SIZE, extent = n_elements)] y: Out<Tile<T, D>>,
    n_elements: i32,
) {
    let relu = T::maximum(x.tensor, T::zeros_like(x.tensor));
    T::store(y.tensor, relu, x.mask, &[], None, None);
}
```

Four steps:

1. Delete `pid`, `block_start`, `offsets` and `in_bounds` — the generated
   prelude computes them.
2. Change each pointer parameter to a `Tile` and tag it:
   `#[tile(block = BLOCK_SIZE, extent = n_elements)] x: In<Tile<T, D>>`.
3. Replace each `T::load(...)` with `x.tensor`.
4. Pass `x.mask` to `T::store` instead of the hand-built `Some(in_bounds)`.

## The multi-axis case

Two worked examples, both from `teenygrad-1nr.18.1`:

- **`channel_bias_add_forward`** — `[N, C]` with a broadcast `(C,)` operand.
  An input may declare a *subset* of the output's axes; the prelude loads that
  operand's single element and broadcasts it across the blocked axis.
- **`batch_norm_2d_nchw_forward_inference`** — `[B, C, HW]`, `B` on
  `Axis::Y`, four broadcast per-channel operands.

Stack one `#[tile(...)]` per axis, **outermost first**:

```rust
#[tile(extent = H)]                    // one CTA per index, no `block`
#[tile(block = BLOCK_W, extent = W)]   // block-tiled
x: In<Tile<T, D>>,
```

Rules the macro enforces:

- Every `Tile` parameter must declare the **same axes in the same order**
  (a subset is allowed for a broadcast input, but not a different order).
- Exactly **one** axis may carry `block = ..`. Two blocked axes need a 2-D
  index tile — `teenygrad-1nr.18.5`.
- There must be an `Out<Tile<..>>`: the output's axes are what the grid covers
  and what every input resolves against.
- Either *all* `Tile` parameters declare `#[tile(...)]` or none do. A partial
  declaration is a compile error.

## The four attribute keys

| key | required | meaning |
|---|---|---|
| `extent = N` | yes | The `{NAME}: i32` **parameter** the axis's total extent is read from. Propagation is name matching: two axes anywhere declaring the same `extent_param` are the same free variable. |
| `block = BLOCK_X` | no | The `const {NAME}: i32` giving the tile size, **or a decimal integer literal** for an axis the kernel steps one element at a time and so has no `BLOCK_*` const for. Omitted means an untiled axis — one CTA per index, and no binding, so nothing propagates for it. |
| `name = "C"` | no | The axis's identity for `GridSpec` matching, and the suffix of the index the prelude binds (`tile_c`). Defaults to `extent`'s spelling, so **omit it unless they genuinely differ**. |
| `dim = X\|Y\|Z` | no | Which hardware grid dimension the axis reads. Defaults to `X`. |
| `reduce` | no | Bare flag marking the axis this tensor reduces over. At most one per tensor, since `reduction_axis` is a single index. |
| `window(stride = S, pad = P, kernel = K, output = O)` | no | The axis is read through a strided sliding window. `stride`, `pad` and `kernel` each take a const generic's name **or a decimal integer literal**; `output` always names a runtime parameter. `pad` is optional — most pools have no padding const. `output` names the **output** variable the window resolves against, leaving `extent` free to stay truthful about the input's own extent. |

Anything else is a compile error naming these keys.

Three further attributes go on the **function**, not a parameter:

| attribute | meaning |
|---|---|
| `#[tile_loop(trip_count = [A, B, ..])]` | The kernel's accumulation loop. The list is names multiplied together, not a formula — no consumer evaluates it yet. |
| `#[tile_carry(name = [EXTENT, ..])]` | One loop-carried accumulator and its shape. `[1]` for a scalar carry. |
| `#[tile_grid(order = [A, B, ..])]` | The grid axes outermost to innermost, when the body's `pid` decode order differs from the output's dim order, or when a loop covers an axis so it is not a grid axis at all. A subset is meaningful; a repeat is an error. |
| `#[tile_grid(swizzled)]` | No `GridSpec` is emitted. For a `pid` decode that is not a mixed-radix decode of the output's axes — the matmul family's `GROUP_M` grouping. |

### When to reach for `#[tile_grid]`

`grid_spec()` is built from the single `Out` parameter's axes in **tensor dim
order**, and `GridSpec::axes` is documented as mattering, outermost to
innermost. Those agree for nearly every kernel, so the attribute is usually
unnecessary. They disagree when the body permutes: `transpose_2d_forward`'s
output is `[N, M]` while it decodes `pid_m = pid / num_pid_n` outer, so its
grid runs M then N. Declaration order cannot serve both, because an axis's
`dims` entry comes from its position in the list.

Read the `pid` decode before trusting the default. If it yields the output's
dims outermost-to-innermost, omit the attribute.

Three shapes need it:

| body | attribute | why |
|---|---|---|
| `transpose_2d_forward` | `order = [M, N]` | output is `[N, M]`, decode is M-outer |
| `batch_norm_normalize_forward` | `order = [C]` | one program per channel, N walked by a loop |
| `matmul_forward`, `linear_forward` | `swizzled` | `GROUP_M` grouping is not a mixed-radix decode |

`swizzled` means *no* `GridSpec` rather than a wrong one. `GridAxisBinding::dim`
documents several axes on one dim as `pid % extent`, `pid / extent`, repeat —
and a fused rider reads the anchor's decoded values, so asserting a decode the
body does not perform would hand out wrong indices.

## Padding is a degenerate window

A pad kernel maps an output index to an input index by a pure shift
(`ip_range = ol_range - PAD_LEFT`), so an output tile of `block` reads `block`
input elements at a shifted origin. That is a window of `stride = 1,
kernel = 1`: `(block - 1) * 1 + 1 = block`. The pad family has no stride or
kernel const to name, so those fields take literals:

```rust
#[tile(block = BLOCK_OL, extent = L,
       window(stride = 1, pad = PAD_LEFT, kernel = 1, output = OL))]
input_ptr: In<T::Pointer<D>>,
```

This is the one family where the input region is *not* larger than the output
tile — the opposite end of the arithmetic the conv and pool rung exercises.

It is exact for an interior tile, which is what a receptive field describes. A
tile overlapping the pad region reads *fewer* distinct input elements, and the
four families differ in where the out-of-range lanes land: masked off
(constant), mirrored (reflection), clamped (replication) or wrapped
(circular). That changes which elements are read, not how many, so `block`
remains a correct upper bound on the footprint.

## A reduced axis, and why it needs no new field

A rank-reducing kernel's input has an axis the output does not: `reduce_sum` is
`[n_outer, n_inner] → [n_outer]`. Bind the outer axis on both sides, mark the
inner one `reduce`, and bind nothing to it:

```rust
#[tile(block = 1, extent = n_outer)]
#[tile(extent = n_inner, reduce)]
x_ptr: In<T::Pointer<D>>,
#[tile(block = 1, extent = n_outer)]
y_ptr: Out<T::Pointer<D>>,
```

A 4-row output tile then resolves the input to `[Some(4), None]` — four whole
input rows. An axis with no binding keeps its full extent, which for a reduced
axis is the truth rather than a fallback: one output row needs its whole input
row. `teenygrad-1nr.16` recorded this as inexpressible; it is not.

Two things not to do:

- **Do not bind the reduction's block const.** `BLOCK_INNER` is the load width
  covering the whole row under a mask, not a tiling of the axis. Binding it
  claims the axis is chunked when it is read in one piece. Same for GEMM's
  `BLOCK_K`, whose real role is the loop's chunk and belongs in `#[tile_loop]`.
- **Do not expect the `ConstLookup` to fill it.** `resolve_inputs` only visits
  an axis that has a binding, so a reduced axis stays `None` however complete
  the const table is. A consumer reads `None` as "full extent".

`block = 1` records the kernel's own granularity when there is no const to name
— one row per program. It is documentation: the block used in resolution comes
from the propagated output tile, so a multi-row tile still resolves correctly.

## A windowed axis the kernel does not block

A window has to hang off a `TileAxisBinding`, and a binding needs a
`block_const`. The 2-D and 3-D convs and pools block `OW` alone — their `pid`
decode yields a scalar `oh` (and `od`) — so those spatial axes have no
`BLOCK_OH` to name, and before `teenygrad-1tl.7` their windows were silently
dropped, leaving the spec claiming the input was read at full extent.

Write the window anyway and omit `block`. The macro emits a binding with the
decimal literal `"1"`, and `(1 - 1) * stride + kernel` is exactly the rows one
output row reads:

```rust
#[tile(extent = H, window(stride = STRIDE_H, pad = PAD_H, kernel = KH, output = OH))]
#[tile(block = BLOCK_OW, extent = W,
       window(stride = STRIDE_W, pad = PAD_W, kernel = KW, output = OW))]
x_ptr: In<T::Pointer<D>>,
```

This describes the body as written; it does not make the axis tileable. Such an
axis resolves against the output variable its window names, and the output
leaves `OH` untiled, so no block ever propagates for it and it stays at full
extent. Giving it a real block means rewriting the `pid` decode — a port
(`teenygrad-1nr.18.4`), not a declaration.

## What the attributes cannot express yet

Most of the original gaps are closed: `window` (`teenygrad-1nr.18.2`),
`loop_spec` via `#[tile_loop]`/`#[tile_carry]` (`.18.3`), `reduction_axis` via
`reduce`, and a prelude over several blocked axes (`.18.5`). What remains:

| need | kernel family | blocked on |
|---|---|---|
| `divide_by` built per instance from a runtime param | GroupNorm (`C / G`) | `teenygrad-1nr.15` |
| relating an input axis to *no* output axis | `reduce_*`, global pools | `teenygrad-1nr.16` |
| `grid_spec` over several blocked axes | matmul family | route-1 path still takes the first blocked axis |
| supplying const *values* so a window resolves | conv, pools | `teenygrad-1nr.30` |

That last one is why a declared window can still resolve to full extent: nothing
builds a `ConstLookup` at graph level yet. Declare it regardless — the
declaration is what `.30` will read.

Do not hand-author a `KernelTileSpec` beside a kernel to work around a gap. A
spec written next to a kernel is decoupled from it, so nothing catches the two
disagreeing — `teeny-kernels` carried seven such `const`s until they were
deleted for exactly that reason.

## Checklist

- [ ] Every `Tile` parameter carries `#[tile(...)]`.
- [ ] `cargo test -p teeny-kernels --test test_tile_declarations` passes.
- [ ] The kernel's own `tile_spec()` `validate()`s (assert it in a
      `#[cfg(test)] mod tests` beside the kernel, as `relu.rs` and `conv2d.rs`
      do).
- [ ] Numerics re-checked against a PyTorch fixture, not against values
      computed in the test — see `tests/fixtures/generate.py`.
- [ ] Source/asm snapshots re-recorded only after the numerics pass.
