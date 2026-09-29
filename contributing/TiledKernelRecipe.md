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
| `block = BLOCK_X` | no | The `const {NAME}: i32` giving the tile size. Omitted means an untiled axis — one CTA per index. |
| `name = "C"` | no | The axis's identity for `GridSpec` matching, and the suffix of the index the prelude binds (`tile_c`). Defaults to `extent`'s spelling, so **omit it unless they genuinely differ**. |
| `dim = X\|Y\|Z` | no | Which hardware grid dimension the axis reads. Defaults to `X`. |

Anything else is a compile error naming these four.

## What the attributes cannot express yet

The spec type is richer than the attribute vocabulary. `#[tiled_kernel]`
currently hardcodes `window: None`, `divide_by: None`, `reduction_axis: None`
and `loop_spec: None`, always emits one dim per axis, and gives every input and
output the **first** tile parameter's axis set.

So these are blocked on macro work, not kernel work:

| need | kernel family | blocked on |
|---|---|---|
| `TileWindow` (stride/pad/kernel) | conv 1/2/3d, the nine pools | `teenygrad-1nr.18.2` |
| accumulation loop / `loop_spec` | conv, pools, norms, reductions, attention | `teenygrad-1nr.18.3` |
| two blocked axes | matmul family | `teenygrad-1nr.18.5` |
| per-parameter distinct axis sets | any input whose axes differ from the output's | see `teenygrad-1tl.7` |

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
