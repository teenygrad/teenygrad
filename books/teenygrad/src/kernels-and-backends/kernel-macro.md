# The `#[kernel]` Macro

Fully documented in the kernels book: **[What `#[kernel]`
Generates](https://docs.teenygrad.org/kernels/first-kernel/kernel-macro)**.

[`teeny-macros`](https://docs.teenygrad.org/api/teenygrad/teeny_macros/) provides
the `#[kernel]` attribute macro (`teeny_macros::kernel`), which marks a function
as a kernel definition consumed by `teeny-triton`/`teeny-kernels`.

## What it emits

From one annotated function, three items:

1. **Your function, unchanged.** It type-checks as ordinary Rust. Nothing calls
   it — the macro converts its tokens back to a string, and that string is what
   the GPU compiler sees.
2. **A struct**, named by PascalCasing the function name: `vector_add` becomes
   `VectorAdd`. Const generics become runtime fields, lowercased —
   `const BLOCK_SIZE` becomes `block_size`, and an argument to `new()`. It
   implements `teeny_core::device::program::Kernel`.
3. **A dispatcher**, only when you pass `dtypes = [...]` or `backward = ...`.
   It maps a runtime `DtypeRepr` to a monomorphized kernel.

It also generates an `extern "C"` entry point, as text, with the generics filled
in — `{name}_entry_point` — because `teenyc` cannot call a generic Rust
function. That symbol is what the loader resolves.

## Arguments

```rust,ignore
#[kernel]                                 // no dispatch
#[kernel(backward = GeluBackward)]        // pair with a gradient kernel
#[kernel(dtypes = [f32, f64])]            // explicit dtype set
```

`backward` opts into dispatch too, inferring the dtype set from the dtype
parameter's trait bound.

See [Compile-Time Parameters and Dtype
Dispatch](https://docs.teenygrad.org/kernels/patterns/specialisation) for the
inference rules, and [Common Compile
Errors](https://docs.teenygrad.org/kernels/reference/compile-errors) for the
macro's diagnostics and what they mean.

## Where to go

| You want | Read |
|---|---|
| The generated struct, field by field | [What `#[kernel]` Generates](https://docs.teenygrad.org/kernels/first-kernel/kernel-macro) |
| The real generated entry point | Same chapter — it quotes the committed snapshot |
| How a kernel fits the pipeline | [Writing a Triton Kernel](./writing-a-kernel.md) |
| What `teenyc` does with the text | [From Rust to PTX](https://docs.teenygrad.org/kernels/orientation/rust-to-ptx) |
