# What `#[kernel]` Generates

Chapter 5 used a type you never wrote:

```rust,ignore
let kernel = VectorAdd::<f32>::new(BLOCK_SIZE);
```

`#[kernel]` generated it. This chapter is about what else it generated, because
knowing that turns several confusing things into obvious ones.

## The three outputs

From one annotated function, the macro emits three items.

**Your function, unchanged.** It compiles as ordinary Rust and is type-checked
by the ordinary compiler. Nothing calls it. Its purpose is to be checked and to
be *read* — the macro converts its tokens back into a string, and that string is
what the GPU compiler sees.

**A struct**, named by converting the function name to PascalCase:
`vector_add` becomes `VectorAdd`. This is the thing you construct and pass
around.

**A dispatcher**, but only if you asked for one with `#[kernel(dtypes = [...])]`
or `#[kernel(backward = ...)]`. Chapter 15 covers it. Without those arguments,
you get two items, not three.

## The struct

The generic parameters get split three ways. The `Triton` parameter disappears —
it exists only to give the body its operations. Const generics become **runtime
fields**. Everything else stays a type parameter.

So `vector_add<T: Triton, D: Num, const BLOCK_SIZE: i32>` produces a
`VectorAdd<D>` with these members:

```rust,ignore
pub name: &'static str,          // "vector_add"
pub id: String,                  // "vector_add__f32__128"
pub block_size: i32,             // from `const BLOCK_SIZE`, lowercased
pub kernel_source: String,       // your function, as text
pub entry_point_source: String,  // the wrapper, as text
pub source: String,              // the two joined
```

Two details worth pinning down, because both bite.

**Const generics are lowercased.** `const BLOCK_SIZE` becomes the field
`block_size`, and a positional argument to `new()`. With one constant that is
harmless. With four it is not:

```rust,ignore
Conv2dBnSiluGemmForward::<f32>::new(BLOCK_M, BLOCK_N, BLOCK_K, GROUP_M)
```

Nothing checks the order. They are all `i32`.

**`id` identifies the kernel.** It is the function name, the dtypes, and the
const values, joined by double underscores — `vector_add__f32__128`. Change the
block size and you get a different id, so a different compilation. That is why
one kernel with two block sizes is two compiled kernels.

It is not, however, what names the files on disk. The cache uses a content hash:

```text
/tmp/teenyc_cache/vector_add_5f69418a643d1353dba2ce66de8ed3dc4e1644c0d9474da4517ed6e7d3f67ff9.o
```

`{name}_{sha256 of the source}`, with three files per kernel — `.o` (the PTX,
despite the extension), `.mlir`, and `.rs` (the full generated source that
`teenyc` was given). The readable `id` and the on-disk name are two different
things, and it is the hash you will be looking at.

The struct also implements `Kernel`, whose `Args` associated type is the
launch-argument tuple — `(*mut f32, *mut f32, *mut f32, i32)` for this kernel,
derived from the function's parameter list with `T::Pointer<D>` rewritten to a
raw pointer.

## The entry point

This is the part worth looking at properly, because it explains the shape of
everything else.

`teenyc` cannot call a generic Rust function. It needs one concrete symbol with
a C calling convention. So the macro writes one, as text, with the generic
parameters filled in.

Here is the real one, taken from the `.rs` file the Chapter 5 run left in the
cache — the actual text `teenyc` compiled:

```rust,ignore
use triton::llvm::triton::num::*;
use triton::llvm::triton::pointer::LlvmPointer;
type LlvmTriton = triton::llvm::triton::LlvmTriton;

#[no_mangle]
pub extern "C" fn vector_add_entry_point(a_ptr: *mut f32, b_ptr: *mut f32, out_ptr: *mut f32, n_elements: i32) {
    let a_ptr = LlvmPointer(a_ptr as *mut _);
    let b_ptr = LlvmPointer(b_ptr as *mut _);
    let out_ptr = LlvmPointer(out_ptr as *mut _);
    vector_add::<LlvmTriton, f32, 128>(a_ptr, b_ptr, out_ptr, n_elements);
}
```

Those are the last eight lines of a 3,068-line file. The rest is the DSL itself,
spliced in ahead of your kernel — which is what "compiled against a small
generated environment" from Chapter 3 means in practice.

Read the last line of the wrapper first:

```rust,ignore
vector_add::<LlvmTriton, f32, 128>(...)
```

There it is. `T` is `LlvmTriton`, the implementation of the `Triton` trait that
knows how to emit GPU operations. `D` is `f32`. And `BLOCK_SIZE` is `128` — a
literal, in the source text, exactly as Chapter 6 said it must be.

The rest:

- **`#[no_mangle]` and `extern "C"`** give a symbol with a predictable name,
  `{function name}_entry_point`, which is what the loader looks up in the PTX.
- **The pointer rewrapping** turns the raw `*mut f32` the C ABI can carry into
  the `LlvmPointer` the DSL works with. Three lines of ceremony you never see.
- **The parameters** are your parameters, in order. This is the contract behind
  the tuple you pass to `launch`: position for position, nothing checked.

## Why your kernel is generic

The `T: Triton` parameter now makes sense. It is not there for you.

It is there so the *same source text* can be compiled twice: once by rustc, with
`T` as an abstract type parameter, which is what type-checks your kernel; and
once by `teenyc`, with `T` as `LlvmTriton`, which is what emits GPU code.

One function, two compilers, two meanings of `T`.

## Seeing it yourself

```bash
cargo expand -p teeny-triton --example vector_add
```

That prints everything the macro produced. It is worth doing once. The generated
source strings appear as one long escaped literal, which is ugly but is exactly
what gets compiled.

The easier route is to run the kernel and read the `.rs` file it leaves in the
cache — same content, already unescaped, alongside the `.mlir` and the PTX:

```bash
ls $TEENYC_CACHE_DIR    # or /tmp/teenyc_cache
```

## The consequences, collected

Everything below follows from the above, and each has bitten someone:

- **You cannot call helper functions from a kernel body** unless they are part of
  the DSL. Your crate is not in scope where the captured text is compiled.
- **Combine functions must be `fn` pointers, not closures.** A closure that
  captured a variable could not be written out as text. Chapter 13.
- **The struct's field names differ from the generic names.** `BLOCK_SIZE` →
  `block_size`.
- **Changing a constant recompiles.** Different id, different cache entry.
- **A `#[kernel]` fn is callable from Rust and does nothing useful if you call
  it.** Nothing warns you.

Next: watching it happen, by reading what came out.
