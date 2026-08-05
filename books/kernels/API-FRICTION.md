# API friction

Places where writing a kernel in Rust is harder than writing the same kernel in
Python Triton, and boilerplate a kernel author has to repeat for every new op.

This is product feedback, not a list of bugs. Everything here works. It is
recorded because a book has to explain each of these, and the length of that
explanation is a decent proxy for how much the API is costing its users.

Ordered by how much of the book each one costs.

---

## 1. `load` and `store` take eight and six positional parameters

**Every call in the tree looks like this:**

```rust
let a = T::load(
    a_ptr.add_offsets(offsets),
    Some(in_bounds),
    None,
    &[],
    None,
    None,
    None,
    false,
);
```

Six of those eight arguments are "no thanks". In Python Triton the same line is
`tl.load(a_ptr + offsets, mask=in_bounds)`. Rust has no keyword or default
arguments, so every optional parameter has to be spelled at every call site.

**Cost to the book.** Chapter 7 has to explain all eight parameters before the
reader can read any kernel, when only two matter for the first several chapters.
Every subsequent code sample carries six tokens of noise per memory access.

**Options, roughly in order of how much they'd help:**

- A builder: `T::load(ptr).mask(in_bounds).other(zeros).get()`. Reads well,
  but the DSL is compiled from captured source text, so anything added here must
  survive that round trip.
- A macro with named arguments: `load!(ptr, mask = in_bounds)`.
- Keep `load` and add `load_masked(ptr, mask)` / `load_masked_or(ptr, mask,
  other)` / `store_masked(ptr, val, mask)` covering the three shapes that
  actually occur. Smallest change; would remove most of the noise.

## 2. Every kernel repeats the same three `where` clauses

```rust
where
    T::I32Tensor: types::Tensor<i32, 1>,
    T::I32Tensor: Comparison<i32, BoolTensor = T::BoolTensor>,
    T::Pointer<D>: AddOffsets<i32, 1, T::I32Tensor, Output = T::Tensor<T::Pointer<D>>>,
```

Identical, character for character, in `elemwise_add`, `softmax`, `detect_decode`
and essentially every other kernel. It is the price of `arange` + `lt` +
`add_offsets`, which is to say: the price of indexing memory at all.

**Cost to the book.** Chapter 5 shows a reader their first kernel and three of
its lines are an incantation that will not be explained until much later. The
honest options are to explain associated-type bounds in chapter 5 (wrong
audience — the reader has never written a kernel) or to say "copy this" (which
the brief rules out).

**Suggested fix.** A supertrait that bundles them, so a kernel body reads
`T: Triton1D<D>` or similar:

```rust
pub trait Triton1D<D: Dtype>: Triton
where
    Self::I32Tensor: types::Tensor<i32, 1>
        + Comparison<i32, BoolTensor = Self::BoolTensor>,
    Self::Pointer<D>: AddOffsets<i32, 1, Self::I32Tensor,
        Output = Self::Tensor<Self::Pointer<D>>>,
{}
```

This is the single highest-value change on this list for the book's first ten
chapters.

## 3. Launch arguments are positional and unchecked

`RuntimeOp::pack_args` visits arguments in order:

```rust
visitor.visit_ptr(inputs[0].0); // a_ptr
visitor.visit_ptr(inputs[1].0); // b_ptr
visitor.visit_ptr(output);      // out_ptr
visitor.visit_i32(n as i32);    // n_elements
```

The trailing comments are load-bearing — they are the only thing tying this
sequence to the kernel's parameter list. Nothing checks that the order, the
count or the types agree with the `#[kernel]` fn they are packing for. Swap two
`visit_ptr` calls and you get wrong numbers, not an error.

The macro already knows the parameter list: it generates
`type Args<'a> = (*mut D, *mut D, *mut D, i32)`. That type is used for the direct
`device.launch` path but not for the `RuntimeOp` path.

**Suggested fix.** Have the macro generate a typed packer — a struct with named
fields, one per kernel parameter, that implements the visit sequence. Then
`pack_args` becomes field assignment and the compiler checks the arity.

## 4. Const generics become lowercased runtime fields, silently

`const BLOCK_SIZE: i32` on the fn becomes `pub block_size: i32` on the generated
struct, and a positional argument to `new()`. Two const generics means two
positional `i32`s at the constructor:

```rust
Conv2dBnSiluGemmForward::<f32>::new(BLOCK_M, BLOCK_N, BLOCK_K, GROUP_M)
```

Nothing stops a caller passing them in the wrong order, and the name change from
`BLOCK_M` to `block_m` is only documented inside the macro's implementation.

**Suggested fix.** Generate a `Params` struct with named fields, or at minimum
document the transformation in the macro's public docs — Chapter 8 currently has
to reverse-engineer it from `kernel.rs:303`.

## 5. `reduce` and `associative_scan` take `fn` pointers, not closures

```rust
fn reduce<D, O>(x: …, axis: i32, combine_fn: fn(…) -> …, keep_dims: bool) -> …;
```

A closure that captures anything is rejected. This is correct — the combine
function is compiled as source text, so it must be statically known, exactly as
Python Triton requires `@triton.jit` on the helper. But the error a reader gets
is a generic Rust closure-coercion message that says nothing about kernels.

**Suggested fix.** Nothing in the API. Worth a callout in Chapter 13 and an entry
in the compile-errors reference.

## 6. The `#[kernel]` fn is never called, and nothing says so

The annotated function is emitted unchanged and is callable Rust, but calling it
does nothing useful — the real artefact is its source text, captured into a
string on the generated struct. A reader who writes `elemwise_add_forward(...)`
in their own code gets no warning.

**Suggested fix.** Emit the function as `#[doc(hidden)]`, or under a name that
signals it is not for calling. Failing that, the generated struct's doc comment
should say it.

## 7. The kernel body is `no_std`-ish in a way the type system does not express

Kernel bodies are compiled against a generated DSL sysroot
(`kernels/teeny-triton/build.rs` splices `teeny-core`'s `dtype/mod.rs` into a
fake `std::ops`). So `std` is available to the *host* compiler when it
type-checks the fn, but not to `teenyc` when it compiles the captured text. A
`println!` in a kernel body type-checks and then fails in an unfamiliar
compiler.

**Cost to the book.** Chapter 3 has to explain this before Chapter 5, or readers
will hit it and have no model for the error.

## 8. Two spellings for comparisons

`Triton::lt(x, y)` (trait method, tensor vs tensor), `Triton::lt_scalar(x, y)`
(tensor vs scalar), and `x.lt(y)` (via `Comparison`, which is what the `where`
clause in every kernel imports). All three exist; kernels in the tree use the
method form almost exclusively.

**Suggested fix.** Decide which is idiomatic and document it. The reference
section has to list all three regardless, but the chapters should use one.

## 9. `RuntimeOp` has 20 methods, most with defaults

Implementing an op means reading all of them to find the four you need
(`n_activation_inputs`, `param_shapes`, `pack_args`, `block`, `grid`) plus, under
`training`, five more. The `#[cfg(feature = "training")]` split means the set
changes depending on feature flags, which the book has to explain twice.

**Suggested fix.** Nothing structural, but a `RuntimeOp` doc example showing the
minimal impl would save Chapter 21 a page.

## 10. `launch_config` invites a launch that cannot work

`teeny_cuda::testing` offers three ways to build a launch configuration, and the
most obvious-looking one is the one that fails:

```rust,ignore
let cfg = launch_config(n_elements, block_size);      // both from you
let cfg = launch_config_from_program(n, &program);    // threads from PTX
let cfg = launch_config_with_grid(grid_x, &program);  // grid from you, threads from PTX
```

Only the third is generally correct. `teenyc` chooses the thread count and
records it in the PTX as `.reqntid`; the driver rejects any other block
dimension with a bare `CUDA error: 1 (invalid argument)` that names nothing.

The trap is that `launch_config` *works* whenever your `BLOCK_SIZE` happens to
equal the compiler's choice — 128 for a simple elementwise kernel, which is the
first thing anybody writes. It then breaks the moment the constant changes. The
book's own vector-add example was written this way and was correct only by
coincidence until a block-size sweep exposed it.

**Suggested fix.** Have `launch_config` validate against
`program.metadata.threads_per_block()` and return a real error naming both
numbers, or drop it in favour of the two program-aware forms. A helper whose
correctness depends on a coincidence is worse than no helper.

While there: `KernelMetadata` is `pub(crate)`, so a caller cannot ask what thread
count the compiler picked except by going through these helpers. Exposing
`threads_per_block()` on `CudaProgram` would make the constraint discoverable
instead of folklore.
