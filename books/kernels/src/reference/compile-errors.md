# Common Compile Errors

Errors you will actually hit, with the real text and the fix. Grouped by which
compiler produced them, because that tells you where to look.

## From the `#[kernel]` macro

These come from the proc macro, at ordinary `cargo build` time.

---

```text
#[kernel] requires a type parameter with a `Triton` bound
```

**Cause.** The macro finds the GPU by looking for the type parameter bounded by
`Triton`. Yours has none.

**Fix.** Add it. Every kernel starts:

```rust,ignore
pub fn my_kernel<T: Triton, D: Num, const BLOCK_SIZE: i32>(...)
```

This is also what you get if you wrote `Triton` but did not import it.

---

```text
`dtypes` must be a list, e.g. `dtypes = [f32, f64]`
```

**Cause.** `#[kernel(dtypes = f32)]` — a bare name where a list belongs.

**Fix.** Brackets: `#[kernel(dtypes = [f32])]`.

---

```text
`f16` is not a known scalar dtype
```

Or any other name. **Cause.** The dtype list only accepts scalar dtype
identifiers: `bool`, `i8`…`i64`, `u8`…`u64`, `f16`, `bf16`, `f32`, `f64`.

**Fix.** Use one of those. Note that `f16` and `bf16` *parse* but cannot be
monomorphized — see Chapter 15 and `KNOWN-GAPS.md` item 4.

---

```text
duplicate dtype `f32` in `dtypes`
```

**Fix.** Remove the repeat.

---

```text
unknown `#[kernel]` argument `dtype` (expected `dtypes` or `backward`)
```

**Cause.** A typo. There are exactly two arguments.

---

```text
cannot infer supported dtypes: a `#[kernel]` that opts into dispatch without an
explicit `dtypes = [..]` must have a dtype type parameter bound by one of
Dtype/Num/Int/Float/Bool
```

**Cause.** You used `#[kernel(backward = ...)]`, which opts into dispatch, but
your dtype parameter is bounded by something the macro cannot expand into a
dtype set.

**Fix.** Either bound it by one of the five, or list the dtypes explicitly.
Chapter 15 has the table.

## From rustc, about kernel bodies

---

```text
the trait bound `<T as Triton>::Pointer<D>: AddOffsets<...>` is not satisfied
```

**Cause.** A missing `where` clause. This is the most common error in a first
kernel, and it means you called `add_offsets` without declaring that you would.

**Fix.** The three clauses every kernel carries:

```rust,ignore
where
    T::I32Tensor: types::Tensor<i32, 1>,
    T::I32Tensor: Comparison<i32, BoolTensor = T::BoolTensor>,
    T::Pointer<D>: AddOffsets<i32, 1, T::I32Tensor, Output = T::Tensor<T::Pointer<D>>>,
```

Copy them verbatim. They are identical in every kernel in this tree, which is
the first item in `API-FRICTION.md`.

---

```text
expected fn pointer `fn(...) -> ...`
   found closure `[closure@src/kernels/mine.rs:42:30]`
```

**Cause.** A closure passed to `T::reduce` or `T::associative_scan`.

**Fix.** Make it a named function. The combine function is compiled from
captured source text, so it must be statically known — Chapter 13. The error
does not mention kernels at all, which is why it belongs on this list.

---

```text
no method named `lt` found for associated type `<T as Triton>::I32Tensor`
```

**Cause.** The `Comparison` trait is not in scope, or its `where` clause is
missing.

**Fix.** `use teeny_triton::triton::types::Comparison;` — usually already
covered by the glob import every kernel starts with.

---

```text
mismatched types
   expected `Option<<T as Triton>::BoolTensor>`
      found `Option<<T as Triton>::I32Tensor>`
```

**Cause.** An integer tensor used as a mask. A mask must come from a
comparison.

**Fix.** `Some(offsets.lt(n))`, not `Some(offsets)`.

## From `teenyc`

These appear when `compile_kernel` runs, which is at *run* time — Chapter 3.

---

```text
no teenyc rustup toolchain found; set TEENYC_PATH to the teenyc binary, or
install one with `cargo teeny install-toolchain` (see cargo-teeny)
```

**Fix.** Either install it, or set `TEENYC_PATH`. Chapter 4. There is
deliberately no fallback to a bare `teenyc` on `$PATH`.

---

```text
multiple teenyc rustup toolchains found (a, b); set TEENYC_PATH to disambiguate
```

**Fix.** Set `TEENYC_PATH` to the one you want.

---

**A `teenyc` failure mentioning names you did not use.** The kernel body is
compiled against a small generated environment, not your crate. `println!`, your
own helper functions, and most of `std` are not there.

**Fix.** Use only the `Triton` trait and plain arithmetic in a kernel body. For
printing, `T::device_print`.

## From the CUDA driver

At load time, when the PTX becomes machine code.

---

```text
PTX .version 8.6 does not support .target sm_120a
```

**Cause.** `teenyc`'s default PTX version is newer than your driver accepts.
Seen on Blackwell.

**Fix.** `TEENYC_PTX_VERSION=87`. A `teenyc`-side default; the SDK cannot work
around it.

---

```text
wrapper.h:17:10: fatal error: 'cuda.h' file not found
```

**Cause.** Not a kernel error at all — `teeny-cuda`'s build.rs generating
bindings, with no CUDA toolkit installed. A driver alone is not enough.

**Fix.** Install the toolkit, or build without the `cuda` feature. Chapter 4.

---

```text
custom op 'my.op' is not handled — implement CustomOp::lower()
```

**Cause.** A `CustomOp` whose `lower` still returns the default `None`.

**Fix.** Implement it. Chapter 20.

## Things that do not error

The worst list, because there is nothing to search for.

| Symptom | Likely cause |
|---|---|
| Wrong numbers, no crash | Argument order in `pack_args` or the launch tuple. Nothing checks it |
| Wrong numbers only at the end of a buffer | Missing mask. Chapter 7 |
| Wrong reduction results | Masked lanes not filled with the identity. Chapter 10 |
| Gradients all zero | `has_backward` not overridden, or `backward_grid` left at its `[0,0,0]` default. Chapter 22 |
| Results differ run to run | Atomics. Expected; compare with a tolerance. Chapter 14 |
| Correct but slow after a port | Block size tuned for a different card. Chapter 24 |
| Kernel silently does nothing | Calling the `#[kernel]` function directly. It is not the kernel; the struct is |

For any of these, the first move is Chapter 9: read the MLIR. Count the loads
and stores, check the constants, and look for the mask operand.
