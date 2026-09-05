# Python Triton to Rust

Every method on the `Triton` trait, alphabetically, with its nearest Python
Triton spelling.

Two things to know before using this table.

**The Rust column is authoritative; the Python column is a signpost.** The Rust
signatures come from `kernels/teeny-triton/src/triton/mod.rs` and are checked
against it. The Python names are for orientation when porting — Triton moves
things between `tl` and `tl.math` between releases, so check against the version
you are porting from.

**Optional arguments are spelled out in Rust.** Python has keyword arguments
with defaults; Rust does not. Where Python writes `tl.load(p + o, mask=m)`, Rust
writes the full parameter list with `None`s. The "Rust" column below gives the
name and the arguments that carry meaning, not the full signature.

`D` is a dtype type parameter throughout. `T` is the `Triton` implementation.

## The table

| Rust | Python Triton | Notes |
|---|---|---|
| `T::abs(x)` | `tl.abs(x)` | |
| `T::advance(ptr, offsets)` | `tl.advance` | Block pointers only |
| `T::arange(start, end)` | `tl.arange(start, end)` | Half-open. Returns `I32Tensor` |
| `T::arange_f32(start, end)` | `tl.arange(...).to(tl.float32)` | Avoids an intermediate some backends reject |
| `T::argmax(x, axis, tie_break_left, keep_dims)` | `tl.argmax` | |
| `T::argmin(x, axis, tie_break_left, keep_dims)` | `tl.argmin` | |
| `T::associative_scan(x, axis, f, reverse)` | `tl.associative_scan` | `f` is a `fn` pointer, not a closure |
| `T::assume(cond)` | `tl.assume` | Compiler hint |
| `T::atan(x)` | `tl.math.atan` | |
| `T::atomic_add(ptr, val, mask, sem, scope)` | `tl.atomic_add` | Returns the previous value |
| `T::atomic_and(...)` | `tl.atomic_and` | Integers |
| `T::atomic_cas(ptr, cmp, val, sem, scope)` | `tl.atomic_cas` | No mask argument |
| `T::atomic_max(...)` | `tl.atomic_max` | |
| `T::atomic_min(...)` | `tl.atomic_min` | |
| `T::atomic_or(...)` | `tl.atomic_or` | Integers |
| `T::atomic_xchg(...)` | `tl.atomic_xchg` | |
| `T::atomic_xor(...)` | `tl.atomic_xor` | Integers |
| `T::broadcast(a, b)` | `tl.broadcast` | Returns both, broadcast together |
| `T::broadcast_to(x, shape)` | `tl.broadcast_to` | |
| `T::cast::<Src, Dst>(x, rounding, bitcast)` | `x.to(dtype)` | `bitcast: true` reinterprets bits |
| `T::cat(a, b, can_reorder)` | `tl.cat` | |
| `T::cdiv(x, div)` | `tl.cdiv` | Scalar, not tensor |
| `T::ceil(x)` | `tl.math.ceil` | Floats |
| `T::clamp(x, lo, hi)` | `tl.clamp` | |
| `T::cos(x)` | `tl.cos` | |
| `T::cumprod(x, axis, reverse)` | `tl.cumprod` | |
| `T::cumsum(x, axis, reverse)` | `tl.cumsum` | |
| `T::debug_barrier()` | `tl.debug_barrier` | |
| `T::device_assert(cond, msg, mask)` | `tl.device_assert` | Runs on the device |
| `T::device_print(prefix, val, hex)` | `tl.device_print` | |
| `T::div_rn(x, y)` | `tl.math.div_rn` | Round to nearest |
| `T::dot::<D, O>(a, b, acc, precision, max_imprecise)` | `tl.dot` | `D` inputs, `O` accumulator. Chapter 11 |
| `T::dot_scaled(...)` | `tl.dot_scaled` | FP8 and narrower |
| `T::eq(x, y)` | `x == y` | Returns `BoolTensor` |
| `T::eq_scalar(x, y)` | `x == scalar` | |
| `T::erf(x)` | `tl.math.erf` | |
| `T::exp(x)` | `tl.exp` | |
| `T::exp2(x)` | `tl.exp2` | |
| `T::expand_dims(x, axis)` | `tl.expand_dims` | |
| `T::fdiv(x, y, ieee_rounding)` | `tl.fdiv` | |
| `T::flip(x, dim)` | `tl.flip` | `None` flips all dimensions |
| `T::floor(x)` | `tl.math.floor` | |
| `T::fma(x, y, z)` | `tl.math.fma` | `x * y + z` |
| `T::full(shape, value)` | `tl.full` | |
| `T::gather(src, index, axis)` | `tl.gather` | |
| `T::ge(x, y)` / `T::ge_scalar(x, y)` | `x >= y` | |
| `T::gt(x, y)` / `T::gt_scalar(x, y)` | `x > y` | |
| `T::histogram(x, num_bins, mask)` | `tl.histogram` | Bins of width 1 from 0 |
| `T::inline_asm_elementwise(asm, constraints, is_pure, pack)` | `tl.inline_asm_elementwise` | Ends portability. Chapter 24 |
| `T::interleave(a, b)` | `tl.interleave` | |
| `T::join(a, b)` | `tl.join` | New minor dimension |
| `T::le(x, y)` / `T::le_scalar(x, y)` | `x <= y` | |
| `T::load(ptr, mask, other, ...)` | `tl.load(ptr, mask=, other=)` | Eight arguments. Chapter 7 |
| `T::load_scalar_f32_as_i32(ptr, offset)` | — | No Python equivalent. Reads an `f32` index and truncates |
| `T::load_tensor_descriptor(desc, offsets)` | `desc.load(offsets)` | TMA. Chapter 11 |
| `T::log(x)` | `tl.log` | |
| `T::log2(x)` | `tl.log2` | |
| `T::lt(x, y)` / `T::lt_scalar(x, y)` | `x < y` | The bounds-check idiom. Chapter 7 |
| `T::make_block_ptr(base, shape, strides, offsets, block_shape, order)` | `tl.make_block_ptr` | |
| `T::make_tensor_descriptor(base, shape, strides, block_shape, padding)` | `tl.make_tensor_descriptor` | Chapter 11 |
| `T::max(x, axis, keep_dims)` | `tl.max` | |
| `T::max_constancy(x, values)` | `tl.max_constancy` | Compiler hint |
| `T::max_contiguous(x, values)` | `tl.max_contiguous` | Compiler hint |
| `T::max_with_indices(x, axis, tie_break_left, keep_dims)` | `tl.max(..., return_indices=True)` | |
| `T::maximum(x, y)` | `tl.maximum` | Element-wise, not a reduction |
| `T::min(x, axis, keep_dims)` | `tl.min` | |
| `T::min_with_indices(...)` | `tl.min(..., return_indices=True)` | |
| `T::minimum(x, y)` | `tl.minimum` | Element-wise |
| `T::multiple_of(x, values)` | `tl.multiple_of` | Compiler hint |
| `T::ne(x, y)` / `T::ne_scalar(x, y)` | `x != y` | |
| `T::num_programs(axis)` | `tl.num_programs` | |
| `T::permute(x, dims)` | `tl.permute` | |
| `T::program_id(axis)` | `tl.program_id` | Chapter 6 |
| `T::rand(seed, offsets, n_rounds)` | `tl.rand` | Philox |
| `T::randint(seed, offsets, n_rounds)` | `tl.randint` | |
| `T::randint4x(seed, offsets, n_rounds)` | `tl.randint4x` | Four streams |
| `T::randn(seed, offsets, n_rounds)` | `tl.randn` | |
| `T::ravel(x, can_reorder)` | `tl.ravel` | Flatten to 1-D |
| `T::reduce(x, axis, f, keep_dims)` | `tl.reduce` | `f` is a `fn` pointer. Chapter 13 |
| `T::reshape(x, shape, can_reorder)` | `tl.reshape` | |
| `T::rsqrt(x)` | `tl.rsqrt` | |
| `T::shared_alloc::<D>(shape)` | — | No Python equivalent. Allocates a kernel-lifetime rank-2 indexed shared-memory buffer |
| `T::shared_barrier()` | — | No Python equivalent. CTA-wide handshake between an indexed shared-memory write and a later read |
| `T::shared_load_index::<D>(buf, index)` | — | No Python equivalent. Loads the 1-D tile at row `index` (slices dim 0) |
| `T::shared_store_index(buf, index, src)` | — | No Python equivalent. Stores a 1-D tile into row `index` (slices dim 0) |
| `T::shared_trans::<D>(buf)` | — | No Python equivalent. Transposes a rank-2 buffer's row/column order — a view, not a copy |
| `T::sigmoid(x)` | `tl.sigmoid` | |
| `T::sin(x)` | `tl.sin` | |
| `T::softmax(x, dim, keep_dims, ieee_rounding)` | `tl.softmax` | Numerically stable. Chapter 10 |
| `T::sort(x, dim, descending)` | `tl.sort` | |
| `T::split(x)` | `tl.split` | Last dimension must be 2 |
| `T::sqrt(x)` | `tl.sqrt` | |
| `T::sqrt_rn(x)` | `tl.math.sqrt_rn` | Round to nearest |
| `T::static_assert(cond, msg)` | `tl.static_assert` | Compile time |
| `T::static_print(msg)` | `tl.static_print` | Compile time |
| `T::store(dest, src, mask, ...)` | `tl.store(ptr, val, mask=)` | Six arguments. Chapter 7 |
| `T::store_tensor_descriptor(desc, offsets, value)` | `desc.store(offsets, value)` | TMA |
| `T::sum(x, axis, keep_dims)` | `tl.sum` | Chapter 10 |
| `T::swizzle2d(i, j, size_i, size_j, size_g)` | `tl.swizzle2d` | Bank-conflict avoidance |
| `T::trans(x, dims)` | `tl.trans` | Alias for `permute` |
| `T::umulhi(x, y)` | `tl.umulhi` | High 32 bits of a `u32` product |
| `T::view(x, shape)` | `tl.view` | Order not preserved |
| `T::where_(cond, x, y)` | `tl.where` | Trailing underscore: `where` is a Rust keyword |
| `T::xor_sum(x, axis, keep_dims)` | `tl.xor_sum` | Integers |
| `T::zeros::<D>(shape)` | `tl.zeros` | |
| `T::zeros_like(x)` | `tl.zeros_like` | |

## Pointer arithmetic

Not trait methods, but you need them in every kernel:

| Rust | Python Triton |
|---|---|
| `ptr.add_offsets(offsets)` | `ptr + offsets` |
| `x.lt(y)`, `x.ge(y)`, … | `x < y`, `x >= y`, … |
| `a + b`, `a * b`, `-a` | Same operators |
| `mask_a & mask_b` | `mask_a & mask_b` |

`add_offsets` comes from the `AddOffsets` trait and the comparison methods from
`Comparison` — the two `where` clauses every kernel carries. Chapter 7.

## Things with no Rust equivalent

| Python Triton | Status |
|---|---|
| `@triton.autotune` | Does not exist. Block sizes are chosen by hand. Chapter 15 |
| `@triton.heuristics` | Does not exist |
| `num_warps=` | Not settable. Only readable from compiled PTX metadata |
| `num_stages=` | Not settable |
| `tl.constexpr` | Const generics instead. Chapter 6 |
| `@triton.jit` on a helper | A plain `fn` pointer. Chapter 13 |

The first four are recorded in
[`KNOWN-GAPS.md`](https://github.com/teenygrad/teenygrad/blob/main/books/kernels/KNOWN-GAPS.md).

## Things Rust catches that Python does not

The reason for the type parameters and the `where` clauses:

| Mistake | Python Triton | Rust |
|---|---|---|
| Loading `f32`, storing to an `f16` buffer | Runtime, or silent | Type error |
| Using an `i32` tensor as a mask | Runtime error | Type error |
| Mismatched tensor ranks | Runtime error | Type error |
| A closure as a combine function | Runtime error | Type error |
| An unsupported dtype | Runtime error | Compile error, listing what is supported |
| Wrong argument order at the launch site | Runtime, wrong numbers | **Also silent.** Chapter 21 |

The last row is the exception, and it is the one place this SDK gives up an
advantage it could have. It is the third item in
[`API-FRICTION.md`](https://github.com/teenygrad/teenygrad/blob/main/books/kernels/API-FRICTION.md).
