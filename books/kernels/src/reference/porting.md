# Appendix: Porting a Python Triton Kernel

A worked port, start to finish. The example is a fused softmax — the same
operation as Chapter 10, approached from the other direction, because it is the
kernel most people have already written in Python.

## The Python original

Here is the shape of a typical hand-written Triton softmax. One program per row,
the whole row loaded at once, masked because rows are rarely a power of two:

```python
import triton
import triton.language as tl

@triton.jit
def softmax_kernel(x_ptr, y_ptr, n_cols, BLOCK_SIZE: tl.constexpr):
    row = tl.program_id(0)
    row_start = row * n_cols

    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < n_cols

    x = tl.load(x_ptr + row_start + cols, mask=mask, other=-float("inf"))

    x = x - tl.max(x, axis=0)
    num = tl.exp(x)
    y = num / tl.sum(num, axis=0)

    tl.store(y_ptr + row_start + cols, y, mask=mask)
```

## Step 1 — The signature

Python's `tl.constexpr` marks a compile-time constant. In Rust that is a const
generic, and you also need the two type parameters Python does not have: the
GPU, and the dtype.

```rust,ignore
pub fn softmax_forward<T: Triton, D: Float, const BLOCK_SIZE: i32>(
    x_ptr: T::Pointer<D>,
    y_ptr: T::Pointer<D>,
    n_cols: i32,
)
```

`D: Float` rather than `Num`, because `exp` is only defined for floats. Python
would have discovered that at run time, on a GPU, if at all.

Then the three `where` clauses. They are the same in every kernel:

```rust,ignore
where
    T::I32Tensor: types::Tensor<i32, 1>,
    T::I32Tensor: Comparison<i32, BoolTensor = T::BoolTensor>,
    T::Pointer<D>: AddOffsets<i32, 1, T::I32Tensor, Output = T::Tensor<T::Pointer<D>>>,
```

## Step 2 — Indices

```python
row = tl.program_id(0)
row_start = row * n_cols
cols = tl.arange(0, BLOCK_SIZE)
mask = cols < n_cols
```

```rust,ignore
let row = T::program_id(Axis::X);
let row_start = row * n_cols;
let cols = T::arange(0, BLOCK_SIZE);
let mask = cols.lt(n_cols);
```

Nearly identical. `0` becomes `Axis::X`, and `<` becomes `.lt(...)` because Rust
cannot overload comparison operators to return something other than `bool`.

## Step 3 — The load

This is where the two languages diverge most.

```python
x = tl.load(x_ptr + row_start + cols, mask=mask, other=-float("inf"))
```

```rust,ignore
let neg_inf = T::full(&[BLOCK_SIZE], D::from_f64(f64::NEG_INFINITY));
let x = T::load(
    x_ptr.add_offsets(cols + row_start),
    Some(mask),
    Some(neg_inf),
    &[],
    None,
    None,
    None,
    false,
);
```

Three differences:

- **`+` becomes `add_offsets`.** Python overloads `+` on pointers; Rust uses a
  named method from the `AddOffsets` trait.
- **The fill value is a tensor.** Python broadcasts a scalar; Rust wants a tensor
  of the right dtype, built with `T::full`.
- **Every optional argument is spelled out.** Six `None`s and a `false`. There is
  no way around it today, and it is the first item in `API-FRICTION.md`.

Note the fill value itself: negative infinity, because these lanes feed a
maximum. Chapter 10 has the table.

## Step 4 — The arithmetic

```python
x = x - tl.max(x, axis=0)
num = tl.exp(x)
y = num / tl.sum(num, axis=0)
```

```rust,ignore
let row_max = T::max(x, Some(0), true);
let shifted = x - row_max;
let num = T::exp(shifted);
let denom = T::sum(num, Some(0), true);
let y = num / denom;
```

Two differences worth care:

**`axis=0` becomes `Some(0)`.** `None` means "reduce everything", which Python
spells by omitting the argument.

**`keep_dims` is explicit, and you want `true`.** Python's default keeps the
result broadcastable against the input. Rust makes you say so, and `false` gives
a shape that will not line up with `x`. This is the most common porting mistake.

**No reassignment.** Python rebinds `x`; the Rust version uses new names because
the tensors are `Copy` values, not mutable buffers.

## Step 5 — The store

```python
tl.store(y_ptr + row_start + cols, y, mask=mask)
```

```rust,ignore
T::store(
    y_ptr.add_offsets(cols + row_start),
    y,
    Some(mask),
    &[],
    None,
    None,
);
```

## Step 6 — The launch

Python launches with a grid expression and passes the constant as a keyword:

```python
softmax_kernel[(n_rows,)](x, y, n_cols, BLOCK_SIZE=triton.next_power_of_2(n_cols))
```

Rust splits this into building, compiling and launching — Chapter 5:

```rust,ignore
let kernel = SoftmaxForward::<f32>::new(BLOCK_SIZE);
let ptx = std::fs::read(compile_kernel(&kernel, &Target::new(capability), false, false)?)?;
let program = testing::load_program_from_ptx::<SoftmaxForward<f32>>(&ptx)?;

let cfg = CudaLaunchConfig { grid: [n_rows as u32, 1, 1], block: [BLOCK_SIZE as u32, 1, 1], cluster: [1, 1, 1] };
device.launch(&program, &cfg, (
    x_buf.as_device_ptr() as *mut f32,
    y_buf.as_device_ptr() as *mut f32,
    n_cols as i32,
))?;
```

More ceremony, and one thing to be careful of: **the argument tuple is
positional and unchecked.** Python matches by name at the call site; Rust does
not. Chapter 21.

`BLOCK_SIZE` is a Rust constant now, so `next_power_of_2` happens on the host
before you pick which pre-built kernel to use — you cannot compute it per launch
without compiling a new kernel.

## What the port gained

Four errors that Python would have found at run time, or not at all:

| Mistake | Python | Rust |
|---|---|---|
| `x` is an integer tensor, `exp` undefined | Runtime error on a GPU | `D: Float` fails to compile |
| Mask built from the wrong tensor | Wrong numbers | Type error: not a `BoolTensor` |
| Storing `f32` into an `f16` buffer | Silent | Type error |
| Forgetting `other=` before a max | Wrong numbers, non-deterministic | Still silent. Chapter 10 |

Three out of four. The last is a real limitation: nothing knows that a load
feeding a reduction needs an identity fill.

## What it cost

- **Three `where` clauses**, in every kernel, identical.
- **Six `None`s per load**, four per store.
- **A compile step you now manage** rather than a decorator.
- **No autotuner.** Python would have swept block sizes for you. Chapter 15.

## A checklist

For your own ports:

1. `tl.constexpr` → const generic. Everything else → a normal argument.
2. Pick the tightest dtype bound: `Float` if you use `exp`/`log`/`sqrt`, `Int`
   for bitwise, `Num` otherwise.
3. Copy the three `where` clauses.
4. `ptr + offsets` → `ptr.add_offsets(offsets)`.
5. `<` → `.lt()`, and friends.
6. `mask=`/`other=` → the second and third arguments; pad the rest with `None`
   and `false`.
7. `axis=n` → `Some(n)`; keep `keep_dims: true` unless you know otherwise.
8. Fill masked lanes with the reduction's identity.
9. Check the launch tuple against the signature, by eye, twice.
10. Compile it and read the MLIR. Chapter 9.
