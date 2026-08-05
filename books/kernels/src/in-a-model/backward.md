# Training: The Backward Kernel

Everything so far runs a model forwards. Training needs it to run backwards
too — and a custom op that cannot produce gradients is a wall across the middle
of your network.

## What a backward pass needs

Training works by chain rule. Each operation is asked: given how much the loss
changes with respect to *your output*, how much does it change with respect to
your *inputs*?

So a backward kernel takes the upstream gradient `dy` and produces `dx` — one
per input the forward pass consumed, plus one per parameter that gets trained.

Two consequences shape everything below.

**It needs values from the forward pass.** Most gradients depend on what went in
or came out. So the executor keeps the forward activations and hands them back.

**The shapes are mirrored.** Where forward went inputs → output, backward goes
output-gradient → input-gradients. A forward with two inputs has a backward that
writes two buffers.

## Declaring the pair

```rust
{{#include ../../../../kernels/teeny-kernels/src/nn/activation/gelu.rs:gelu_forward}}
```

`#[kernel(backward = GeluBackward)]` names the struct that computes this
kernel's gradient. `GeluBackward` is generated from another `#[kernel]` function
in the same file, exactly like the forward one — there is no separate macro for
backward kernels.

This attribute is used throughout the tree: every activation with a derivative,
the elementwise ops, the losses. It also opts the kernel into dtype dispatch,
which is where the implicit `f32`/`f64` set from Chapter 15 comes from.

## Writing the gradient

The simplest case is addition, where the gradient flows through unchanged to
both inputs:

```rust,ignore
// grad_a[i] = dy[i],  grad_b[i] = dy[i]
let dy = T::load(dy_ptr.add_offsets(offsets), Some(in_bounds), ...);
T::store(grad_a_ptr.add_offsets(offsets), dy, Some(in_bounds), &[], None, None);
T::store(grad_b_ptr.add_offsets(offsets), dy, Some(in_bounds), &[], None, None);
```

*From `kernels/teeny-kernels/src/nn/tensor/elemwise_add.rs`.*

Softmax is the more representative shape. Its gradient needs the forward
*output*, not the input:

```text
dx_i = y_i * (dy_i - sum_j(y_j * dy_j))
```

so `softmax_backward` takes `dy_ptr`, `y_ptr` and `dx_ptr`. The saved `y` is
what the executor hands back, and the inner sum is a row-wide reduction —
Chapter 10's machinery, in a backward kernel.

That is the usual pattern: a backward kernel is a normal kernel with more
pointer arguments.

## Wiring it to the runtime

`RuntimeOp` has a parallel set of methods for the backward pass, all behind the
`training` cargo feature:

```rust,ignore
#[cfg(feature = "training")]
fn has_backward(&self) -> bool { true }

#[cfg(feature = "training")]
fn pack_backward_args(
    &self,
    inputs: &[(RawPtr, &[usize])],   // forward inputs, from the activation cache
    params: &[RawPtr],               // forward parameters
    output: RawPtr,                  // forward output — the saved `y`
    output_shape: &[usize],
    grad_output: RawPtr,             // dy, from the consumer
    grad_output_row_stride: i32,
    grad_inputs: &[RawPtr],          // dx, one per activation input — you write these
    grad_params: &[RawPtr],          // dW, one per parameter
    visitor: &mut dyn ArgVisitor,
);

#[cfg(feature = "training")]
fn backward_block(&self) -> [u32; 3];

#[cfg(feature = "training")]
fn backward_grid(&self, input_shapes: &[&[usize]], output_shape: &[usize]) -> [u32; 3];
```

Same contract as Chapter 21, same absence of checking, with more buffers to get
in the right order.

Two defaults are worth knowing because they fail quietly:

- **`has_backward` defaults to `false`.** Forget it and your op contributes no
  gradient. Nothing errors; the parameters upstream of it simply never learn.
- **`backward_grid` defaults to `[0, 0, 0]`.** A grid of zero launches no
  programs. Also silent, also produces zeros.

If gradients are zero everywhere behind your op, check those two first.

The elementwise-add op shows the whole shape:

```rust,ignore
#[cfg(feature = "training")]
fn pack_backward_args(&self, ..., grad_output, ..., grad_inputs, ..., visitor) {
    let n: usize = output_shape.iter().product();
    visitor.visit_ptr(grad_output);    // dy_ptr
    visitor.visit_ptr(grad_inputs[0]); // grad_a_ptr
    visitor.visit_ptr(grad_inputs[1]); // grad_b_ptr
    visitor.visit_i32(n as i32);       // n_elements
}
```

## For a custom op

A `CustomOp` supplies its backward source through a separate method:

```rust,ignore
fn lower_backward_source(&self) -> String {
    MyOpBackward::<D>::new(self.block_size).source.clone()
}
```

The default returns an empty string, which the lowering reads as "no backward",
so an op without it is inference-only.

## Inference and training modes

The lowering is told which it is building for:

```rust,ignore
pub enum LoweringMode {
    Inference,  // default — no backward kernels
    Training,
}
```

In `Inference` the backward kernels are never compiled, which is the right
default: a deployed model should not pay for machinery it will not use.

The `training` cargo feature is the compile-time half of the same distinction.
It gates the backward methods on `RuntimeOp` entirely, so the trait's shape
changes with your feature flags. Build without it and `has_backward` does not
exist to be overridden.

## Checking a gradient

An analytically-derived gradient with a sign error still produces plausible
numbers, and a model that trains slightly worse than it should is an
extraordinarily hard bug to find later.

Check it numerically instead. The derivative is a limit, so approximate it:

```text
dx_i  ≈  (f(x + h·e_i) - f(x - h·e_i)) / 2h
```

Perturb one input by a small `h`, run the forward pass twice, and compare
against what your backward kernel produced. With `h` around `1e-3` in `f32` this
agrees to a few decimal places. Too small and rounding dominates; too large and
the approximation does.

Do it once per backward kernel, on a small input, and keep it as a test. Chapter
14's warning applies: if your backward uses atomics it is not bit-reproducible,
so compare with a tolerance.

Next: building all of this for a machine you do not have in front of you.
