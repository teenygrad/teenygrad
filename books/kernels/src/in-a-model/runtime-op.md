# Wiring the Runtime

`CustomOp` described your operation to the graph. `RuntimeOp` describes it to
the thing that launches kernels: how many inputs, what buffers, which arguments
in what order, and how big a grid.

The trait has many methods. You need four.

## The four

```rust,ignore
fn n_activation_inputs(&self) -> usize;
fn param_shapes(&self, input_shapes: &[&[usize]], output_shape: &[usize]) -> Vec<Vec<usize>>;
fn pack_args(&self, inputs, params, output, output_shape, output_row_stride, visitor);
fn grid(&self, output_shape: &[usize]) -> [u32; 3];
```

Everything else has a default that is correct for a single-launch, inference-only
op.

Note the shapes here are `&[usize]`, not the `Shape` of the last chapter. By the
time `RuntimeOp` is consulted the batch dimension is known, so there are no
`None`s left. Symbolic shapes are a graph-construction concern; runtime shapes
are concrete.

## Inputs versus parameters

The distinction matters and is easy to get backwards.

An **activation input** comes from another node in the graph. It changes every
inference. `n_activation_inputs` says how many your op consumes, and they arrive
in `pack_args` as `inputs`, in recording order.

A **parameter** is a buffer your op owns. Weights, biases, lookup tables,
precomputed geometry. It is allocated once when the model loads and persists.
`param_shapes` declares them; they arrive as `params`.

`param_shapes` receives concrete shapes, so a parameter can be sized from the
input:

```rust,ignore
fn param_shapes(&self, input_shapes: &[&[usize]], _output_shape: &[usize]) -> Vec<Vec<usize>> {
    let a = input_shapes[0][2];          // boxes is [B, 4, A]
    vec![vec![a], vec![a], vec![a]]      // anchor_x, anchor_y, strides
}
```

Parameters are zero-initialised by default, because they are usually trained
weights loaded from a checkpoint. When yours is a constant you computed on the
host, `param_init_data` uploads it:

```rust,ignore
fn param_init_data(&self, param_idx: usize) -> Option<Vec<u8>> {
    let data: &[f32] = match param_idx {
        0 => &self.anchor_x,
        1 => &self.anchor_y,
        2 => &self.strides,
        _ => return None,
    };
    Some(data.iter().flat_map(|&f| D::from_f64(f as f64).to_le_bytes()).collect())
}
```

*From `vision-rs/src/models/yolo/kernels/detect_decode.rs`.*

Little-endian bytes, in the buffer's element type — which is why that conversion
goes through `D::from_f64` rather than writing `f32` bytes directly. The byte
count must match `param_shapes()[idx]`'s product times the dtype size, and
nothing checks it.

If a parameter needs a name, for loading from a checkpoint under a dotted key,
`param_names` supplies one per slot.

## Packing arguments

This is where your op meets your kernel, and it is the part with no safety net.

```rust,ignore
fn pack_args(
    &self,
    inputs: &[(RawPtr, &[usize])],
    params: &[RawPtr],
    output: RawPtr,
    output_shape: &[usize],
    _output_row_stride: i32,
    visitor: &mut dyn ArgVisitor,
) {
    let b = output_shape[0] as i32;
    let a = output_shape[2] as i32;
    visitor.visit_ptr(inputs[0].0); // boxes_ptr
    visitor.visit_ptr(params[0]);   // anchor_x_ptr
    visitor.visit_ptr(params[1]);   // anchor_y_ptr
    visitor.visit_ptr(params[2]);   // strides_ptr
    visitor.visit_ptr(output);      // out_ptr
    visitor.visit_i32(b);           // _B
    visitor.visit_i32(a);           // A
}
```

Seven calls, in exactly the order the kernel declares its parameters. The
trailing comments are load-bearing: they are the only thing connecting this
sequence to the function signature.

**Nothing checks it.** Not the order, not the count, not the types. Swap two
`visit_ptr` calls and you get wrong numbers, silently. Pass six arguments to a
seven-parameter kernel and the seventh is whatever was in that register.

The macro already knows the right answer — it generates `type Args<'a> = (*mut
f32, ..., i32)` for the by-hand launch path — and that knowledge is simply not
used here. It is the third item in
[`API-FRICTION.md`](https://github.com/teenygrad/teenygrad/blob/main/books/kernels/API-FRICTION.md).

Until that changes: write `pack_args` immediately after the kernel signature,
keep the comments, and make the first test one that checks numbers rather than
that it runs.

## Grid (and where threads come from)

```rust,ignore
fn grid(&self, output_shape: &[usize]) -> [u32; 3] {
    let b = output_shape[0];
    let a = output_shape[2];
    let a_tiles = a.div_ceil(self.block_a as usize);
    [(b * a_tiles) as u32, 1, 1]
}
```

`grid` is how many programs to launch — Chapter 6's division, rounded up, with
the real output shape. Compute it from the *data* one program covers (your
`BLOCK_SIZE` / tile consts), not from CUDA thread counts.

Threads per program are not part of `RuntimeOp`. `teenyc` picks them and writes
`.reqntid` into the PTX; the CUDA executor reads
`program.metadata.threads_per_block()` at launch. For a tiled kernel the split
is especially clear: `conv2d_bn_silu`'s `BLOCK_OW` is 16 (tile width) while the
compiled kernel typically runs with 128 threads. Chapter 16 pulls those apart.

`detect_decode` launches a flat grid over both the batch and the anchor tiles,
which the kernel then splits apart with a divide and a remainder — the pattern
from Chapter 6.

## Multi-launch ops

Some operations need several launches. Channel-concatenation scatters N input
chunks into one output buffer and wants one launch per chunk.

Override `n_launches`, and the executor calls `pack_args_for_launch` and
`grid_for_launch` with the index instead:

```rust,ignore
fn n_launches(&self) -> usize { self.n_chunks }

fn pack_args_for_launch(&self, launch_idx: usize, inputs, params, output, ...) {
    // pack for chunk `launch_idx`
}
```

The defaults delegate to `pack_args` and `grid`, so an op with one launch never
sees these.

## Row stride

`pack_args` receives `output_row_stride`, which is not always the last dimension
of the shape.

The default is the natural row-major stride. But a kernel using tensor
descriptors — the TMA path from Chapter 11 — needs rows aligned to 16 bytes,
which for `f32` means a multiple of 4 elements. Override
`forward_output_row_stride` to round up, and the executor allocates the padded
buffer and tells you the real stride.

Use the argument, not `output_shape.last()`, or a TMA kernel will read the wrong
addresses.

## The entry point, and an open question

`CustomOp::lower` returns four things:

```rust,ignore
fn lower(&self) -> Option<(String, String, String, Arc<dyn RuntimeOp>)> {
    let kernel = DetectDecodeForward::<D>::new(self.block_a);
    let runtime_op = Arc::new(DetectDecodeRuntimeOp::<D>::new(...));
    Some((
        "detect_decode_forward".to_string(),  // name
        kernel.source,                        // source
        "entry_point".to_string(),            // entry-point symbol
        runtime_op,
    ))
}
```

The third element is meant to be the PTX symbol to resolve, and the tree is
inconsistent about it. The trait's documentation says "conventionally
`{name}_entry_point`", which is what every built-in op in `teeny-kernels`
produces. Every `CustomOp` in vision-rs passes the bare literal
`"entry_point"`.

**It does not matter, because the value is never read.** `LoadedModel` resolves
kernels with `CudaProgram::<ErasedKernel>::try_from_ptx(&ptx)`, an overload that
takes no entry-point argument — it parses the symbol out of the compiled PTX's
`.visible .entry` directive instead. The only remaining consumer of the field
anywhere in the workspace is a `println!` in a test.

So pass whatever you like and it will work. Pass
`format!("{}_entry_point", name)` anyway: it matches what the macro actually
emits, it matches `Kernel::entry_point_name()`, and it is what will still be
right if the field is ever wired up.

This is worth knowing rather than just worth ignoring. A parameter that looks
load-bearing and is not is a trap for the next person, which is why it stays in
[`KNOWN-GAPS.md`](https://github.com/teenygrad/teenygrad/blob/main/books/kernels/KNOWN-GAPS.md)
as item 1 with a suggested fix — delete it and derive it from the name.

Next: making the op differentiable.
