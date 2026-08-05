# Your Kernel as a Graph Op

Every kernel so far has been launched by hand: you built it, compiled it,
allocated buffers, and called `launch`. That is the whole mechanism, and for a
standalone kernel it is all you need.

Models do not work that way. A model is a graph of operations, and the framework
decides what runs, in what order, with which buffers. To put your kernel in one,
you have to describe it in terms the graph understands.

This part is about that. It is more machinery than the rest of the book, and
none of it is needed to make a kernel *work* — only to make it a citizen of a
model.

## The two halves

There are two separate jobs, and separating them is the thing to understand
first.

**`CustomOp` is the graph-level description.** What the operation is called,
what shape it produces, and how to get a kernel out of it. It is consulted while
the graph is being built and lowered, before anything runs.

**`RuntimeOp` is the launch-time description.** How many inputs it takes, what
scratch buffers it needs, how to pack arguments, and how big a grid to launch.
It is consulted every time the op executes.

One is about shapes and identity. The other is about pointers and grids. The
next chapter is `RuntimeOp`; this one is everything before it.

## Symbolic tensors

A graph is built by recording. You start with a placeholder and every operation
on it appends a node rather than computing anything:

```rust,ignore
let (x, graph) = SymTensor::input(DtypeRepr::F32, vec![None, Some(784)]);
```

`SymTensor` is a handle: a node index, a dtype, a shape, and a shared reference
to the graph. Cloning one is cheap; it shares the graph.

The shape is a `Vec<Option<usize>>`, and the `None` is the point. It means "this
dimension is not known yet" — almost always the batch axis. So `vec![None,
Some(784)]` is "any number of rows of 784". Concrete sizes arrive later, when
the model is loaded and given real inputs.

This is why your op cannot simply be handed shapes. It has to be able to *infer*
its output shape from symbolic inputs, before anyone knows the batch size.

## Recording your op

```rust,ignore
let y = x.record_custom(CustomData::new(MyOp::new(block_size)), &[], None);
```

Three arguments:

- **the op**, wrapped in `CustomData`, which is an `Arc<dyn CustomOp>` that also
  implements `Debug`;
- **additional inputs**, since `self` is the first one — a two-input op passes
  `&[&other]`;
- **an output dtype**, or `None` to keep the primary input's.

It returns a new `SymTensor` pointing at the node it just added. From the
graph's point of view your op is now indistinguishable from a built-in one.

## Implementing `CustomOp`

Four methods matter:

```rust,ignore
pub trait CustomOp: Any + Send + Sync {
    fn name(&self) -> &str;
    fn infer_output_shape(&self, input_shapes: &[&Shape]) -> Shape;
    fn as_any(&self) -> &dyn Any;
    fn lower(&self) -> Option<(String, String, String, Arc<dyn RuntimeOp>)> { None }
    fn lower_backward_source(&self) -> String { String::new() }
}
```

**`name`** is used in errors and debug output. Namespace it — the vision-rs ops
use `"yolo.detect_decode"` — because a bare `"decode"` in a lowering failure
tells nobody anything.

**`infer_output_shape`** is the one with real content. It gets every input's
symbolic shape, in the order they were recorded, and returns the output's.
Shape-preserving ops are one line:

```rust,ignore
fn infer_output_shape(&self, input_shapes: &[&Shape]) -> Shape {
    input_shapes[0].clone()
}
```

An op that changes rank does the arithmetic here, propagating `None` wherever a
dimension stays dynamic. Getting this wrong does not fail here — it fails much
later, when a buffer is allocated at the wrong size.

**`as_any`** is boilerplate, always `fn as_any(&self) -> &dyn Any { self }`. It
lets a lowering downcast back to your concrete type.

**`lower`** is how your op becomes a kernel, and is the subject of the next
chapter.

## A real one

There is no `CustomOp` implementation in the teenygrad repository itself — the
built-in ops go through `Op` variants instead. The worked examples are in
vision-rs, and `DetectDecodeOp` is the clearest:

```rust,ignore
pub struct DetectDecodeOp<D: FloatBytes + Send + Sync + 'static> {
    pub anchor_x: Vec<f32>,
    pub anchor_y: Vec<f32>,
    pub strides: Vec<f32>,
    pub block_a: i32,
    _phantom: PhantomData<D>,
}

impl<D: FloatBytes + Send + Sync + 'static> CustomOp for DetectDecodeOp<D> {
    fn name(&self) -> &str { "yolo.detect_decode" }

    fn infer_output_shape(&self, input_shapes: &[&Shape]) -> Shape {
        // boxes [B, 4, A] → [B, 4, A]: shape-preserving
        input_shapes[0].clone()
    }

    fn as_any(&self) -> &dyn Any { self }

    // lower() — next chapter
}
```

*From `vision-rs/src/models/yolo/kernels/detect_decode.rs`.*

Notice what the struct holds: not tensors, but the *configuration* the kernel
needs — a precomputed anchor grid and a block size. A `CustomOp` is built once,
when the model is defined, and consulted many times. It should hold parameters,
never per-inference state.

## What the lowering does with it

When the graph is compiled, `TritonLowering` walks it and turns every node into
something executable. Your node hits this arm:

```rust,ignore
Op::Custom { data } => match data.0.lower() {
    Some((name, kernel_source, entry_point, runtime_op)) => {
        Box::new(KernelExecutable { name, kernel_source, entry_point, ... })
    }
    None => {
        return Err(anyhow::anyhow!(
            "custom op '{}' is not handled — implement CustomOp::lower()",
            data.name()
        ));
    }
},
```

*From `kernels/teeny-kernels/src/graph/mod.rs`.*

So `lower` returning `None` — the default — is a runtime error naming your op,
not a compile error. If you implement `CustomOp` and forget `lower`, this is the
message you will get.

The result is a `KernelExecutable`: kernel source, entry-point symbol, output
shape and dtype, and the `RuntimeOp`. From there it is compiled to PTX exactly
as in Chapter 9 — the graph path and the by-hand path converge on the same
compiler.

## Where fusion happens

Chapter 12 fused operations by writing one kernel that did several things. The
graph has its own kind: the lowering can recognise a pattern of nodes and emit a
single kernel for them, or split one node into several.

Both happen in this tree. Conv2d-with-bias becomes two DAG nodes, which is why
`Lowering::extra_dag_names` exists — the extra node needs a name so its weights
load. And the fused conv kernels from Chapter 12 are selected by shape, in the
lowering, from one graph node.

Your custom op does not participate in this. It lowers to exactly the kernel you
give it, and the graph will not fuse it with its neighbours. If you want fusion,
fuse it yourself, in the kernel.

Next: the half that runs.
