# Tensors & the Computational Graph

`teeny-core::graph` ([API docs](https://docs.teenygrad.org/api/teeny-core/graph/)) defines the
types every other crate builds on:

- **`Shape`** — `Vec<Option<usize>>`; `None` entries are symbolic/dynamic dimensions (e.g. a batch
  dimension left unbound until trace time).
- **`DtypeRepr`** — the element dtype of a tensor (`F32`, etc. — see
  [The Dtype System](./dtype-system.md)).
- **`Op`** — the set of graph operations (elementwise, reductions, layout, etc.).
- **`GraphNode`** / **`Graph`** — the traced computational graph: nodes are `Op` applications,
  edges are data dependencies.
- **`SymTensor`** — a *symbolic* tensor handle used while tracing a model: calling layers with a
  `SymTensor` input records `Op`s into a `Graph` rather than executing eagerly.

## Tracing

Building a `Graph` from a model is "tracing": construct a `SymTensor::input(dtype, shape)`, call
your model with it (via the `nn::Layer` trait), and read back the `Graph` that was recorded as a
side effect. See [`teeny-cli`](https://docs.teenygrad.org/api/teeny-cli/)'s `aot_compile` for a
concrete example of this pattern.

What happens to the `Graph` next is covered in [Compilation Flow](./compilation-flow.md).
