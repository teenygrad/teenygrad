# Your First Model

The canonical end-to-end example in the workspace today is
[`teeny-vision`](https://docs.teenygrad.org/api/teenygrad/teeny_vision/)'s LeNet-5 over MNIST:

```rust,ignore
use teeny_vision::mnist::mnist_lenet5;

let model = mnist_lenet5::<f32>();
```

That model is used in two different ways elsewhere in the workspace, both worth reading as
reference implementations:

- **Training/inference example**: `models/teeny-vision/examples/mnist.rs` — a full,
  runnable example (`cargo run --example mnist`, with `TEENYC_PATH` set — see
  [Installation](./installation.md)).
- **Ahead-of-time kernel compilation**: `utilities/teeny-cli/src/main.rs` — traces the model with
  a symbolic input and AOT-compiles its kernels via
  [`teeny_cli::aot_compile`](https://docs.teenygrad.org/api/teenygrad/teeny_cli/).

## The shape of a model

Models are built from `teeny-core`'s [`nn`](https://docs.teenygrad.org/api/teenygrad/teeny_core/nn/) layer
types, implementing the `Layer` trait. A traced model produces a symbolic computational graph
(`SymTensor` in, `Graph` out) — see [Tensors & the Computational Graph](../core-concepts/tensors-and-graph.md)
and [Compilation Flow](../core-concepts/compilation-flow.md) for what happens to that graph next.

> **TODO**: walk through building a small model from scratch (not just pointing at the existing
> LeNet-5 example) once the `nn` layer API stabilizes enough to commit to in a book.
