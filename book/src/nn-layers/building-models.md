# Building Models with `nn`

`teeny-core::nn` ([API docs](https://docs.teenygrad.org/api/teenygrad/teeny_core/nn/)) provides the `Layer`
trait and a set of standard layers:

- **Convolution**: `conv1d`, `conv2d`, `conv3d`
- **Normalization**: `batchnorm`, `groupnorm`, `instancenorm`, `layernorm`, `rmsnorm`
- **Other**: `activation`, `flatten`, `linear`, `pad`, `pool`

Every layer implements `Layer<I>`, with an associated `Output` type and a `call(&self, input: I)
-> Self::Output` method — this is what gets invoked during [tracing](../core-concepts/tensors-and-graph.md#tracing)
to build up a `Graph`.

Composite models (e.g. `teeny-vision::mnist::mnist_lenet5`) are just structs composing these
layers and implementing `Layer` themselves by chaining calls to their fields' `call` methods — see
`models/teeny-vision/src/mnist/` for a complete example.

> **TODO**: document the parameter-management/initialization story once it's stable enough to
> commit to in the book (constructing layers with random vs. loaded weights, interaction with
> [name scopes](../core-concepts/name-scopes.md)).
