# Node Names

A graph can name its nodes: `Graph::names` maps a node's index to a name. Names are how
weights find their way into a compiled model, and they label nodes in traces and debug
output. teeny-core only stores them; whoever builds the graph decides what they are.

- **The ONNX importer** (`teeny-onnx`) names nodes after the model's own node and initializer
  names.
- **A model's graph builder** names the nodes that hold its weights after the checkpoint's
  tensors, with `SymTensor::named`:

```rust,ignore
let q = h
    .record_custom(CustomData::new(Linear4Op::<f32>::new(hidden, heads * head_dim, 128)), &[], None)
    .named("model.layers.0.self_attn.q_proj");
```

- **Optimizer passes and lowerings** carry names over when they rewrite or split nodes, so
  every compiled node that holds parameters still has its name.

A compiled model then lists each parameter slot as `<node name>.<slot name>`
(`param_info_named`, e.g. `model.layers.0.self_attn.q_proj.weight`), and the model's loader
looks each one up in the checkpoint. teeny-llm's FunctionGemma builds its names with a small
`Scope` value passed down its graph builder (`model` → `layers.3` → `self_attn`), producing
Hugging Face's layout, and its loader refuses a checkpoint with tensors no slot took.
