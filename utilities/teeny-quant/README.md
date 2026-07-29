# teeny-quant

Weight-only post-training quantization for [teenygrad](https://teenygrad.org): reads a
`.safetensors` model checkpoint and writes a quantized one (INT8, INT4, or FP8), following the
[compressed-tensors](https://github.com/vllm-project/compressed-tensors) convention so the output
stays loadable by existing HF/vLLM-side tooling. Initially validated against Ultralytics YOLO
models.

Static activation quantization (calibrating scales from a forward pass over sample inputs, rather
than from the weights alone) is separate, larger scope tracked as `teenygrad-303.10` -- it depends
on running a model's ONNX export through `teeny-onnx`/`teeny-compiler`, which isn't wired up yet.

## Prerequisites

- **Rust**: any stable or nightly toolchain. No system-library or custom-compiler dependencies.

## Getting started

```toml
[dependencies]
teeny-quant = "0"
```

Or run the binary:

```bash
# INT8, per-channel (the default granularity).
cargo run -p teeny-quant --bin teeny-quant -- quantize \
  --input model.safetensors --output model-int8.safetensors --scheme int8

# INT4, group-wise along the reduction axis (axis 1 for a [out, in] weight matrix).
cargo run -p teeny-quant --bin teeny-quant -- quantize \
  --input model.safetensors --output model-int4.safetensors \
  --scheme int4 --granularity group --axis 1 --group-size 128

# What's in a checkpoint (plain or quantized).
cargo run -p teeny-quant --bin teeny-quant -- inspect model-int8.safetensors

# Per-tensor quantization error (max abs error, mean abs error, SQNR).
cargo run -p teeny-quant --bin teeny-quant -- validate \
  --original model.safetensors --quantized model-int8.safetensors
```

See the `quant`, `format`, and `validate` modules for the library API.

## License

Apache-2.0. See [LICENSE-APACHE](https://github.com/teenygrad/teenygrad/blob/main/LICENSE-APACHE).
