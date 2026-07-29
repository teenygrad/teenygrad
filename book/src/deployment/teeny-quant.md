# `teeny-quant` and Model Quantization

[`teeny-quant`](https://docs.teenygrad.org/api/teeny-quant/) quantizes `.safetensors` model
checkpoints for deployment — smaller weights, faster inference — as both a reusable library and a
`teeny-quant` binary. It's initially being validated against Ultralytics YOLO models.

> ⚠️ **Weight-only today.** `teeny-quant` currently does post-training quantization from the
> checkpoint's weights alone. Static *activation* quantization (calibrating scales from a forward
> pass over sample inputs, TensorRT-INT8-calibration style) is planned but not implemented yet —
> it needs a model forward pass, which means running an ONNX export through
> [`teeny-onnx`](https://docs.teenygrad.org/api/teeny-onnx/) and `teeny-compiler`'s `ndarray`
> backend, and op coverage there hasn't been verified for a CNN detection model's op set (`Conv`,
> `BatchNormalization`, `Concat`, ...).

## Schemes and granularity

Three quantization schemes:

- **INT8** — symmetric or asymmetric affine quantization.
- **INT4** — same affine math as INT8, nibble-packed two values per `U8` byte (`safetensors` has
  no native 4-bit dtype).
- **FP8** — `F8_E4M3` or `F8_E5M2`, both natively supported `safetensors` dtypes.

Independently of scheme, pick a **granularity**: `tensor` (one scale for the whole tensor),
`channel` (one scale per index along an axis, reduced over every other axis — the usual choice
for per-output-channel weight quantization), or `group` (GPTQ/AWQ-style: subdivide one axis into
fixed-size chunks, with every other axis getting its own independent set of groups). `channel` and
`group` are *not* the same iteration pattern — see the crate's `quant::groups` module docs if
you're calling the library directly rather than the CLI.

## CLI

```bash
# INT8, per-channel (the default granularity).
cargo run -p teeny-quant --bin teeny-quant -- quantize \
  --input model.safetensors --output model-int8.safetensors --scheme int8

# INT4, group-wise along the reduction axis (axis 1 for a [out, in] weight matrix).
cargo run -p teeny-quant --bin teeny-quant -- quantize \
  --input model.safetensors --output model-int4.safetensors \
  --scheme int4 --granularity group --axis 1 --group-size 128

# What's in a checkpoint (plain or quantized) -- tensor names/dtypes/shapes, plus the
# quantization_config if present.
cargo run -p teeny-quant --bin teeny-quant -- inspect model-int8.safetensors

# Per-tensor quantization error: max abs error, mean abs error, SQNR (dB).
cargo run -p teeny-quant --bin teeny-quant -- validate \
  --original model.safetensors --quantized model-int8.safetensors
```

Tensors with rank < 2 (biases, norm weights) are left unquantized and listed in the output's
`ignore` metadata, rather than quantized — the usual default for PTQ tooling.

## Output format

Output follows the [compressed-tensors](https://github.com/vllm-project/compressed-tensors)
convention layered on plain `.safetensors`, so quantized checkpoints stay loadable by existing
HF/vLLM-side tooling for INT8/FP8: a quantized `foo.weight` keeps its name, gains a
`foo.weight_scale` (and `foo.weight_zero_point` for asymmetric schemes) sibling tensor, and the
file's `__metadata__` header carries a `quantization_config` JSON blob describing the scheme,
granularity, and which tensors were left unquantized. INT4's nibble-packing isn't bit-compatible
with compressed-tensors' own int32-based `pack-quantized` layout — that's `teeny-quant`'s own,
documented convention (see `quant::pack4`).

## Relationship to `teeny-core`'s dtype system

[`teeny-core::dtype`](../core-concepts/dtype-system.md) defines `F8E4M3FN`/`BF16`/`I4` as marker
traits for a future typed kernel dtype system, but has no concrete implementations yet.
`teeny-quant` doesn't depend on or wait for that — it works directly on raw `safetensors` bytes
(`safetensors::Dtype`, not `teeny_core::dtype::Dtype`), including its own from-scratch FP8
bit-conversion codec, since neither `teeny-core` nor the `half` crate (f16/bf16 only) has one.
