#!/usr/bin/env python3
"""Builds the ONNX corpus that kernels/teeny-kernels/tests/test_onnx_corpus.rs checks.

Each model is exported to ONNX (opset 18, static shapes), run once in onnxruntime on
fixed seeded inputs, and written with its inputs and onnxruntime's outputs:

    <out>/<name>/
        model.onnx              weights inline
        manifest.json           {"inputs": [{name, dtype, shape, file}],
                                 "outputs": [{name, dtype, shape, file}]}
        in_<i>.bin, out_<i>.bin raw little-endian tensors (f32 or i64)

The models are picked for op coverage rather than size:

    resnet18          Conv, BatchNorm (folded), Relu, MaxPool, GlobalAveragePool, Gemm
    mobilenet_v3      depthwise Conv, HardSwish/HardSigmoid, squeeze-excite Mul
    bert_tiny         Gather embedding, LayerNorm, batched MatMul, Softmax, Erf GELU,
                      attention-mask arithmetic (Unsqueeze, Cast, Sub, Mul)
    functiongemma     Gemma 3 decoder at 16 tokens, no cache: RMSNorm from Pow/ReduceMean/
                      Sqrt, RoPE from Cos/Sin/Slice/Concat, GQA Expand, causal and
                      sliding-window masks (Trilu/Where), tanh GELU

Usage (needs torch, torchvision, transformers, onnx, onnxruntime; functiongemma needs the
gated Hugging Face model):

    python export.py --out ~/.cache/teeny-onnx/corpus [--only resnet18,bert_tiny]
"""

import argparse
import json
import os

import numpy as np
import onnx
import onnxruntime as ort
import torch

OPSET = 18


def resnet18():
    import torchvision
    m = torchvision.models.resnet18(weights="IMAGENET1K_V1").eval()
    x = torch.randn(1, 3, 224, 224, generator=torch.Generator().manual_seed(0))
    return m, (x,), ["image"], ["logits"]


def mobilenet_v3():
    import torchvision
    m = torchvision.models.mobilenet_v3_small(weights="IMAGENET1K_V1").eval()
    x = torch.randn(1, 3, 224, 224, generator=torch.Generator().manual_seed(1))
    return m, (x,), ["image"], ["logits"]


class _LastHidden(torch.nn.Module):
    """Returns only the first output (hidden states or logits), as a plain tensor."""

    def __init__(self, m, key):
        super().__init__()
        self.m, self.key = m, key

    def forward(self, *args):
        return getattr(self.m(*args), self.key)


def bert_tiny():
    from transformers import AutoModel
    m = AutoModel.from_pretrained("google/bert_uncased_L-2_H-128_A-2", attn_implementation="eager").eval()
    g = torch.Generator().manual_seed(2)
    ids = torch.randint(1000, 20000, (1, 16), generator=g)
    mask = torch.ones(1, 16, dtype=torch.int64)
    mask[0, 12:] = 0
    return _LastHidden(m, "last_hidden_state"), (ids, mask), ["input_ids", "attention_mask"], ["hidden"]


def functiongemma():
    from transformers import AutoModelForCausalLM
    m = AutoModelForCausalLM.from_pretrained(
        "google/functiongemma-270m-it", dtype=torch.float32, attn_implementation="eager"
    ).eval()
    m.config.use_cache = False
    g = torch.Generator().manual_seed(3)
    ids = torch.randint(100, 50000, (1, 16), generator=g)
    ids[0, 0] = 2  # <bos>
    return _LastHidden(m, "logits"), (ids,), ["input_ids"], ["logits"]


MODELS = {
    "resnet18": resnet18,
    "mobilenet_v3": mobilenet_v3,
    "bert_tiny": bert_tiny,
    "functiongemma": functiongemma,
}

DTYPES = {np.float32: "f32", np.int64: "i64"}


def export(name, build, out):
    model, args, in_names, out_names = build()
    d = os.path.join(out, name)
    os.makedirs(d, exist_ok=True)
    path = os.path.join(d, "model.onnx")
    with torch.no_grad():
        try:
            torch.onnx.export(model, args, path, input_names=in_names, output_names=out_names,
                              opset_version=OPSET, dynamo=False, do_constant_folding=True)
        except torch.onnx.errors.UnsupportedOperatorError as e:
            # The TorchScript exporter lacks some ops newer models use (transformers' mask
            # code calls aten::diff); the dynamo exporter covers them.
            print(f"{name}: TorchScript export failed ({e}); using the dynamo exporter", flush=True)
            prog = torch.onnx.export(model, args, input_names=in_names, output_names=out_names,
                                     opset_version=OPSET, dynamo=True, optimize=True)
            prog.save(path, external_data=False)
    onnx.checker.check_model(path)

    feeds = {n: a.numpy() for n, a in zip(in_names, args)}
    so = ort.SessionOptions()
    so.log_severity_level = 3
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    sess = ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])
    outs = sess.run(None, feeds)

    manifest = {"inputs": [], "outputs": []}
    for i, n in enumerate(in_names):
        a = np.ascontiguousarray(feeds[n])
        f = f"in_{i}.bin"
        a.tofile(os.path.join(d, f))
        manifest["inputs"].append({"name": n, "dtype": DTYPES[a.dtype.type],
                                   "shape": list(a.shape), "file": f})
    for i, (n, a) in enumerate(zip([o.name for o in sess.get_outputs()], outs)):
        a = np.ascontiguousarray(a)
        f = f"out_{i}.bin"
        a.tofile(os.path.join(d, f))
        manifest["outputs"].append({"name": n, "dtype": DTYPES[a.dtype.type],
                                    "shape": list(a.shape), "file": f})
    with open(os.path.join(d, "manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=1)

    ops = sorted({n.op_type for n in onnx.load(path, load_external_data=False).graph.node})
    size = os.path.getsize(path) / 1e6
    print(f"{name}: {size:.0f} MB, {len(ops)} op types: {' '.join(ops)}", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--only", default="")
    a = p.parse_args()
    out = os.path.expanduser(a.out)
    names = [n for n in a.only.split(",") if n] or list(MODELS)
    for n in names:
        export(n, MODELS[n], out)


if __name__ == "__main__":
    main()
