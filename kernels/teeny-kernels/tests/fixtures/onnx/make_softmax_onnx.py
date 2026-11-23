#!/usr/bin/env python3
"""Builds the Softmax models for test_softmax_graph.

Each is a single `Softmax` node. They differ only in rank and axis, which separates the
two faults in teenygrad's lowering of it:

  softmax2d.onnx        [6, 4] axis -1    passes: the grid and the axis both happen to be right
  softmax2d_axis0.onnx  [6, 4] axis 0     the grid is right, the axis is not
  softmax4d_last.onnx   [1, 3, 2, 4] -1   the axis is right, the grid is not
  softmax4d_mid.onnx    [1, 3, 2, 4] 1    neither is right

`SoftmaxRuntimeOp::grid` is `[output_shape[0], 1, 1]`, which is one CTA per entry of the
first axis rather than one per row, so any rank above 2 leaves most of the output
unwritten. The lowering takes `n_cols` from the last axis and drops `Op::Softmax`'s
`dim`, so a softmax over anything else is computed over the last axis instead.

    python3 kernels/teeny-kernels/tests/fixtures/onnx/make_softmax_onnx.py /tmp/softmax
    SOFTMAX_ONNX=/tmp/softmax/softmax4d_mid.onnx SOFTMAX_DATA=/tmp/softmax \
      cargo test -p teeny-kernels --features cuda,hardware \
      --test test_softmax_graph -- --ignored --nocapture
"""

import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto as T
from onnx import helper as h


def softmax(x: np.ndarray, axis: int) -> np.ndarray:
    shifted = x - x.max(axis=axis, keepdims=True)
    e = np.exp(shifted)
    return e / e.sum(axis=axis, keepdims=True)


def build(out_dir: Path, name: str, shape: tuple[int, ...], axis: int) -> None:
    x = h.make_tensor_value_info("x", T.FLOAT, list(shape))
    y = h.make_tensor_value_info("y", T.FLOAT, list(shape))
    model = h.make_model(
        h.make_graph([h.make_node("Softmax", ["x"], ["y"], axis=axis)], name, [x], [y]),
        opset_imports=[h.make_opsetid("", 17)],
    )
    model.ir_version = 9
    onnx.checker.check_model(model)
    onnx.save(model, out_dir / f"{name}.onnx")

    rng = np.random.default_rng(0)
    xv = (rng.standard_normal(shape).astype(np.float32) * 2)
    ref = softmax(xv, axis if axis >= 0 else axis + len(shape)).astype(np.float32)
    xv.tofile(out_dir / f"{name}_x.f32")
    ref.tofile(out_dir / f"{name}_ref.f32")

    rows = int(np.prod(shape[:-1]))
    print(
        f"{name}: shape {shape} axis {axis}, {xv.size} elements, "
        f"{rows} rows of {shape[-1]} in a last-axis view, shape[0] = {shape[0]}, "
        f"reference min {ref.min():.6f}"
    )


def main() -> None:
    out_dir = Path(sys.argv[1] if len(sys.argv) > 1 else ".")
    out_dir.mkdir(parents=True, exist_ok=True)
    build(out_dir, "softmax2d", (6, 4), -1)
    build(out_dir, "softmax2d_axis0", (6, 4), 0)
    build(out_dir, "softmax4d_last", (1, 3, 2, 4), -1)
    build(out_dir, "softmax4d_mid", (1, 3, 2, 4), 1)


if __name__ == "__main__":
    main()
