#!/usr/bin/env python3
"""Builds the minimal L2-normalize models for test_l2_normalize_graph.

`x / clip(||x||, 1e-12)` as FoundationStereo writes it: Shape, ReduceL2 with keepdims,
Clip, Expand, Div. Two variants, which differ only in where the reduced axis sits:

  l2norm.onnx     [1, 6, 4, 1] reducing axis 2, which is innermost in effect -- passes
  l2norm5d.onnx   [1, 2, 5, 3, 4] reducing axis 2, which has a stride -- fails

The second is the repro for teenygrad's reduce ops discarding their axes: the kernels
reduce a flattened [outer, inner] view, so a reduce over anything but the innermost axis
sums adjacent elements instead of strided ones. A normalized element cannot exceed one,
which is what the test asserts.

Writes the model, the input, and the reference next to each other:

    python3 kernels/teeny-kernels/tests/fixtures/onnx/make_l2norm_onnx.py /tmp/l2norm
    L2NORM_ONNX=/tmp/l2norm/l2norm5d.onnx L2NORM_DATA=/tmp/l2norm \
      cargo test -p teeny-kernels --features cuda,hardware \
      --test test_l2_normalize_graph -- --ignored --nocapture
"""

import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto as T
from onnx import helper as h
from onnx import numpy_helper as nh


def build(out_dir: Path, name: str, shape, axis: int, zero_at) -> None:
    x = h.make_tensor_value_info("x", T.FLOAT, list(shape))
    y = h.make_tensor_value_info("y", T.FLOAT, list(shape))
    nodes = [
        h.make_node(
            "Constant", [], ["eps"],
            value=nh.from_array(np.array(1e-12, np.float32), "eps"),
        ),
        h.make_node("Shape", ["x"], ["xshape"]),
        h.make_node("ReduceL2", ["x"], ["norm"], axes=[axis], keepdims=1),
        h.make_node("Clip", ["norm", "eps"], ["clipped"]),
        h.make_node("Expand", ["clipped", "xshape"], ["expanded"]),
        h.make_node("Div", ["x", "expanded"], ["y"]),
    ]
    model = h.make_model(
        h.make_graph(nodes, name, [x], [y]),
        opset_imports=[h.make_opsetid("", 17)],
    )
    model.ir_version = 9
    onnx.checker.check_model(model)
    onnx.save(model, out_dir / f"{name}.onnx")

    # One position's whole vector is zero, as the cost volume's padding is: its norm is
    # then the clip's floor, and every element over it must still come out zero.
    rng = np.random.default_rng(0)
    xv = (rng.standard_normal(shape).astype(np.float32) * 10)
    xv[zero_at] = 0.0
    norm = np.sqrt((xv**2).sum(axis=axis, keepdims=True))
    ref = (xv / np.maximum(norm, 1e-12)).astype(np.float32)

    xv.tofile(out_dir / f"{name}_x.f32")
    ref.tofile(out_dir / f"{name}_ref.f32")
    print(f"{name}: shape {tuple(shape)}, {xv.size} elements, "
          f"{norm.size} norms, reference max |v| {abs(ref).max():.6f}")


def main() -> None:
    out_dir = Path(sys.argv[1] if len(sys.argv) > 1 else ".")
    out_dir.mkdir(parents=True, exist_ok=True)
    # Axis 2 with a trailing 1: innermost in effect.
    build(out_dir, "l2norm", (1, 6, 4, 1), 2, (0, 2, slice(None), 0))
    # Axis 2 with two axes after it: a real stride.
    build(out_dir, "l2norm5d", (1, 2, 5, 3, 4), 2, (0, 1, slice(None), 2, 3))


if __name__ == "__main__":
    main()
