#!/usr/bin/env python3
"""Generate fixtures for Softmax/LogSoftmax over any axis (teenygrad#152).

The kernels normalize the innermost axis of a flattened [rows, cols] view. A single
axis that is not innermost (ONNX opset 13+, PyTorch) is transposed innermost and back
around the kernel; ONNX's pre-13 "flatten from axis" meaning is that view as it is.
CASES and their order must match tests/test_softmax_axis.rs.
"""

import os
import numpy as np
import torch

torch.manual_seed(42)

BASE = os.path.dirname(os.path.abspath(__file__))

# (name, shape, axis, flatten_from)
CASES = [
    # teenygrad-20il's acceptance shapes.
    ("s6x4_last", [6, 4], -1, False),
    ("s4x8_last", [4, 8], -1, False),
    ("s4x6_last", [4, 6], -1, False),
    ("s1x3x2x4_last", [1, 3, 2, 4], -1, False),
    ("s6x4_axis0", [6, 4], 0, False),
    ("s1x3x2x4_axis1", [1, 3, 2, 4], 1, False),
    # Every outer axis of a rank-4 tensor, and a rank-5 middle axis.
    ("s2x3x4x5_axis0", [2, 3, 4, 5], 0, False),
    ("s2x3x4x5_axis1", [2, 3, 4, 5], 1, False),
    ("s2x3x4x5_axis2", [2, 3, 4, 5], -2, False),
    ("s1x2x5x3x4_axis2", [1, 2, 5, 3, 4], 2, False),
    # Only extent-1 axes after the named one: no transpose needed.
    ("s2x3x1_axis1", [2, 3, 1], 1, False),
    # FoundationStereo's disparity softmax ([1, 9, 96, 128] axis 1), scaled down.
    ("s1x9x24x32_axis1", [1, 9, 24, 32], 1, False),
    # ONNX before opset 13: everything from the axis on is one row.
    ("s1x3x2x4_flatten1", [1, 3, 2, 4], 1, True),
    ("s2x3x4_flatten1", [2, 3, 4], 1, True),
]


def save(name, tensor):
    arr = np.ascontiguousarray(tensor.detach().cpu().numpy().astype(np.float32))
    arr.tofile(os.path.join(BASE, name))


for name, shape, axis, flatten_from in CASES:
    x = torch.empty(shape).uniform_(-4, 4)
    xd = x.double()
    if flatten_from:
        k = axis % len(shape)
        rows = xd.reshape(int(np.prod(shape[:k])), -1)
        sm = torch.softmax(rows, dim=-1).reshape(shape)
        lsm = torch.log_softmax(rows, dim=-1).reshape(shape)
    else:
        sm = torch.softmax(xd, dim=axis)
        lsm = torch.log_softmax(xd, dim=axis)
    save(f"x_{name}.bin", x)
    save(f"softmax_{name}.bin", sm)
    save(f"log_softmax_{name}.bin", lsm)
    print(f"  {name:24s} {shape} axis {axis}{' (flatten)' if flatten_from else ''}")
