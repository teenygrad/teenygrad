#!/usr/bin/env python3
"""Generate fixtures for the strided_copy_forward tests (Transpose, Slice, Expand).

Each case saves a contiguous input and torch's contiguous result for the same
operation. The case names must match tests/test_strided_copy.rs.
"""

import os
import numpy as np
import torch

torch.manual_seed(42)

BASE = os.path.dirname(os.path.abspath(__file__))


def save(name, tensor):
    arr = np.ascontiguousarray(tensor.detach().cpu().numpy().astype(np.float32))
    arr.tofile(os.path.join(BASE, name))
    print(f"  {name:40s}  {list(arr.shape)}")


def case(name, x, y):
    save(f"x_{name}.bin", x)
    save(f"expected_{name}.bin", y.contiguous())


x = torch.empty(2, 3, 4, 5).uniform_(-5, 5)
case("transpose", x, x.permute(0, 2, 3, 1))
case("slice", x, x[:, 1:3, 0:4:2, 1:5:3])
# A negative step walks an axis backwards from its last element (ONNX Slice).
case("slice_reversed", x, x.flip(1)[:, :, :, 0:5:2])

big = torch.empty(9, 33, 17).uniform_(-5, 5)
case("transpose_multi_block", big, big.permute(1, 2, 0))

src = torch.empty(1, 3, 1, 5).uniform_(-5, 5)
case("expand", src, src.expand(2, 3, 4, 5))
# Fewer input axes than output axes: the leading ones are broadcast.
low = torch.empty(3, 1, 5).uniform_(-5, 5)
case("expand_rank_up", low, low.expand(2, 3, 4, 5))
