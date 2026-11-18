#!/usr/bin/env python3
"""Generate fixtures for reductions over axes that are not innermost (teenygrad#151).

The lowering transposes such axes innermost before the [outer, inner] reduction
kernels run, so these check that composition against PyTorch. One input of SHAPE is
reduced over each entry of AXES by each op, and every op's outputs are saved
concatenated, in AXES order, to expected_<op>.bin. AXES and the ops must match
tests/test_reduce_axes.rs.

The input is in [0.5, 2) so that ReduceLogSum's log and ReduceProd stay finite.
"""

import os
import numpy as np
import torch

torch.manual_seed(42)

BASE = os.path.dirname(os.path.abspath(__file__))

SHAPE = [2, 3, 4, 5]
AXES = [[0], [1], [2], [0, 2], [1, 3], [0, 1, 2], [0, 3]]

OPS = {
    "reduce_sum": lambda x, a: x.sum(dim=a, keepdim=True),
    "reduce_mean": lambda x, a: x.mean(dim=a, keepdim=True),
    "reduce_max": lambda x, a: x.amax(dim=a, keepdim=True),
    "reduce_min": lambda x, a: x.amin(dim=a, keepdim=True),
    "reduce_prod": lambda x, a: prod(x, a),
    "reduce_l1": lambda x, a: x.abs().sum(dim=a, keepdim=True),
    "reduce_l2": lambda x, a: x.square().sum(dim=a, keepdim=True).sqrt(),
    "reduce_log_sum": lambda x, a: x.sum(dim=a, keepdim=True).log(),
    "reduce_log_sum_exp": lambda x, a: x.logsumexp(dim=a, keepdim=True),
    "reduce_sum_square": lambda x, a: x.square().sum(dim=a, keepdim=True),
}


def prod(x, axes):
    # torch.prod takes a single dim.
    for a in sorted(axes, reverse=True):
        x = x.prod(dim=a, keepdim=True)
    return x


def save(name, tensor):
    arr = np.ascontiguousarray(tensor.detach().cpu().numpy().astype(np.float32))
    arr.tofile(os.path.join(BASE, name))
    print(f"  {name:40s}  {list(arr.shape)}")


x = torch.empty(SHAPE, dtype=torch.float64).uniform_(0.5, 2.0).float().double()  # f32 values, f64 math
save("x.bin", x)
for op, f in OPS.items():
    save(f"expected_{op}.bin", torch.cat([f(x, a).flatten() for a in AXES]))

# teenygrad#151's repro: L2 norm over axis 2 of [1, 2, 5, 3, 4], the layout of
# FoundationStereo's feature normalization.
l2 = torch.empty([1, 2, 5, 3, 4], dtype=torch.float64).uniform_(-3.0, 3.0).float().double()
save("x_l2norm5d.bin", l2)
save("expected_l2norm5d.bin", l2.square().sum(dim=2, keepdim=True).sqrt())
