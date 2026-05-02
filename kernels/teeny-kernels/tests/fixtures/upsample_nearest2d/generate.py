#!/usr/bin/env python3
"""Generate fixtures for upsample_nearest2d forward and backward tests.

Layout: NCHW float32.

Dimensions:
    B=2, C=4, H=4, W=4, SCALE_H=2, SCALE_W=2
    OH = H * SCALE_H = 8
    OW = W * SCALE_W = 8
"""

import os
import numpy as np
import torch
import torch.nn.functional as F

torch.manual_seed(42)

BASE = os.path.dirname(os.path.abspath(__file__))

def save(name, tensor):
    arr = tensor.detach().cpu().numpy().astype(np.float32)
    if not arr.flags["C_CONTIGUOUS"]:
        arr = np.ascontiguousarray(arr)
    arr.tofile(os.path.join(BASE, name))
    print(f"  {name:40s}  {list(arr.shape)}")

B, C, H, W = 2, 4, 4, 4
SCALE_H, SCALE_W = 2, 2
OH, OW = H * SCALE_H, W * SCALE_W

print("upsample_nearest2d forward")
x = torch.empty(B, C, H, W).uniform_(-5, 5).requires_grad_(True)
y = F.interpolate(x, scale_factor=(SCALE_H, SCALE_W), mode="nearest")
save("x.bin",               x.detach())
save("expected_forward.bin", y.detach())

print("upsample_nearest2d backward")
torch.manual_seed(43)
dy = torch.empty(B, C, OH, OW).uniform_(-5, 5)
y.backward(dy)
save("dy.bin",              dy)
save("expected_backward.bin", x.grad)
