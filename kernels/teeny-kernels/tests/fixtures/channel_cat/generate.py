#!/usr/bin/env python3
"""Generate fixtures for channel_cat forward and backward tests.

Layout: flat NC (channels-last), index = n_spatial * C + c.

Dimensions:
    N_SPATIAL = 64   (e.g. batch=2, H=4, W=8)
    C_TOTAL   = 32
    CHUNK_C   = 16   (two equal chunks concatenated)
"""

import os
import numpy as np
import torch

torch.manual_seed(42)

BASE = os.path.dirname(os.path.abspath(__file__))

def save(name, tensor):
    arr = tensor.detach().cpu().numpy().astype(np.float32)
    if not arr.flags["C_CONTIGUOUS"]:
        arr = np.ascontiguousarray(arr)
    arr.tofile(os.path.join(BASE, name))
    print(f"  {name:40s}  {list(arr.shape)}")

N_SPATIAL = 64
C_TOTAL   = 32
CHUNK_C   = 16

print("channel_cat forward")
x0 = torch.empty(N_SPATIAL, CHUNK_C).uniform_(-5, 5)
x1 = torch.empty(N_SPATIAL, CHUNK_C).uniform_(-5, 5)
y  = torch.cat([x0, x1], dim=1)
save("x0.bin",          x0)
save("x1.bin",          x1)
save("expected_cat.bin", y)

print("channel_cat backward")
dy = torch.empty(N_SPATIAL, C_TOTAL).uniform_(-5, 5)
save("dy.bin",           dy)
save("expected_dx0.bin", dy[:, 0:CHUNK_C].contiguous())
save("expected_dx1.bin", dy[:, CHUNK_C:C_TOTAL].contiguous())
