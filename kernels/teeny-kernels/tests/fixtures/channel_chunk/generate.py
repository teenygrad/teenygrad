#!/usr/bin/env python3
"""Generate fixtures for channel_chunk forward and backward tests.

Layout: flat NC (channels-last), index = n_spatial * C + c.

Dimensions:
    N_SPATIAL = 64   (e.g. batch=2, H=4, W=8)
    C_TOTAL   = 32
    CHUNK_C   = 16   (split into 2 equal halves)
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

print("channel_chunk forward")
x = torch.empty(N_SPATIAL, C_TOTAL).uniform_(-5, 5)
save("x.bin",              x)
save("expected_chunk0.bin", x[:, 0:CHUNK_C].contiguous())
save("expected_chunk1.bin", x[:, CHUNK_C:C_TOTAL].contiguous())

print("channel_chunk backward")
dy = torch.empty(N_SPATIAL, CHUNK_C).uniform_(-5, 5)
dx = torch.zeros(N_SPATIAL, C_TOTAL)
dx[:, 0:CHUNK_C] = dy          # chunk 0 backward; channels CHUNK_C..C_TOTAL stay zero
save("dy.bin",         dy)
save("expected_dx.bin", dx)
