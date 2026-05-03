#!/usr/bin/env python3
"""Generate fixtures for channel_bias_add forward/backward tests.

Layout: NC flat (N=B*H*W, C=channels).

Dimensions: N_SPATIAL=64 (B=2,H=4,W=8), C=16.
"""
import os
import numpy as np

np.random.seed(42)
BASE = os.path.dirname(os.path.abspath(__file__))

N_SPATIAL = 64   # B*H*W
C         = 16

x    = np.random.randn(N_SPATIAL, C).astype(np.float32)
bias = np.random.randn(C).astype(np.float32)

# Forward: y = x + bias (broadcast)
y = x + bias[np.newaxis, :]

# Backward input: dy (upstream gradient)
dy = np.random.randn(N_SPATIAL, C).astype(np.float32)
# dx = dy (identity)
dx = dy.copy()
# dbias[c] = sum over N of dy[:, c]
dbias = dy.sum(axis=0)

def save(name, arr):
    path = os.path.join(BASE, name)
    arr.astype(np.float32).flatten(order='C').tofile(path)
    print(f"  {name:30s}  shape={list(arr.shape)}")

save("x.bin",        x)
save("bias.bin",     bias)
save("expected_y.bin", y)
save("dy.bin",       dy)
save("expected_dx.bin", dx)
save("expected_dbias.bin", dbias)
print("done")
