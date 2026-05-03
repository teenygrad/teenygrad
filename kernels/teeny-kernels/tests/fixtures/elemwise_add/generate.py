#!/usr/bin/env python3
"""Generate fixtures for the ElemwiseAdd forward and backward CUDA tests.

Forward:  out[i] = a[i] + b[i]
Backward: grad_a[i] = dy[i],  grad_b[i] = dy[i]

Files written (all flat little-endian f32):
    a.bin           — input tensor a  (N,)
    b.bin           — input tensor b  (N,)
    dy.bin          — upstream gradient  (N,)
    expected_fwd.bin  — a + b          (N,)
    expected_ga.bin   — grad_a = dy    (N,)
    expected_gb.bin   — grad_b = dy    (N,)

Usage:
    python tests/fixtures/elemwise_add/generate.py
"""

import os
import numpy as np

np.random.seed(42)

BASE = os.path.dirname(os.path.abspath(__file__))
N    = 1024


def save(name, arr):
    arr = arr.astype(np.float32)
    if not arr.flags["C_CONTIGUOUS"]:
        arr = np.ascontiguousarray(arr)
    arr.tofile(os.path.join(BASE, name))
    print(f"  {name:30s}  {list(arr.shape)}")


a  = np.random.randn(N).astype(np.float32)
b  = np.random.randn(N).astype(np.float32)
dy = np.random.randn(N).astype(np.float32)

out    = a + b
grad_a = dy.copy()
grad_b = dy.copy()

print(f"Saving fixtures to {BASE}")
save("a.bin",            a)
save("b.bin",            b)
save("dy.bin",           dy)
save("expected_fwd.bin", out)
save("expected_ga.bin",  grad_a)
save("expected_gb.bin",  grad_b)

print(f"\nN={N}  a[0]={a[0]:.4f}  b[0]={b[0]:.4f}  out[0]={out[0]:.4f}")
print("done")
