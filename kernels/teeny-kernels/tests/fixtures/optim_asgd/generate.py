#!/usr/bin/env python3
"""Generate PyTorch fixtures for ASGD optimizer tests."""
import os
import numpy as np
import torch

torch.manual_seed(42)
BASE = os.path.dirname(os.path.abspath(__file__))

def save(name, t):
    arr = t.detach().cpu().numpy().astype(np.float32)
    if not arr.flags["C_CONTIGUOUS"]:
        arr = np.ascontiguousarray(arr)
    arr.tofile(os.path.join(BASE, name))
    print(f"  {name:40s} {arr.shape}")

N = 1024
LR, WD = 0.01, 1e-4
T0   = 10
STEP = 20   # step > t0 so averaging is active
D_AX = max(1.0, STEP - T0)   # = 10.0

print(f"asgd_step  (step={STEP}, t0={T0}, d_ax={D_AX})")
p  = torch.randn(N)
g  = torch.randn(N) * 0.1
ax = torch.randn(N) * 0.1   # running average of params

g_eff  = g + WD * p
p_new  = p - LR * g_eff
ax_new = ax + (p_new - ax) / D_AX

save("asgd_params_in.bin",  p)
save("asgd_grad.bin",       g)
save("asgd_ax_in.bin",      ax)
save("asgd_params_out.bin", p_new)
save("asgd_ax_out.bin",     ax_new)

print("done")
