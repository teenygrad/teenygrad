#!/usr/bin/env python3
"""Generate PyTorch fixtures for Rprop optimizer tests."""
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
ETA_PLUS, ETA_MINUS = 1.2, 0.5
STEP_MIN, STEP_MAX  = 1e-6, 50.0

print("rprop_step")
p         = torch.randn(N)
g         = torch.randn(N) * 0.1
prev_g    = torch.randn(N) * 0.1   # previous gradient
step_size = torch.full((N,), 0.01) # initial per-param step sizes

prod      = g * prev_g
sign_pos  = prod > 0
sign_neg  = prod < 0

step_after_pos = torch.where(sign_pos, step_size * ETA_PLUS,  step_size)
step_scaled    = torch.where(sign_neg, step_after_pos * ETA_MINUS, step_after_pos)
step_clamped   = torch.clamp(step_scaled, STEP_MIN, STEP_MAX)

g_masked  = torch.where(sign_neg, torch.zeros_like(g), g)
g_sign    = torch.sign(g_masked)
p_new     = p - g_sign * step_clamped

save("rprop_params_in.bin",    p)
save("rprop_grad.bin",         g)
save("rprop_prev_grad_in.bin", prev_g)
save("rprop_step_size_in.bin", step_size)
save("rprop_params_out.bin",   p_new)
save("rprop_step_size_out.bin",step_clamped)
save("rprop_prev_grad_out.bin",g_masked)

print("done")
