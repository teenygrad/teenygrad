#!/usr/bin/env python3
"""Generate PyTorch fixtures for Adamax optimizer tests."""
import os, math
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
LR, BETA1, BETA2, EPS, WD, STEP = 0.002, 0.9, 0.999, 1e-8, 1e-4, 5

print("adamax_step")
p       = torch.randn(N)
g       = torch.randn(N) * 0.1
exp_avg = torch.randn(N) * 0.01
exp_inf = torch.abs(torch.randn(N) * 0.01) + 1e-6  # exp_inf (must be > 0)

bias_c1 = 1 - BETA1 ** STEP
clr     = LR / bias_c1

g_eff       = g + WD * p
exp_avg_new = BETA1 * exp_avg + (1 - BETA1) * g_eff
exp_inf_new = torch.maximum(BETA2 * exp_inf, torch.abs(g_eff) + EPS)
p_new       = p - clr * exp_avg_new / exp_inf_new

save("adamax_params_in.bin",   p)
save("adamax_grad.bin",        g)
save("adamax_exp_avg_in.bin",  exp_avg)
save("adamax_exp_inf_in.bin",  exp_inf)
save("adamax_params_out.bin",  p_new)
save("adamax_exp_avg_out.bin", exp_avg_new)
save("adamax_exp_inf_out.bin", exp_inf_new)

print("done")
