#!/usr/bin/env python3
"""Generate PyTorch fixtures for SGD optimizer tests."""
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
LR, WD, MOMENTUM, DAMPENING = 0.01, 1e-4, 0.9, 0.0

# ── SGD (no momentum) ─────────────────────────────────────────────────────────
print("sgd_step")
p  = torch.randn(N)
g  = torch.randn(N) * 0.1
p_r = p.clone().requires_grad_(False)
p_r2 = p_r.clone()
p_r2.data.add_(g + WD * p_r2, alpha=-LR)
save("sgd_params_in.bin",  p)
save("sgd_grad.bin",       g)
save("sgd_params_out.bin", p_r2)

# ── SGD with momentum ─────────────────────────────────────────────────────────
print("sgd_momentum_step")
p_m   = torch.randn(N)
g_m   = torch.randn(N) * 0.1
buf_m = torch.randn(N) * 0.05   # pre-existing momentum buffer

g_eff   = g_m + WD * p_m
buf_new = MOMENTUM * buf_m + (1 - DAMPENING) * g_eff
p_new   = p_m - LR * buf_new

save("sgd_mom_params_in.bin",  p_m)
save("sgd_mom_grad.bin",       g_m)
save("sgd_mom_buf_in.bin",     buf_m)
save("sgd_mom_params_out.bin", p_new)
save("sgd_mom_buf_out.bin",    buf_new)

# ── SGD Nesterov ──────────────────────────────────────────────────────────────
print("sgd_nesterov_step")
p_n   = torch.randn(N)
g_n   = torch.randn(N) * 0.1
buf_n = torch.randn(N) * 0.05

g_eff_n  = g_n + WD * p_n
buf_new_n = MOMENTUM * buf_n + (1 - DAMPENING) * g_eff_n
g_nes    = g_eff_n + MOMENTUM * buf_new_n
p_new_n  = p_n - LR * g_nes

save("sgd_nes_params_in.bin",  p_n)
save("sgd_nes_grad.bin",       g_n)
save("sgd_nes_buf_in.bin",     buf_n)
save("sgd_nes_params_out.bin", p_new_n)
save("sgd_nes_buf_out.bin",    buf_new_n)

print("done")
