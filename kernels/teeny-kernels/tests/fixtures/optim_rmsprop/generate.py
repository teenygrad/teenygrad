#!/usr/bin/env python3
"""Generate PyTorch fixtures for RMSprop optimizer tests."""
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
LR, ALPHA, EPS, WD, MU = 0.01, 0.99, 1e-8, 1e-4, 0.9

# ── RMSprop (no momentum) ─────────────────────────────────────────────────────
print("rmsprop_step")
p  = torch.randn(N)
g  = torch.randn(N) * 0.1
sq = torch.abs(torch.randn(N) * 0.1) + 0.01

g_eff   = g + WD * p
sq_new  = ALPHA * sq + (1 - ALPHA) * g_eff * g_eff
p_new   = p - LR * g_eff / (torch.sqrt(sq_new) + EPS)

save("rms_params_in.bin",  p)
save("rms_grad.bin",       g)
save("rms_sq_avg_in.bin",  sq)
save("rms_params_out.bin", p_new)
save("rms_sq_avg_out.bin", sq_new)

# ── RMSprop with momentum ─────────────────────────────────────────────────────
print("rmsprop_momentum_step")
p_m  = torch.randn(N)
g_m  = torch.randn(N) * 0.1
sq_m = torch.abs(torch.randn(N) * 0.1) + 0.01
buf  = torch.randn(N) * 0.01

g_eff_m  = g_m + WD * p_m
sq_new_m = ALPHA * sq_m + (1 - ALPHA) * g_eff_m * g_eff_m
buf_new  = MU * buf + g_eff_m / torch.sqrt(sq_new_m + EPS)
p_new_m  = p_m - LR * buf_new

save("rmsm_params_in.bin",  p_m)
save("rmsm_grad.bin",       g_m)
save("rmsm_sq_avg_in.bin",  sq_m)
save("rmsm_buf_in.bin",     buf)
save("rmsm_params_out.bin", p_new_m)
save("rmsm_sq_avg_out.bin", sq_new_m)
save("rmsm_buf_out.bin",    buf_new)

print("done")
