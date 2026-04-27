#!/usr/bin/env python3
"""Generate PyTorch fixtures for Adagrad / Adadelta optimizer tests."""
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

# ── Adagrad ───────────────────────────────────────────────────────────────────
print("adagrad_step")
LR, EPS, WD = 0.01, 1e-10, 1e-4
p   = torch.randn(N)
g   = torch.randn(N) * 0.1
s   = torch.abs(torch.randn(N) * 0.1) + 0.01  # sum (must be > 0)

g_eff   = g + WD * p
s_new   = s + g_eff * g_eff
p_new   = p - LR * g_eff / (torch.sqrt(s_new) + EPS)

save("adagrad_params_in.bin",  p)
save("adagrad_grad.bin",       g)
save("adagrad_sum_in.bin",     s)
save("adagrad_params_out.bin", p_new)
save("adagrad_sum_out.bin",    s_new)

# ── Adadelta ──────────────────────────────────────────────────────────────────
print("adadelta_step")
LR_D, RHO, EPS_D, WD_D = 1.0, 0.9, 1e-6, 1e-4
p_d  = torch.randn(N)
g_d  = torch.randn(N) * 0.1
sq_d = torch.abs(torch.randn(N) * 0.01) + 1e-4  # square_avg
ac_d = torch.abs(torch.randn(N) * 0.01) + 1e-4  # acc_delta

g_eff_d     = g_d + WD_D * p_d
sq_new      = RHO * sq_d + (1 - RHO) * g_eff_d * g_eff_d
delta       = torch.sqrt(ac_d + EPS_D) / torch.sqrt(sq_new + EPS_D) * g_eff_d
p_new_d     = p_d - LR_D * delta
ac_new      = RHO * ac_d + (1 - RHO) * delta * delta

save("adadelta_params_in.bin",    p_d)
save("adadelta_grad.bin",         g_d)
save("adadelta_sq_avg_in.bin",    sq_d)
save("adadelta_acc_delta_in.bin", ac_d)
save("adadelta_params_out.bin",   p_new_d)
save("adadelta_sq_avg_out.bin",   sq_new)
save("adadelta_acc_delta_out.bin",ac_new)

print("done")
