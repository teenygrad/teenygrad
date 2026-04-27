#!/usr/bin/env python3
"""Generate PyTorch fixtures for Adam / AdamW optimizer tests."""
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
LR, BETA1, BETA2, EPS, WD = 0.001, 0.9, 0.999, 1e-8, 1e-4
STEP = 5   # non-trivial step count so bias corrections matter

# ── Adam ──────────────────────────────────────────────────────────────────────
print("adam_step")
p       = torch.randn(N)
g       = torch.randn(N) * 0.1
m       = torch.randn(N) * 0.01   # exp_avg
v       = torch.abs(torch.randn(N) * 0.001)  # exp_avg_sq (must be ≥ 0)

bias_c1       = 1 - BETA1 ** STEP
bias_c2       = 1 - BETA2 ** STEP
bias_c2_sqrt  = math.sqrt(bias_c2)
step_size     = LR / bias_c1

g_eff  = g + WD * p
m_new  = BETA1 * m + (1 - BETA1) * g_eff
v_new  = BETA2 * v + (1 - BETA2) * g_eff * g_eff
denom  = torch.sqrt(v_new) / bias_c2_sqrt + EPS
p_new  = p - step_size * m_new / denom

save("adam_params_in.bin",      p)
save("adam_grad.bin",           g)
save("adam_exp_avg_in.bin",     m)
save("adam_exp_avg_sq_in.bin",  v)
save("adam_params_out.bin",     p_new)
save("adam_exp_avg_out.bin",    m_new)
save("adam_exp_avg_sq_out.bin", v_new)

# ── AdamW ─────────────────────────────────────────────────────────────────────
print("adamw_step")
p_w   = torch.randn(N)
g_w   = torch.randn(N) * 0.1
m_w   = torch.randn(N) * 0.01
v_w   = torch.abs(torch.randn(N) * 0.001)

p_dec  = p_w * (1 - LR * WD)   # decoupled weight decay
m_wn   = BETA1 * m_w + (1 - BETA1) * g_w
v_wn   = BETA2 * v_w + (1 - BETA2) * g_w * g_w
denom_w = torch.sqrt(v_wn) / bias_c2_sqrt + EPS
p_wn   = p_dec - step_size * m_wn / denom_w

save("adamw_params_in.bin",      p_w)
save("adamw_grad.bin",           g_w)
save("adamw_exp_avg_in.bin",     m_w)
save("adamw_exp_avg_sq_in.bin",  v_w)
save("adamw_params_out.bin",     p_wn)
save("adamw_exp_avg_out.bin",    m_wn)
save("adamw_exp_avg_sq_out.bin", v_wn)

print("done")
