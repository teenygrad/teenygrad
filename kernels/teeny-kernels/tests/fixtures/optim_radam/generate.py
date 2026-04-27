#!/usr/bin/env python3
"""Generate PyTorch fixtures for RAdam optimizer tests (both phases)."""
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
RHO_INF = 2 / (1 - BETA2) - 1

def radam_scalars(step):
    bc1      = 1 - BETA1 ** step
    bc2      = 1 - BETA2 ** step
    bc2_sqrt = math.sqrt(bc2)
    rho_t    = RHO_INF - 2 * step * (BETA2 ** step) / bc2
    adaptive = rho_t > 5
    if adaptive:
        rect      = math.sqrt((rho_t-4)*(rho_t-2)*RHO_INF / ((RHO_INF-4)*(RHO_INF-2)*rho_t))
        step_size = LR * rect / bc1
    else:
        step_size = LR / bc1
        bc2_sqrt  = 1.0   # unused in SGD phase
    return adaptive, step_size, bc2_sqrt, rho_t

# ── RAdam adaptive phase (use step=100 so rho_t >> 5) ─────────────────────────
STEP_A = 100
print(f"radam_adaptive (step={STEP_A})")
adaptive_a, step_size_a, bc2sqrt_a, rho_a = radam_scalars(STEP_A)
assert adaptive_a, f"Expected adaptive phase at step={STEP_A}, rho_t={rho_a}"

p_a = torch.randn(N)
g_a = torch.randn(N) * 0.1
m_a = torch.randn(N) * 0.01
v_a = torch.abs(torch.randn(N) * 0.001)

g_eff_a = g_a + WD * p_a
m_an    = BETA1 * m_a + (1 - BETA1) * g_eff_a
v_an    = BETA2 * v_a + (1 - BETA2) * g_eff_a ** 2
denom_a = torch.sqrt(v_an) / bc2sqrt_a + EPS
p_an    = p_a - step_size_a * m_an / denom_a

save("radam_adap_params_in.bin",      p_a)
save("radam_adap_grad.bin",           g_a)
save("radam_adap_exp_avg_in.bin",     m_a)
save("radam_adap_exp_avg_sq_in.bin",  v_a)
save("radam_adap_params_out.bin",     p_an)
save("radam_adap_exp_avg_out.bin",    m_an)
save("radam_adap_exp_avg_sq_out.bin", v_an)

# ── RAdam SGD-fallback phase (step=1, rho_t < 5) ──────────────────────────────
STEP_S = 1
print(f"radam_sgd (step={STEP_S})")
adaptive_s, step_size_s, _, rho_s = radam_scalars(STEP_S)
assert not adaptive_s, f"Expected SGD phase at step={STEP_S}, rho_t={rho_s}"

p_s = torch.randn(N)
g_s = torch.randn(N) * 0.1
m_s = torch.zeros(N)   # fresh state
v_s = torch.zeros(N)

g_eff_s = g_s + WD * p_s
m_sn    = BETA1 * m_s + (1 - BETA1) * g_eff_s
v_sn    = BETA2 * v_s + (1 - BETA2) * g_eff_s ** 2
p_sn    = p_s - step_size_s * m_sn

save("radam_sgd_params_in.bin",      p_s)
save("radam_sgd_grad.bin",           g_s)
save("radam_sgd_exp_avg_in.bin",     m_s)
save("radam_sgd_exp_avg_sq_in.bin",  v_s)
save("radam_sgd_params_out.bin",     p_sn)
save("radam_sgd_exp_avg_out.bin",    m_sn)
save("radam_sgd_exp_avg_sq_out.bin", v_sn)

print("done")
