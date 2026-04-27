#!/usr/bin/env python3
"""Generate PyTorch fixtures for NAdam optimizer tests."""
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
LR, BETA1, BETA2, EPS, WD = 0.002, 0.9, 0.999, 1e-8, 1e-4
MOMENTUM_DECAY = 0.004
STEP = 5

# Compute NAdam schedule coefficients for step STEP
# mu_t  = beta1 * (1 - 0.5 * 0.96^(step   * momentum_decay))
# mu_t1 = beta1 * (1 - 0.5 * 0.96^((step+1) * momentum_decay))
mu_t  = BETA1 * (1 - 0.5 * 0.96 ** (STEP     * MOMENTUM_DECAY))
mu_t1 = BETA1 * (1 - 0.5 * 0.96 ** ((STEP+1) * MOMENTUM_DECAY))

# mu_product = product of mu_1 ... mu_t
mu_product      = math.prod(BETA1 * (1 - 0.5 * 0.96 ** (i * MOMENTUM_DECAY)) for i in range(1, STEP+1))
mu_product_next = mu_product * mu_t1

bias_c2      = 1 - BETA2 ** STEP
bias_c2_sqrt = math.sqrt(bias_c2)

coeff_g = (1 - mu_t)  / (1 - mu_product)
coeff_m = mu_t1       / (1 - mu_product_next)

print("nadam_step")
print(f"  coeff_g={coeff_g:.6f}  coeff_m={coeff_m:.6f}  bias_c2_sqrt={bias_c2_sqrt:.6f}")

p       = torch.randn(N)
g       = torch.randn(N) * 0.1
m       = torch.randn(N) * 0.01
v       = torch.abs(torch.randn(N) * 0.001)

g_eff  = g + WD * p
m_new  = BETA1 * m + (1 - BETA1) * g_eff
v_new  = BETA2 * v + (1 - BETA2) * g_eff * g_eff
denom  = torch.sqrt(v_new) / bias_c2_sqrt + EPS
p_new  = p - LR * (coeff_g * g_eff + coeff_m * m_new) / denom

save("nadam_params_in.bin",      p)
save("nadam_grad.bin",           g)
save("nadam_exp_avg_in.bin",     m)
save("nadam_exp_avg_sq_in.bin",  v)
save("nadam_params_out.bin",     p_new)
save("nadam_exp_avg_out.bin",    m_new)
save("nadam_exp_avg_sq_out.bin", v_new)

print("done")
