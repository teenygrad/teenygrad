#!/usr/bin/env python3
"""Generate PyTorch fixtures for BCE/probability loss integration tests."""

import os
import numpy as np
import torch
import torch.nn.functional as F

torch.manual_seed(42)

BASE = os.path.dirname(os.path.abspath(__file__))


def save(path, tensor):
    arr = tensor.detach().cpu().numpy().astype(np.float32)
    if not arr.flags["C_CONTIGUOUS"]:
        arr = np.ascontiguousarray(arr)
    arr.tofile(path)
    print(f"  {os.path.relpath(path, BASE):50s}  {arr.shape}")


N = 1024

# ── BCELoss ────────────────────────────────────────────────────────────────────
print("bce_loss")
# input must be in (0, 1) — use sigmoid of random values
x_bce_raw = torch.empty(N).uniform_(-3, 3)
x_bce = torch.sigmoid(x_bce_raw)  # in (0,1)
t_bce = torch.empty(N).uniform_(0, 1)   # soft targets also in [0,1]
dy_bce = torch.empty(N).uniform_(-2, 2)

x_r = x_bce.clone().requires_grad_(True)
out_bce = F.binary_cross_entropy(x_r, t_bce, reduction='none')
out_bce.backward(dy_bce)
dx_bce = x_r.grad.detach()

save(f"{BASE}/bce_input.bin",             x_bce)
save(f"{BASE}/bce_target.bin",            t_bce)
save(f"{BASE}/bce_dy.bin",               dy_bce)
save(f"{BASE}/bce_expected_forward.bin", out_bce.detach())
save(f"{BASE}/bce_expected_backward.bin", dx_bce)

# ── BCEWithLogitsLoss ──────────────────────────────────────────────────────────
print("bce_with_logits_loss")
x_bwl = torch.empty(N).uniform_(-5, 5)   # raw logits
t_bwl = torch.empty(N).uniform_(0, 1)    # binary targets in [0,1]
dy_bwl = torch.empty(N).uniform_(-2, 2)

x_r = x_bwl.clone().requires_grad_(True)
out_bwl = F.binary_cross_entropy_with_logits(x_r, t_bwl, reduction='none')
out_bwl.backward(dy_bwl)
dx_bwl = x_r.grad.detach()

save(f"{BASE}/bwl_input.bin",             x_bwl)
save(f"{BASE}/bwl_target.bin",            t_bwl)
save(f"{BASE}/bwl_dy.bin",               dy_bwl)
save(f"{BASE}/bwl_expected_forward.bin", out_bwl.detach())
save(f"{BASE}/bwl_expected_backward.bin", dx_bwl)

# ── SoftMarginLoss ─────────────────────────────────────────────────────────────
print("soft_margin_loss")
x_sml = torch.empty(N).uniform_(-5, 5)
# SoftMarginLoss requires targets in {-1, +1}
t_sml = torch.where(torch.randint(0, 2, (N,), dtype=torch.float32) > 0,
                    torch.ones(N), -torch.ones(N))
dy_sml = torch.empty(N).uniform_(-2, 2)

x_r = x_sml.clone().requires_grad_(True)
out_sml = F.soft_margin_loss(x_r, t_sml, reduction='none')
out_sml.backward(dy_sml)
dx_sml = x_r.grad.detach()

save(f"{BASE}/sml_input.bin",             x_sml)
save(f"{BASE}/sml_target.bin",            t_sml)
save(f"{BASE}/sml_dy.bin",               dy_sml)
save(f"{BASE}/sml_expected_forward.bin", out_sml.detach())
save(f"{BASE}/sml_expected_backward.bin", dx_sml)

# ── KLDivLoss ──────────────────────────────────────────────────────────────────
print("kl_div_loss")
# input = log-prob, target = prob (both in a meaningful range)
raw_inp = torch.empty(N).uniform_(-5, 0)  # log-probs are negative
t_kl = torch.abs(torch.empty(N).uniform_(0, 1))  # non-negative probabilities
dy_kl = torch.empty(N).uniform_(-2, 2)

x_r = raw_inp.clone().requires_grad_(True)
out_kl = F.kl_div(x_r, t_kl, reduction='none')
out_kl.backward(dy_kl)
dx_kl = x_r.grad.detach()

save(f"{BASE}/kl_input.bin",             raw_inp)
save(f"{BASE}/kl_target.bin",            t_kl)
save(f"{BASE}/kl_dy.bin",               dy_kl)
save(f"{BASE}/kl_expected_forward.bin", out_kl.detach())
save(f"{BASE}/kl_expected_backward.bin", dx_kl)

# ── PoissonNLLLoss (log_input=True) ───────────────────────────────────────────
print("poisson_nll_loss")
x_pnll = torch.empty(N).uniform_(-2, 2)   # log-rate (can be any real)
t_pnll = torch.empty(N).uniform_(0, 10)   # counts (non-negative)
dy_pnll = torch.empty(N).uniform_(-2, 2)

x_r = x_pnll.clone().requires_grad_(True)
out_pnll = F.poisson_nll_loss(x_r, t_pnll, log_input=True, full=False, reduction='none')
out_pnll.backward(dy_pnll)
dx_pnll = x_r.grad.detach()

save(f"{BASE}/pnll_input.bin",             x_pnll)
save(f"{BASE}/pnll_target.bin",            t_pnll)
save(f"{BASE}/pnll_dy.bin",               dy_pnll)
save(f"{BASE}/pnll_expected_forward.bin", out_pnll.detach())
save(f"{BASE}/pnll_expected_backward.bin", dx_pnll)

# ── GaussianNLLLoss ────────────────────────────────────────────────────────────
print("gaussian_nll_loss")
x_gnll  = torch.empty(N).uniform_(-3, 3)   # predicted mean
t_gnll  = torch.empty(N).uniform_(-3, 3)   # ground truth
v_gnll  = torch.empty(N).uniform_(0.5, 5)  # positive variance
dy_gnll = torch.empty(N).uniform_(-2, 2)

x_r = x_gnll.clone().requires_grad_(True)
v_r = v_gnll.clone().requires_grad_(True)
# PyTorch's GaussianNLLLoss eps=1e-6 by default
out_gnll = F.gaussian_nll_loss(x_r, t_gnll, v_r, eps=1e-6, full=False, reduction='none')
out_gnll.backward(dy_gnll)
dx_gnll = x_r.grad.detach()

save(f"{BASE}/gnll_input.bin",             x_gnll)
save(f"{BASE}/gnll_target.bin",            t_gnll)
save(f"{BASE}/gnll_var.bin",               v_gnll)
save(f"{BASE}/gnll_dy.bin",               dy_gnll)
save(f"{BASE}/gnll_expected_forward.bin", out_gnll.detach())
save(f"{BASE}/gnll_expected_backward.bin", dx_gnll)

print("\nDone — BCE/probability loss fixtures generated.")
