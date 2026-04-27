#!/usr/bin/env python3
"""Generate PyTorch fixtures for elementwise loss integration tests.

Run from any directory:
    python3 tests/fixtures/loss_elementwise/generate.py
"""

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

# ── L1Loss ─────────────────────────────────────────────────────────────────────
print("l1_loss")
x_l1 = torch.empty(N).uniform_(-5, 5)
y_l1 = torch.empty(N).uniform_(-5, 5)
dy_l1 = torch.empty(N).uniform_(-2, 2)

x_r = x_l1.clone().requires_grad_(True)
out_l1 = F.l1_loss(x_r, y_l1, reduction='none')
out_l1.backward(dy_l1)
dx_l1 = x_r.grad.detach()

save(f"{BASE}/x.bin",                 x_l1)
save(f"{BASE}/y.bin",                 y_l1)
save(f"{BASE}/dy.bin",                dy_l1)
save(f"{BASE}/expected_forward.bin",  out_l1.detach())
save(f"{BASE}/expected_backward.bin", dx_l1)

# ── MSELoss ────────────────────────────────────────────────────────────────────
print("mse_loss")
x_mse = torch.empty(N).uniform_(-5, 5)
y_mse = torch.empty(N).uniform_(-5, 5)
dy_mse = torch.empty(N).uniform_(-2, 2)

x_r = x_mse.clone().requires_grad_(True)
out_mse = F.mse_loss(x_r, y_mse, reduction='none')
out_mse.backward(dy_mse)
dx_mse = x_r.grad.detach()

save(f"{BASE}/mse_x.bin",                 x_mse)
save(f"{BASE}/mse_y.bin",                 y_mse)
save(f"{BASE}/mse_dy.bin",                dy_mse)
save(f"{BASE}/mse_expected_forward.bin",  out_mse.detach())
save(f"{BASE}/mse_expected_backward.bin", dx_mse)

# ── HuberLoss (delta=1.0) ──────────────────────────────────────────────────────
print("huber_loss")
DELTA = 1.0
x_huber = torch.empty(N).uniform_(-5, 5)
y_huber = torch.empty(N).uniform_(-5, 5)
dy_huber = torch.empty(N).uniform_(-2, 2)

x_r = x_huber.clone().requires_grad_(True)
out_huber = F.huber_loss(x_r, y_huber, delta=DELTA, reduction='none')
out_huber.backward(dy_huber)
dx_huber = x_r.grad.detach()

save(f"{BASE}/huber_x.bin",                 x_huber)
save(f"{BASE}/huber_y.bin",                 y_huber)
save(f"{BASE}/huber_dy.bin",                dy_huber)
save(f"{BASE}/huber_expected_forward.bin",  out_huber.detach())
save(f"{BASE}/huber_expected_backward.bin", dx_huber)

# ── SmoothL1Loss (beta=1.0) ────────────────────────────────────────────────────
print("smooth_l1_loss")
BETA = 1.0
x_sl1 = torch.empty(N).uniform_(-5, 5)
y_sl1 = torch.empty(N).uniform_(-5, 5)
dy_sl1 = torch.empty(N).uniform_(-2, 2)

x_r = x_sl1.clone().requires_grad_(True)
out_sl1 = F.smooth_l1_loss(x_r, y_sl1, beta=BETA, reduction='none')
out_sl1.backward(dy_sl1)
dx_sl1 = x_r.grad.detach()

save(f"{BASE}/sl1_x.bin",                 x_sl1)
save(f"{BASE}/sl1_y.bin",                 y_sl1)
save(f"{BASE}/sl1_dy.bin",                dy_sl1)
save(f"{BASE}/sl1_expected_forward.bin",  out_sl1.detach())
save(f"{BASE}/sl1_expected_backward.bin", dx_sl1)

print("\nDone — elementwise loss fixtures generated.")
