#!/usr/bin/env python3
"""Generate PyTorch fixtures for ranking loss integration tests."""

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
    print(f"  {os.path.relpath(path, BASE):55s}  {arr.shape}")


def save_int32(path, tensor):
    arr = tensor.detach().cpu().numpy().astype(np.int32)
    if not arr.flags["C_CONTIGUOUS"]:
        arr = np.ascontiguousarray(arr)
    arr.tofile(path)
    print(f"  {os.path.relpath(path, BASE):55s}  {arr.shape} (int32)")


N = 1024
N_ROWS, N_COLS = 64, 16
MARGIN = 1.0

# ── MarginRankingLoss ─────────────────────────────────────────────────────────
print("margin_ranking_loss")
x1 = torch.empty(N).uniform_(-3, 3)
x2 = torch.empty(N).uniform_(-3, 3)
# y in {-1, 1}
y = (torch.randint(0, 2, (N,)).float() * 2) - 1
dy = torch.empty(N).uniform_(-2, 2)

x1_r = x1.clone().requires_grad_(True)
x2_r = x2.clone().requires_grad_(True)
out_mrl = F.margin_ranking_loss(x1_r, x2_r, y, margin=MARGIN, reduction='none')
out_mrl.backward(dy)
dx1_mrl = x1_r.grad.detach()
dx2_mrl = x2_r.grad.detach()

save(f"{BASE}/mrl_x1.bin",              x1)
save(f"{BASE}/mrl_x2.bin",              x2)
save(f"{BASE}/mrl_y.bin",               y)
save(f"{BASE}/mrl_dy.bin",              dy)
save(f"{BASE}/mrl_expected_forward.bin", out_mrl.detach())
save(f"{BASE}/mrl_expected_dx1.bin",     dx1_mrl)
save(f"{BASE}/mrl_expected_dx2.bin",     dx2_mrl)

# ── HingeEmbeddingLoss ────────────────────────────────────────────────────────
print("hinge_embedding_loss")
inp = torch.empty(N).uniform_(-3, 3)
y_h = (torch.randint(0, 2, (N,)).float() * 2) - 1
dy_h = torch.empty(N).uniform_(-2, 2)

inp_r = inp.clone().requires_grad_(True)
out_hel = F.hinge_embedding_loss(inp_r, y_h, margin=MARGIN, reduction='none')
out_hel.backward(dy_h)
dx_hel = inp_r.grad.detach()

save(f"{BASE}/hel_input.bin",            inp)
save(f"{BASE}/hel_y.bin",               y_h)
save(f"{BASE}/hel_dy.bin",              dy_h)
save(f"{BASE}/hel_expected_forward.bin", out_hel.detach())
save(f"{BASE}/hel_expected_backward.bin", dx_hel)

# ── MultiMarginLoss ───────────────────────────────────────────────────────────
print("multi_margin_loss")
logits_mm = torch.empty(N_ROWS, N_COLS).uniform_(-3, 3)
targets_mm = torch.randint(0, N_COLS, (N_ROWS,))
dy_mm = torch.ones(N_ROWS)

x_r = logits_mm.clone().requires_grad_(True)
out_mm = F.multi_margin_loss(x_r, targets_mm, margin=MARGIN, reduction='none')
out_mm.backward(dy_mm)
dx_mm = x_r.grad.detach()

save(f"{BASE}/mm_input.bin",             logits_mm)
save_int32(f"{BASE}/mm_targets.bin",     targets_mm)
save(f"{BASE}/mm_dy.bin",               dy_mm)
save(f"{BASE}/mm_expected_forward.bin", out_mm.detach())
save(f"{BASE}/mm_expected_backward.bin", dx_mm)

print("\nDone — ranking loss fixtures generated.")
