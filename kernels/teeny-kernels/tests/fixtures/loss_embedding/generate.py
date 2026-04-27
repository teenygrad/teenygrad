#!/usr/bin/env python3
"""Generate PyTorch fixtures for embedding loss integration tests."""

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


N_ROWS, N_DIM = 64, 64
MARGIN = 0.5
EPS = 1e-6

# ── CosineEmbeddingLoss ───────────────────────────────────────────────────────
print("cosine_embedding_loss")
x1 = F.normalize(torch.empty(N_ROWS, N_DIM).uniform_(-1, 1), dim=1)
x2 = F.normalize(torch.empty(N_ROWS, N_DIM).uniform_(-1, 1), dim=1)
# y in {-1, 1}
y = (torch.randint(0, 2, (N_ROWS,)).float() * 2) - 1
dy = torch.ones(N_ROWS)

x1_r = x1.clone().requires_grad_(True)
x2_r = x2.clone().requires_grad_(True)
out_cel = F.cosine_embedding_loss(x1_r, x2_r, y, margin=MARGIN, reduction='none')
out_cel.backward(dy)
dx1_cel = x1_r.grad.detach()
dx2_cel = x2_r.grad.detach()

save(f"{BASE}/cel_x1.bin",               x1)
save(f"{BASE}/cel_x2.bin",               x2)
save(f"{BASE}/cel_y.bin",               y)
save(f"{BASE}/cel_dy.bin",              dy)
save(f"{BASE}/cel_expected_forward.bin", out_cel.detach())
save(f"{BASE}/cel_expected_dx1.bin",     dx1_cel)
save(f"{BASE}/cel_expected_dx2.bin",     dx2_cel)

# ── TripletMarginLoss ─────────────────────────────────────────────────────────
print("triplet_margin_loss")
anchor = F.normalize(torch.empty(N_ROWS, N_DIM).uniform_(-1, 1), dim=1)
positive = F.normalize(torch.empty(N_ROWS, N_DIM).uniform_(-1, 1), dim=1)
negative = F.normalize(torch.empty(N_ROWS, N_DIM).uniform_(-1, 1), dim=1)
dy_t = torch.ones(N_ROWS)

a_r = anchor.clone().requires_grad_(True)
p_r = positive.clone().requires_grad_(True)
n_r = negative.clone().requires_grad_(True)
# p=2 (Euclidean distance), swap=False, eps=1e-6
out_tml = F.triplet_margin_loss(a_r, p_r, n_r, margin=MARGIN, p=2,
                                eps=EPS, swap=False, reduction='none')
out_tml.backward(dy_t)
da_tml = a_r.grad.detach()
dp_tml = p_r.grad.detach()
dn_tml = n_r.grad.detach()

save(f"{BASE}/tml_anchor.bin",           anchor)
save(f"{BASE}/tml_positive.bin",         positive)
save(f"{BASE}/tml_negative.bin",         negative)
save(f"{BASE}/tml_dy.bin",              dy_t)
save(f"{BASE}/tml_expected_forward.bin", out_tml.detach())
save(f"{BASE}/tml_expected_da.bin",      da_tml)
save(f"{BASE}/tml_expected_dp.bin",      dp_tml)
save(f"{BASE}/tml_expected_dn.bin",      dn_tml)

print("\nDone — embedding loss fixtures generated.")
