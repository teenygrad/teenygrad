#!/usr/bin/env python3
"""Generate PyTorch fixtures for NLL/CrossEntropy loss integration tests."""

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


N_ROWS, N_COLS = 64, 16

# ── NLLLoss ────────────────────────────────────────────────────────────────────
print("nll_loss")
# log_probs: shape [N_ROWS, N_COLS], rows sum to ~1 in prob space
raw_logits = torch.empty(N_ROWS, N_COLS).uniform_(-3, 3)
log_probs = F.log_softmax(raw_logits, dim=-1)
targets = torch.randint(0, N_COLS, (N_ROWS,))
dy = torch.ones(N_ROWS)  # upstream gradient

lp_r = log_probs.clone().requires_grad_(True)
out_nll = F.nll_loss(lp_r, targets, reduction='none')
out_nll.backward(dy)
dx_nll = lp_r.grad.detach()

save(f"{BASE}/nll_log_probs.bin",        log_probs)
save_int32(f"{BASE}/nll_targets.bin",    targets)
save(f"{BASE}/nll_dy.bin",               dy)
save(f"{BASE}/nll_expected_forward.bin", out_nll.detach())
save(f"{BASE}/nll_expected_backward.bin", dx_nll)

# ── CrossEntropyLoss ────────────────────────────────────────────────────────────
print("cross_entropy_loss")
logits_ce = torch.empty(N_ROWS, N_COLS).uniform_(-3, 3)
targets_ce = torch.randint(0, N_COLS, (N_ROWS,))
dy_ce = torch.ones(N_ROWS)

x_r = logits_ce.clone().requires_grad_(True)
out_ce = F.cross_entropy(x_r, targets_ce, reduction='none')
out_ce.backward(dy_ce)
dx_ce = x_r.grad.detach()

save(f"{BASE}/ce_input.bin",             logits_ce)
save_int32(f"{BASE}/ce_targets.bin",     targets_ce)
save(f"{BASE}/ce_dy.bin",               dy_ce)
save(f"{BASE}/ce_expected_forward.bin", out_ce.detach())
save(f"{BASE}/ce_expected_backward.bin", dx_ce)

# ── MultiLabelSoftMarginLoss ───────────────────────────────────────────────────
print("multilabel_soft_margin_loss")
N = 1024
x_mlsm = torch.empty(N).uniform_(-5, 5)
# Binary targets (0 or 1)
t_mlsm = (torch.rand(N) > 0.5).float()
dy_mlsm = torch.empty(N).uniform_(-2, 2)

x_r = x_mlsm.clone().requires_grad_(True)
# MultiLabelSoftMarginLoss with single class = BCEWithLogitsLoss
out_mlsm = F.binary_cross_entropy_with_logits(x_r, t_mlsm, reduction='none')
out_mlsm.backward(dy_mlsm)
dx_mlsm = x_r.grad.detach()

save(f"{BASE}/mlsm_input.bin",             x_mlsm)
save(f"{BASE}/mlsm_target.bin",            t_mlsm)
save(f"{BASE}/mlsm_dy.bin",               dy_mlsm)
save(f"{BASE}/mlsm_expected_forward.bin", out_mlsm.detach())
save(f"{BASE}/mlsm_expected_backward.bin", dx_mlsm)

print("\nDone — NLL/CrossEntropy loss fixtures generated.")
