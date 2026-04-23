#!/usr/bin/env python3
"""Generate PyTorch fixtures for teeny-kernels integration tests.

Run from any directory:
    python3 tests/fixtures/generate.py

Re-run whenever kernel semantics change to regenerate expected outputs.
All inputs use a fixed seed so fixtures are deterministic.
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


# ── vector_add ─────────────────────────────────────────────────────────────────
print("vector_add")
d = os.path.join(BASE, "vector_add")
N = 1024
x_va = torch.empty(N).uniform_(-10, 10)
y_va = torch.empty(N).uniform_(-10, 10)
save(f"{d}/x.bin", x_va)
save(f"{d}/y.bin", y_va)
save(f"{d}/expected.bin", x_va + y_va)

# ── relu ───────────────────────────────────────────────────────────────────────
print("relu")
d = os.path.join(BASE, "relu")
N = 1024
x_relu = torch.empty(N).uniform_(-5, 5)
save(f"{d}/x.bin", x_relu)
save(f"{d}/expected_forward.bin", torch.relu(x_relu))

# Backward: y is a valid relu output (non-negative, mix of 0s and positives)
y_bwd = torch.empty(N).uniform_(0, 5)
y_bwd = torch.where(y_bwd < 1.0, torch.zeros_like(y_bwd), y_bwd)
dy_bwd = torch.empty(N).uniform_(-5, 5)
save(f"{d}/y_backward.bin", y_bwd)
save(f"{d}/dy_backward.bin", dy_bwd)
save(f"{d}/expected_backward.bin", torch.where(y_bwd > 0, dy_bwd, torch.zeros_like(dy_bwd)))

# ── softmax ────────────────────────────────────────────────────────────────────
print("softmax")
d = os.path.join(BASE, "softmax")
N_ROWS, N_COLS = 64, 128
x_sfx_fwd = torch.empty(N_ROWS, N_COLS).uniform_(-5, 5)
y_sfx_fwd = torch.softmax(x_sfx_fwd, dim=-1)
save(f"{d}/x_forward.bin", x_sfx_fwd)
save(f"{d}/expected_forward.bin", y_sfx_fwd)

# Backward: y must be a valid softmax distribution
x_sfx_bwd = torch.empty(N_ROWS, N_COLS).uniform_(-3, 3)
y_sfx_bwd = torch.softmax(x_sfx_bwd, dim=-1)
dy_sfx = torch.empty(N_ROWS, N_COLS).uniform_(-1, 1)
# dx_i = y_i * (dy_i - sum_j(y_j * dy_j))
dot = (y_sfx_bwd * dy_sfx).sum(dim=-1, keepdim=True)
expected_sfx_bwd = y_sfx_bwd * (dy_sfx - dot)
save(f"{d}/y_backward.bin", y_sfx_bwd)
save(f"{d}/dy_backward.bin", dy_sfx)
save(f"{d}/expected_backward.bin", expected_sfx_bwd)

# ── linear ─────────────────────────────────────────────────────────────────────
print("linear")
d = os.path.join(BASE, "linear")
M, N_LIN, K = 64, 48, 64
inp = torch.empty(M, K).uniform_(-5, 5)
wt  = torch.empty(N_LIN, K).uniform_(-5, 5)
bias = torch.empty(N_LIN).uniform_(-2, 2)
dy_lin = torch.empty(M, N_LIN).uniform_(-1, 1)

save(f"{d}/input.bin",  inp)
save(f"{d}/weight.bin", wt)
save(f"{d}/bias.bin",   bias)
save(f"{d}/dy.bin",     dy_lin)

# Forward
save(f"{d}/expected_no_bias.bin",   F.linear(inp, wt))
save(f"{d}/expected_with_bias.bin", F.linear(inp, wt, bias))

# Backward (dx/dw are the same regardless of bias; db = sum over batch)
inp_r = inp.clone().requires_grad_(True)
wt_r  = wt.clone().requires_grad_(True)
b_r   = bias.clone().requires_grad_(True)
F.linear(inp_r, wt_r, b_r).backward(dy_lin)
save(f"{d}/expected_dx.bin", inp_r.grad)
save(f"{d}/expected_dw.bin", wt_r.grad)
save(f"{d}/expected_db.bin", b_r.grad)

# ── conv2d ─────────────────────────────────────────────────────────────────────
print("conv2d")
d = os.path.join(BASE, "conv2d")
B_C, C_IN, C_OUT, H_C, W_C, KH_C, KW_C = 1, 2, 4, 8, 8, 3, 3
# OH = (8 - 3) / 1 + 1 = 6, OW = 6
x_c  = torch.empty(B_C, C_IN, H_C, W_C).uniform_(-1, 1)
w_c  = torch.empty(C_OUT, C_IN, KH_C, KW_C).uniform_(-0.5, 0.5)
dy_c = torch.empty(B_C, C_OUT, 6, 6).uniform_(-1, 1)

x_r = x_c.clone().requires_grad_(True)
w_r = w_c.clone().requires_grad_(True)
y_c_out = F.conv2d(x_r, w_r, stride=1)
y_c_out.backward(dy_c)

save(f"{d}/x.bin",               x_c)
save(f"{d}/w.bin",               w_c)
save(f"{d}/dy.bin",              dy_c)
save(f"{d}/expected_forward.bin", y_c_out.detach())
save(f"{d}/expected_dx.bin",     x_r.grad)
save(f"{d}/expected_dw.bin",     w_r.grad)

# ── avgpool2d ──────────────────────────────────────────────────────────────────
print("avgpool2d")
d = os.path.join(BASE, "avgpool2d")
B_AP, C_AP, H_AP, W_AP = 2, 4, 8, 8
# OH = OW = (8 - 2) / 2 + 1 = 4
x_ap  = torch.empty(B_AP, C_AP, H_AP, W_AP).uniform_(-5, 5)
dy_ap = torch.empty(B_AP, C_AP, 4, 4).uniform_(-2, 2)

x_r_ap = x_ap.clone().requires_grad_(True)
y_ap_out = F.avg_pool2d(x_r_ap, kernel_size=(2, 2), stride=(2, 2))
y_ap_out.backward(dy_ap)

save(f"{d}/x.bin",                x_ap)
save(f"{d}/dy.bin",               dy_ap)
save(f"{d}/expected_forward.bin",  y_ap_out.detach())
save(f"{d}/expected_backward.bin", x_r_ap.grad)

# ── flatten ────────────────────────────────────────────────────────────────────
print("flatten")
d = os.path.join(BASE, "flatten")
B_FL, N_FL, PAD_ROWS = 64, 96, 128
padded = torch.empty(PAD_ROWS, N_FL).uniform_(-5, 5)
# Forward: take every other row (indices 0, 2, 4, ..., 126)
expected_fwd_fl = padded[::2].clone()  # [64, 96], C-contiguous

dy_fl = torch.empty(B_FL, N_FL).uniform_(-2, 2)
# Backward: scatter dy into even rows of a (PAD_ROWS, N) zero buffer
expected_bwd_fl = torch.zeros(PAD_ROWS, N_FL)
expected_bwd_fl[::2] = dy_fl

save(f"{d}/padded.bin",           padded)
save(f"{d}/expected_forward.bin", expected_fwd_fl)
save(f"{d}/dy.bin",               dy_fl)
save(f"{d}/expected_backward.bin", expected_bwd_fl)

print("\nDone — all fixtures generated.")
