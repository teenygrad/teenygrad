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

# ── avgpool1d ─────────────────────────────────────────────────────────────────
print("avgpool1d")
d = os.path.join(BASE, "avgpool1d")
os.makedirs(d, exist_ok=True)
B_AP1, C_AP1, L_AP1, KL_AP1, S_AP1 = 2, 4, 16, 4, 4
# OL = (16-4)/4+1 = 4
x_ap1  = torch.empty(B_AP1, C_AP1, L_AP1).uniform_(-5, 5)
dy_ap1 = torch.empty(B_AP1, C_AP1, 4).uniform_(-2, 2)
x_r_ap1 = x_ap1.clone().requires_grad_(True)
y_ap1 = F.avg_pool1d(x_r_ap1, kernel_size=KL_AP1, stride=S_AP1)
y_ap1.backward(dy_ap1)
save(f"{d}/x.bin",                x_ap1)
save(f"{d}/dy.bin",               dy_ap1)
save(f"{d}/expected_forward.bin", y_ap1.detach())
save(f"{d}/expected_backward.bin", x_r_ap1.grad)

# ── avgpool3d ─────────────────────────────────────────────────────────────────
print("avgpool3d")
d = os.path.join(BASE, "avgpool3d")
os.makedirs(d, exist_ok=True)
# OD=OH=OW=4; kernel=2, stride=2
x_ap3  = torch.empty(1, 2, 8, 8, 8).uniform_(-5, 5)
dy_ap3 = torch.empty(1, 2, 4, 4, 4).uniform_(-2, 2)
x_r_ap3 = x_ap3.clone().requires_grad_(True)
y_ap3 = F.avg_pool3d(x_r_ap3, kernel_size=(2, 2, 2), stride=(2, 2, 2))
y_ap3.backward(dy_ap3)
save(f"{d}/x.bin",                x_ap3)
save(f"{d}/dy.bin",               dy_ap3)
save(f"{d}/expected_forward.bin", y_ap3.detach())
save(f"{d}/expected_backward.bin", x_r_ap3.grad)

# ── maxpool1d ─────────────────────────────────────────────────────────────────
print("maxpool1d")
d = os.path.join(BASE, "maxpool1d")
os.makedirs(d, exist_ok=True)
# OL = (16-4)/4+1 = 4 (non-overlapping so backward has unique max)
x_mp1  = torch.empty(2, 4, 16).uniform_(-5, 5)
dy_mp1 = torch.empty(2, 4, 4).uniform_(-2, 2)
x_r_mp1 = x_mp1.clone().requires_grad_(True)
y_mp1, idx_mp1 = F.max_pool1d(x_r_mp1, kernel_size=4, stride=4, return_indices=True)
y_mp1.backward(dy_mp1)
save(f"{d}/x.bin",                x_mp1)
save(f"{d}/dy.bin",               dy_mp1)
save(f"{d}/expected_forward.bin", y_mp1.detach())
save(f"{d}/expected_backward.bin", x_r_mp1.grad)

# ── maxpool2d ─────────────────────────────────────────────────────────────────
print("maxpool2d")
d = os.path.join(BASE, "maxpool2d")
os.makedirs(d, exist_ok=True)
# Non-overlapping: OH=OW=4
x_mp2  = torch.empty(1, 2, 8, 8).uniform_(-5, 5)
dy_mp2 = torch.empty(1, 2, 4, 4).uniform_(-2, 2)
x_r_mp2 = x_mp2.clone().requires_grad_(True)
y_mp2, idx_mp2 = F.max_pool2d(x_r_mp2, kernel_size=(2, 2), stride=(2, 2), return_indices=True)
y_mp2.backward(dy_mp2)
save(f"{d}/x.bin",                x_mp2)
save(f"{d}/dy.bin",               dy_mp2)
save(f"{d}/expected_forward.bin", y_mp2.detach())
save(f"{d}/expected_backward.bin", x_r_mp2.grad)

# ── maxpool3d ─────────────────────────────────────────────────────────────────
print("maxpool3d")
d = os.path.join(BASE, "maxpool3d")
os.makedirs(d, exist_ok=True)
# Non-overlapping: OD=OH=OW=4
x_mp3  = torch.empty(1, 2, 8, 8, 8).uniform_(-5, 5)
dy_mp3 = torch.empty(1, 2, 4, 4, 4).uniform_(-2, 2)
x_r_mp3 = x_mp3.clone().requires_grad_(True)
y_mp3, idx_mp3 = F.max_pool3d(x_r_mp3, kernel_size=(2, 2, 2), stride=(2, 2, 2), return_indices=True)
y_mp3.backward(dy_mp3)
save(f"{d}/x.bin",                x_mp3)
save(f"{d}/dy.bin",               dy_mp3)
save(f"{d}/expected_forward.bin", y_mp3.detach())
save(f"{d}/expected_backward.bin", x_r_mp3.grad)

# ── lppool1d ──────────────────────────────────────────────────────────────────
print("lppool1d")
d = os.path.join(BASE, "lppool1d")
os.makedirs(d, exist_ok=True)
# OL=(16-4)/4+1=4, p=2
x_lp1  = torch.empty(2, 4, 16).uniform_(0.5, 5)   # positive to keep grad stable
dy_lp1 = torch.empty(2, 4, 4).uniform_(-1, 1)
x_r_lp1 = x_lp1.clone().requires_grad_(True)
y_lp1 = F.lp_pool1d(x_r_lp1, norm_type=2, kernel_size=4, stride=4)
y_lp1.backward(dy_lp1)
save(f"{d}/x.bin",                x_lp1)
save(f"{d}/dy.bin",               dy_lp1)
save(f"{d}/expected_forward.bin", y_lp1.detach())
save(f"{d}/expected_backward.bin", x_r_lp1.grad)

# ── lppool2d ──────────────────────────────────────────────────────────────────
print("lppool2d")
d = os.path.join(BASE, "lppool2d")
os.makedirs(d, exist_ok=True)
# OH=OW=4, p=2
x_lp2  = torch.empty(1, 2, 8, 8).uniform_(0.5, 5)
dy_lp2 = torch.empty(1, 2, 4, 4).uniform_(-1, 1)
x_r_lp2 = x_lp2.clone().requires_grad_(True)
y_lp2 = F.lp_pool2d(x_r_lp2, norm_type=2, kernel_size=(2, 2), stride=(2, 2))
y_lp2.backward(dy_lp2)
save(f"{d}/x.bin",                x_lp2)
save(f"{d}/dy.bin",               dy_lp2)
save(f"{d}/expected_forward.bin", y_lp2.detach())
save(f"{d}/expected_backward.bin", x_r_lp2.grad)

# ── lppool3d ──────────────────────────────────────────────────────────────────
print("lppool3d")
d = os.path.join(BASE, "lppool3d")
os.makedirs(d, exist_ok=True)
# OD=OH=OW=4, p=2
x_lp3  = torch.empty(1, 2, 8, 8, 8).uniform_(0.5, 5)
dy_lp3 = torch.empty(1, 2, 4, 4, 4).uniform_(-1, 1)
x_r_lp3 = x_lp3.clone().requires_grad_(True)
y_lp3 = F.lp_pool3d(x_r_lp3, norm_type=2, kernel_size=(2, 2, 2), stride=(2, 2, 2))
y_lp3.backward(dy_lp3)
save(f"{d}/x.bin",                x_lp3)
save(f"{d}/dy.bin",               dy_lp3)
save(f"{d}/expected_forward.bin", y_lp3.detach())
save(f"{d}/expected_backward.bin", x_r_lp3.grad)

# ── conv1d ─────────────────────────────────────────────────────────────────────
print("conv1d")
d = os.path.join(BASE, "conv1d")
os.makedirs(d, exist_ok=True)
B_1, C_IN_1, C_OUT_1, L_1, KL_1 = 1, 2, 4, 16, 3
# OL = (16 - 3) / 1 + 1 = 14
x_1  = torch.empty(B_1, C_IN_1, L_1).uniform_(-1, 1)
w_1  = torch.empty(C_OUT_1, C_IN_1, KL_1).uniform_(-0.5, 0.5)
dy_1 = torch.empty(B_1, C_OUT_1, 14).uniform_(-1, 1)

x_r1 = x_1.clone().requires_grad_(True)
w_r1 = w_1.clone().requires_grad_(True)
y_1_out = F.conv1d(x_r1, w_r1, stride=1)
y_1_out.backward(dy_1)

save(f"{d}/x.bin",               x_1)
save(f"{d}/w.bin",               w_1)
save(f"{d}/dy.bin",              dy_1)
save(f"{d}/expected_forward.bin", y_1_out.detach())
save(f"{d}/expected_dx.bin",     x_r1.grad)
save(f"{d}/expected_dw.bin",     w_r1.grad)

# ── conv3d ─────────────────────────────────────────────────────────────────────
print("conv3d")
d = os.path.join(BASE, "conv3d")
os.makedirs(d, exist_ok=True)
B_3, C_IN_3, C_OUT_3 = 1, 2, 4
D_3, H_3, W_3 = 4, 4, 8
KD_3, KH_3, KW_3 = 2, 2, 3
# OD = (4-2)/1+1=3, OH = (4-2)/1+1=3, OW = (8-3)/1+1=6
x_3  = torch.empty(B_3, C_IN_3, D_3, H_3, W_3).uniform_(-1, 1)
w_3  = torch.empty(C_OUT_3, C_IN_3, KD_3, KH_3, KW_3).uniform_(-0.5, 0.5)
dy_3 = torch.empty(B_3, C_OUT_3, 3, 3, 6).uniform_(-1, 1)

x_r3 = x_3.clone().requires_grad_(True)
w_r3 = w_3.clone().requires_grad_(True)
y_3_out = F.conv3d(x_r3, w_r3, stride=1)
y_3_out.backward(dy_3)

save(f"{d}/x.bin",               x_3)
save(f"{d}/w.bin",               w_3)
save(f"{d}/dy.bin",              dy_3)
save(f"{d}/expected_forward.bin", y_3_out.detach())
save(f"{d}/expected_dx.bin",     x_r3.grad)
save(f"{d}/expected_dw.bin",     w_r3.grad)

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

import math

N_ACT = 1024  # element count for all element-wise activation fixtures

def act_fixture(name, fwd_fn, bwd_fn, x_range=(-5, 5), dy_range=(-2, 2)):
    """Generate forward+backward fixtures for a pointwise activation."""
    print(name)
    d = os.path.join(BASE, name)
    x = torch.empty(N_ACT).uniform_(*x_range)
    dy = torch.empty(N_ACT).uniform_(*dy_range)

    x_r = x.clone().requires_grad_(True)
    y = fwd_fn(x_r)
    y.backward(dy)
    dx = x_r.grad.detach()

    save(f"{d}/x.bin",                x)
    save(f"{d}/dy.bin",               dy)
    save(f"{d}/expected_forward.bin", y.detach())
    save(f"{d}/expected_backward.bin", dx)

# ── Sigmoid ────────────────────────────────────────────────────────────────────
act_fixture("sigmoid", torch.sigmoid, None)

# ── SiLU ──────────────────────────────────────────────────────────────────────
act_fixture("silu", F.silu, None)

# ── LogSigmoid ────────────────────────────────────────────────────────────────
act_fixture("logsigmoid", F.logsigmoid, None)

# ── Tanh ──────────────────────────────────────────────────────────────────────
act_fixture("tanh", torch.tanh, None)

# ── Tanhshrink ────────────────────────────────────────────────────────────────
act_fixture("tanhshrink", F.tanhshrink, None)

# ── GELU (tanh approximation) ─────────────────────────────────────────────────
act_fixture("gelu", lambda x: F.gelu(x, approximate='tanh'), None)

# ── Mish ──────────────────────────────────────────────────────────────────────
act_fixture("mish", F.mish, None)

# ── Hardtanh (min=-1, max=1) ──────────────────────────────────────────────────
act_fixture("hardtanh", lambda x: F.hardtanh(x, min_val=-1.0, max_val=1.0), None)

# ── ReLU6 ─────────────────────────────────────────────────────────────────────
act_fixture("relu6", F.relu6, None)

# ── Hardsigmoid ───────────────────────────────────────────────────────────────
act_fixture("hardsigmoid", F.hardsigmoid, None)

# ── Hardswish ─────────────────────────────────────────────────────────────────
act_fixture("hardswish", F.hardswish, None)

# ── Hardshrink (lambda=0.5) ───────────────────────────────────────────────────
act_fixture("hardshrink", lambda x: F.hardshrink(x, lambd=0.5), None)

# ── ELU (alpha=1.0) ───────────────────────────────────────────────────────────
act_fixture("elu", lambda x: F.elu(x, alpha=1.0), None)

# ── SELU ──────────────────────────────────────────────────────────────────────
act_fixture("selu", F.selu, None)

# ── CELU (alpha=1.0) ──────────────────────────────────────────────────────────
act_fixture("celu", lambda x: F.celu(x, alpha=1.0), None)

# ── LeakyReLU (negative_slope=0.01) ───────────────────────────────────────────
act_fixture("leaky_relu", lambda x: F.leaky_relu(x, negative_slope=0.01), None)

# ── Threshold (threshold=0.5, value=0.0) ──────────────────────────────────────
# Use x.clone().requires_grad_(True) separately since threshold is non-differentiable at boundary
print("threshold")
d_thr = os.path.join(BASE, "threshold")
x_thr = torch.empty(N_ACT).uniform_(-5, 5)
dy_thr = torch.empty(N_ACT).uniform_(-2, 2)
x_thr_r = x_thr.clone().requires_grad_(True)
y_thr = F.threshold(x_thr_r, threshold=0.5, value=0.0)
y_thr.backward(dy_thr)
save(f"{d_thr}/x.bin",                 x_thr)
save(f"{d_thr}/dy.bin",                dy_thr)
save(f"{d_thr}/expected_forward.bin",  y_thr.detach())
save(f"{d_thr}/expected_backward.bin", x_thr_r.grad.detach())

# ── Softsign ──────────────────────────────────────────────────────────────────
act_fixture("softsign", F.softsign, None)

# ── Softshrink (lambda=0.5) ───────────────────────────────────────────────────
act_fixture("softshrink", lambda x: F.softshrink(x, lambd=0.5), None)

# ── Softplus (beta=1, threshold=20) ───────────────────────────────────────────
act_fixture("softplus", lambda x: F.softplus(x, beta=1, threshold=20), None)

print("\nDone — all fixtures generated.")
