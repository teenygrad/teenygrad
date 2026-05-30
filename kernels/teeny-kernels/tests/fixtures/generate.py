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

# ── padding layers ─────────────────────────────────────────────────────────────
# Shared dims for 1-D pads: B=2, C=4, L=8, pad=(2,3)
B_P1, C_P1, L_P1 = 2, 4, 8
PAD_LEFT_1, PAD_RIGHT_1 = 2, 3
OL_P1 = L_P1 + PAD_LEFT_1 + PAD_RIGHT_1  # 13

# Shared dims for 2-D pads: B=2, C=4, H=6, W=8, pad=(1,2,2,3)
B_P2, C_P2, H_P2, W_P2 = 2, 4, 6, 8
PT_P2, PB_P2, PL_P2, PR_P2 = 1, 2, 2, 3
OH_P2 = H_P2 + PT_P2 + PB_P2  # 9
OW_P2 = W_P2 + PL_P2 + PR_P2  # 13

# Shared dims for 3-D pads: B=2, C=2, D=4, H=4, W=6, pad=(1,1,1,2,2,2)
B_P3, C_P3, DV_P3, H_P3, W_P3 = 2, 2, 4, 4, 6
PD1_P3, PD2_P3 = 1, 1
PH1_P3, PH2_P3 = 1, 2
PW1_P3, PW2_P3 = 2, 2
OD_P3 = DV_P3 + PD1_P3 + PD2_P3  # 6
OH_P3 = H_P3  + PH1_P3 + PH2_P3  # 7
OW_P3 = W_P3  + PW1_P3 + PW2_P3  # 10

VALUE = 1.5

def pad_fixture(name, fwd_fn, bwd_fn, x_shape, out_shape):
    print(name)
    d = os.path.join(BASE, name)
    x = torch.empty(*x_shape).uniform_(-5, 5)
    x_r = x.clone().requires_grad_(True)
    y = fwd_fn(x_r)
    dy = torch.empty(*out_shape).uniform_(-2, 2)
    y.backward(dy)
    dx = x_r.grad.detach()
    save(f"{d}/x.bin",                x)
    save(f"{d}/dy.bin",               dy)
    save(f"{d}/expected_forward.bin",  y.detach())
    save(f"{d}/expected_backward.bin", dx)

# ── constant_pad1d ────────────────────────────────────────────────────────────
pad_fixture("constant_pad1d",
    lambda x: F.pad(x, (PAD_LEFT_1, PAD_RIGHT_1), mode='constant', value=VALUE),
    None,
    (B_P1, C_P1, L_P1), (B_P1, C_P1, OL_P1))

# ── constant_pad2d ────────────────────────────────────────────────────────────
pad_fixture("constant_pad2d",
    lambda x: F.pad(x, (PL_P2, PR_P2, PT_P2, PB_P2), mode='constant', value=VALUE),
    None,
    (B_P2, C_P2, H_P2, W_P2), (B_P2, C_P2, OH_P2, OW_P2))

# ── constant_pad3d ────────────────────────────────────────────────────────────
pad_fixture("constant_pad3d",
    lambda x: F.pad(x, (PW1_P3, PW2_P3, PH1_P3, PH2_P3, PD1_P3, PD2_P3), mode='constant', value=VALUE),
    None,
    (B_P3, C_P3, DV_P3, H_P3, W_P3), (B_P3, C_P3, OD_P3, OH_P3, OW_P3))

# ── reflection_pad1d ──────────────────────────────────────────────────────────
pad_fixture("reflection_pad1d",
    lambda x: F.pad(x, (PAD_LEFT_1, PAD_RIGHT_1), mode='reflect'),
    None,
    (B_P1, C_P1, L_P1), (B_P1, C_P1, OL_P1))

# ── reflection_pad2d ──────────────────────────────────────────────────────────
pad_fixture("reflection_pad2d",
    lambda x: F.pad(x, (PL_P2, PR_P2, PT_P2, PB_P2), mode='reflect'),
    None,
    (B_P2, C_P2, H_P2, W_P2), (B_P2, C_P2, OH_P2, OW_P2))

# ── reflection_pad3d ──────────────────────────────────────────────────────────
pad_fixture("reflection_pad3d",
    lambda x: F.pad(x, (PW1_P3, PW2_P3, PH1_P3, PH2_P3, PD1_P3, PD2_P3), mode='reflect'),
    None,
    (B_P3, C_P3, DV_P3, H_P3, W_P3), (B_P3, C_P3, OD_P3, OH_P3, OW_P3))

# ── replication_pad1d ─────────────────────────────────────────────────────────
pad_fixture("replication_pad1d",
    lambda x: F.pad(x, (PAD_LEFT_1, PAD_RIGHT_1), mode='replicate'),
    None,
    (B_P1, C_P1, L_P1), (B_P1, C_P1, OL_P1))

# ── replication_pad2d ─────────────────────────────────────────────────────────
pad_fixture("replication_pad2d",
    lambda x: F.pad(x, (PL_P2, PR_P2, PT_P2, PB_P2), mode='replicate'),
    None,
    (B_P2, C_P2, H_P2, W_P2), (B_P2, C_P2, OH_P2, OW_P2))

# ── replication_pad3d ─────────────────────────────────────────────────────────
pad_fixture("replication_pad3d",
    lambda x: F.pad(x, (PW1_P3, PW2_P3, PH1_P3, PH2_P3, PD1_P3, PD2_P3), mode='replicate'),
    None,
    (B_P3, C_P3, DV_P3, H_P3, W_P3), (B_P3, C_P3, OD_P3, OH_P3, OW_P3))

# ── circular_pad1d ────────────────────────────────────────────────────────────
pad_fixture("circular_pad1d",
    lambda x: F.pad(x, (PAD_LEFT_1, PAD_RIGHT_1), mode='circular'),
    None,
    (B_P1, C_P1, L_P1), (B_P1, C_P1, OL_P1))

# ── circular_pad2d ────────────────────────────────────────────────────────────
pad_fixture("circular_pad2d",
    lambda x: F.pad(x, (PL_P2, PR_P2, PT_P2, PB_P2), mode='circular'),
    None,
    (B_P2, C_P2, H_P2, W_P2), (B_P2, C_P2, OH_P2, OW_P2))

# ── circular_pad3d ────────────────────────────────────────────────────────────
pad_fixture("circular_pad3d",
    lambda x: F.pad(x, (PW1_P3, PW2_P3, PH1_P3, PH2_P3, PD1_P3, PD2_P3), mode='circular'),
    None,
    (B_P3, C_P3, DV_P3, H_P3, W_P3), (B_P3, C_P3, OD_P3, OH_P3, OW_P3))

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

# ── layernorm ─────────────────────────────────────────────────────────────────
print("layernorm")
d = os.path.join(BASE, "layernorm")
os.makedirs(d, exist_ok=True)
M_LN, N_LN = 16, 128
x_ln = torch.empty(M_LN, N_LN).uniform_(-2, 2)
weight_ln = torch.empty(N_LN).uniform_(0.5, 1.5)
bias_ln = torch.empty(N_LN).uniform_(-0.5, 0.5)
eps_ln = 1e-5
y_ln = F.layer_norm(x_ln, [N_LN], weight_ln, bias_ln, eps=eps_ln)
save(f"{d}/x.bin", x_ln)
save(f"{d}/weight.bin", weight_ln)
save(f"{d}/bias.bin", bias_ln)
save(f"{d}/expected_forward.bin", y_ln)

# ── rmsnorm ───────────────────────────────────────────────────────────────────
print("rmsnorm")
d = os.path.join(BASE, "rmsnorm")
os.makedirs(d, exist_ok=True)
M_RMS, N_RMS = 16, 128
x_rms = torch.empty(M_RMS, N_RMS).uniform_(-2, 2)
weight_rms = torch.empty(N_RMS).uniform_(0.5, 1.5)
eps_rms = 1e-8
# RMSNorm: rrms = 1/sqrt(mean(x^2) + eps), y = x * rrms * weight
rms_sq_mean = x_rms.pow(2).mean(dim=-1, keepdim=True) + eps_rms
rrms_rms = 1.0 / rms_sq_mean.sqrt()  # [M, 1]
y_rms = x_rms * rrms_rms * weight_rms
save(f"{d}/x.bin", x_rms)
save(f"{d}/weight.bin", weight_rms)
save(f"{d}/expected_forward.bin", y_rms)
save(f"{d}/expected_rrms.bin", rrms_rms.squeeze(-1))

# ── groupnorm ─────────────────────────────────────────────────────────────────
print("groupnorm")
d = os.path.join(BASE, "groupnorm")
os.makedirs(d, exist_ok=True)
N_GN, C_GN, L_GN, G_GN = 4, 8, 16, 4
x_gn = torch.empty(N_GN, C_GN, L_GN).uniform_(-2, 2)
weight_gn = torch.empty(C_GN).uniform_(0.5, 1.5)
bias_gn = torch.empty(C_GN).uniform_(-0.5, 0.5)
eps_gn = 1e-5
y_gn = F.group_norm(x_gn, G_GN, weight_gn, bias_gn, eps=eps_gn)
save(f"{d}/x.bin", x_gn)
save(f"{d}/weight.bin", weight_gn)
save(f"{d}/bias.bin", bias_gn)
save(f"{d}/expected_forward.bin", y_gn)

# ── instancenorm ──────────────────────────────────────────────────────────────
print("instancenorm")
d = os.path.join(BASE, "instancenorm")
os.makedirs(d, exist_ok=True)
N_INS, C_INS, L_INS = 4, 8, 16
x_ins = torch.empty(N_INS, C_INS, L_INS).uniform_(-2, 2)
weight_ins = torch.empty(C_INS).uniform_(0.5, 1.5)
bias_ins = torch.empty(C_INS).uniform_(-0.5, 0.5)
eps_ins = 1e-5
# instance_norm with use_input_stats=True (compute per-instance stats)
y_ins = F.instance_norm(x_ins, weight=weight_ins, bias=bias_ins, eps=eps_ins)
save(f"{d}/x.bin", x_ins)
save(f"{d}/weight.bin", weight_ins)
save(f"{d}/bias.bin", bias_ins)
save(f"{d}/expected_forward.bin", y_ins)

# ── batchnorm ─────────────────────────────────────────────────────────────────
print("batchnorm")
d = os.path.join(BASE, "batchnorm")
os.makedirs(d, exist_ok=True)
N_BN, C_BN = 64, 32
EPS_BN = 1e-5
MOMENTUM_BN = 0.1

x_bn = torch.empty(N_BN, C_BN).uniform_(-3, 3)
weight_bn = torch.empty(C_BN).uniform_(0.5, 1.5)
bias_bn = torch.empty(C_BN).uniform_(-0.5, 0.5)
running_mean_bn = torch.zeros(C_BN)
running_var_bn = torch.ones(C_BN)
dy_bn = torch.empty(N_BN, C_BN).uniform_(-2, 2)

save(f"{d}/x.bin", x_bn)
save(f"{d}/weight.bin", weight_bn)
save(f"{d}/bias.bin", bias_bn)
save(f"{d}/running_mean.bin", running_mean_bn)
save(f"{d}/running_var.bin", running_var_bn)
save(f"{d}/dy.bin", dy_bn)

# Inference: uses frozen running stats (training=False)
y_inf_bn = F.batch_norm(x_bn, running_mean_bn.clone(), running_var_bn.clone(),
                        weight=weight_bn, bias=bias_bn, training=False, eps=EPS_BN)
save(f"{d}/expected_forward_inference.bin", y_inf_bn)

# Training forward: biased batch statistics (divide by N, not N-1)
mean_bn = x_bn.mean(dim=0)                           # [C]
var_bn = ((x_bn - mean_bn) ** 2).mean(dim=0)         # biased [C]
rstd_bn = 1.0 / (var_bn + EPS_BN).sqrt()             # [C]
save(f"{d}/expected_mean.bin", mean_bn)
save(f"{d}/expected_rstd.bin", rstd_bn)

# Training forward output via autograd (consistent with biased stats)
x_r_bn = x_bn.clone().requires_grad_(True)
w_r_bn = weight_bn.clone().requires_grad_(True)
b_r_bn = bias_bn.clone().requires_grad_(True)
y_train_bn = F.batch_norm(x_r_bn, running_mean_bn.clone(), running_var_bn.clone(),
                          weight=w_r_bn, bias=b_r_bn,
                          training=True, momentum=MOMENTUM_BN, eps=EPS_BN)
save(f"{d}/expected_forward_training.bin", y_train_bn.detach())

# Training backward
y_train_bn.backward(dy_bn)
save(f"{d}/expected_dx.bin", x_r_bn.grad.detach())
save(f"{d}/expected_dweight.bin", w_r_bn.grad.detach())
save(f"{d}/expected_dbias.bin", b_r_bn.grad.detach())

# ── conv1d_padded (PAD=1) ──────────────────────────────────────────────────────
print("conv1d_padded")
d = os.path.join(BASE, "conv1d_padded")
os.makedirs(d, exist_ok=True)
# OL = (16 + 2*1 - 3) / 1 + 1 = 16
x_1p  = torch.empty(B_1, C_IN_1, L_1).uniform_(-1, 1)
w_1p  = torch.empty(C_OUT_1, C_IN_1, KL_1).uniform_(-0.5, 0.5)
dy_1p = torch.empty(B_1, C_OUT_1, 16).uniform_(-1, 1)

x_r1p = x_1p.clone().requires_grad_(True)
w_r1p = w_1p.clone().requires_grad_(True)
y_1p_out = F.conv1d(x_r1p, w_r1p, stride=1, padding=1)
y_1p_out.backward(dy_1p)

save(f"{d}/x.bin",               x_1p)
save(f"{d}/w.bin",               w_1p)
save(f"{d}/dy.bin",              dy_1p)
save(f"{d}/expected_forward.bin", y_1p_out.detach())
save(f"{d}/expected_dx.bin",     x_r1p.grad)
save(f"{d}/expected_dw.bin",     w_r1p.grad)

# ── conv2d_padded (PAD_H=PAD_W=1) ─────────────────────────────────────────────
print("conv2d_padded")
d = os.path.join(BASE, "conv2d_padded")
os.makedirs(d, exist_ok=True)
# OH = (8 + 2*1 - 3) / 1 + 1 = 8, OW = 8
x_cp  = torch.empty(B_C, C_IN, H_C, W_C).uniform_(-1, 1)
w_cp  = torch.empty(C_OUT, C_IN, KH_C, KW_C).uniform_(-0.5, 0.5)
dy_cp = torch.empty(B_C, C_OUT, 8, 8).uniform_(-1, 1)

x_rcp = x_cp.clone().requires_grad_(True)
w_rcp = w_cp.clone().requires_grad_(True)
y_cp_out = F.conv2d(x_rcp, w_rcp, stride=1, padding=1)
y_cp_out.backward(dy_cp)

save(f"{d}/x.bin",               x_cp)
save(f"{d}/w.bin",               w_cp)
save(f"{d}/dy.bin",              dy_cp)
save(f"{d}/expected_forward.bin", y_cp_out.detach())
save(f"{d}/expected_dx.bin",     x_rcp.grad)
save(f"{d}/expected_dw.bin",     w_rcp.grad)

# ── conv3d_padded (PAD_D=PAD_H=PAD_W=1) ───────────────────────────────────────
print("conv3d_padded")
d = os.path.join(BASE, "conv3d_padded")
os.makedirs(d, exist_ok=True)
# OD = (4+2-2)/1+1=5, OH = (4+2-2)/1+1=5, OW = (8+2-3)/1+1=8
x_3p  = torch.empty(B_3, C_IN_3, D_3, H_3, W_3).uniform_(-1, 1)
w_3p  = torch.empty(C_OUT_3, C_IN_3, KD_3, KH_3, KW_3).uniform_(-0.5, 0.5)
dy_3p = torch.empty(B_3, C_OUT_3, 5, 5, 8).uniform_(-1, 1)

x_r3p = x_3p.clone().requires_grad_(True)
w_r3p = w_3p.clone().requires_grad_(True)
y_3p_out = F.conv3d(x_r3p, w_r3p, stride=1, padding=1)
y_3p_out.backward(dy_3p)

save(f"{d}/x.bin",               x_3p)
save(f"{d}/w.bin",               w_3p)
save(f"{d}/dy.bin",              dy_3p)
save(f"{d}/expected_forward.bin", y_3p_out.detach())
save(f"{d}/expected_dx.bin",     x_r3p.grad)
save(f"{d}/expected_dw.bin",     w_r3p.grad)

# ── flash_attn2 ────────────────────────────────────────────────────────────────
print("flash_attn2")
import math

FA_BATCH    = 2
FA_N_HEADS  = 2
FA_BH       = FA_BATCH * FA_N_HEADS  # 4
FA_N_CTX    = 8
FA_HEAD_DIM = 64
FA_SCALE    = 1.0 / math.sqrt(FA_HEAD_DIM)

d = os.path.join(BASE, "flash_attn2")
os.makedirs(d, exist_ok=True)

q_fa = torch.empty(FA_BH, FA_N_CTX, FA_HEAD_DIM).uniform_(-1, 1)
k_fa = torch.empty(FA_BH, FA_N_CTX, FA_HEAD_DIM).uniform_(-1, 1)
v_fa = torch.empty(FA_BH, FA_N_CTX, FA_HEAD_DIM).uniform_(-1, 1)

scores_fa = q_fa @ k_fa.transpose(-1, -2) * FA_SCALE  # [BH, N_CTX, N_CTX]
o_fa      = F.softmax(scores_fa, dim=-1) @ v_fa        # [BH, N_CTX, HEAD_DIM]
l_fa      = torch.logsumexp(scores_fa, dim=-1)         # [BH, N_CTX]

save(f"{d}/q.bin",                q_fa)
save(f"{d}/k.bin",                k_fa)
save(f"{d}/v.bin",                v_fa)
save(f"{d}/expected_forward_o.bin", o_fa)
save(f"{d}/expected_forward_l.bin", l_fa)

# Backward fixtures — need p (attention weights) and dO
scores_fa = q_fa @ k_fa.transpose(-1, -2) * FA_SCALE  # [BH, N_CTX, N_CTX]
p_fa      = F.softmax(scores_fa, dim=-1)               # [BH, N_CTX_Q, N_CTX_K]
do_fa     = torch.empty(FA_BH, FA_N_CTX, FA_HEAD_DIM).uniform_(-1, 1)

# D_q = rowsum(O * dO)  [BH, N_CTX_Q]
D_fa = (o_fa * do_fa).sum(dim=-1)

# dS = p * (dO @ V^T - D)  [BH, N_CTX_Q, N_CTX_K]
dS_fa = p_fa * (do_fa @ v_fa.transpose(-1, -2) - D_fa.unsqueeze(-1))

dq_fa = dS_fa @ k_fa * FA_SCALE          # [BH, N_CTX_Q, HEAD_DIM]
dk_fa = dS_fa.transpose(-1, -2) @ q_fa * FA_SCALE  # [BH, N_CTX_K, HEAD_DIM]
dv_fa = p_fa.transpose(-1, -2) @ do_fa   # [BH, N_CTX_K, HEAD_DIM]

save(f"{d}/do.bin",                    do_fa)
save(f"{d}/expected_backward_dq.bin",  dq_fa)
save(f"{d}/expected_backward_dk.bin",  dk_fa)
save(f"{d}/expected_backward_dv.bin",  dv_fa)

# ── elemwise_unary ────────────────────────────────────────────────────────────
print("elemwise_unary")
d = os.path.join(BASE, "elemwise_unary")
os.makedirs(d, exist_ok=True)
N_EU = 1024
torch.manual_seed(42)
# x: positive values in [0.1, 2.0] so log/sqrt/acos/acosh stay valid
x_eu = torch.empty(N_EU).uniform_(0.1, 0.9)  # in (0,1) for acos, asin, atanh
dy_eu = torch.ones(N_EU)  # upstream gradient = 1 for simplicity
save(f"{d}/x.bin", x_eu)
save(f"{d}/dy.bin", dy_eu)

# Forward fixtures
save(f"{d}/expected_abs.bin",        x_eu.abs())
save(f"{d}/expected_neg.bin",        -x_eu)
save(f"{d}/expected_sign.bin",       torch.sign(x_eu))
save(f"{d}/expected_ceil.bin",       x_eu.ceil())
save(f"{d}/expected_floor.bin",      x_eu.floor())
save(f"{d}/expected_sqrt.bin",       x_eu.sqrt())
save(f"{d}/expected_reciprocal.bin", 1.0 / x_eu)
save(f"{d}/expected_exp.bin",        x_eu.exp())
save(f"{d}/expected_log.bin",        x_eu.log())
save(f"{d}/expected_erf.bin",        torch.erf(x_eu))
save(f"{d}/expected_sin.bin",        torch.sin(x_eu))
save(f"{d}/expected_cos.bin",        torch.cos(x_eu))
save(f"{d}/expected_tan.bin",        torch.tan(x_eu))
save(f"{d}/expected_asin.bin",       torch.asin(x_eu))
save(f"{d}/expected_acos.bin",       torch.acos(x_eu))
save(f"{d}/expected_atan.bin",       torch.atan(x_eu))
save(f"{d}/expected_sinh.bin",       torch.sinh(x_eu))
save(f"{d}/expected_cosh.bin",       torch.cosh(x_eu))
save(f"{d}/expected_asinh.bin",      torch.asinh(x_eu))
# acosh requires x >= 1; use a separate tensor in [1.1, 3.0]
x_acosh = torch.empty(N_EU).uniform_(1.1, 3.0)
save(f"{d}/expected_acosh.bin",      torch.acosh(x_acosh))
# atanh requires |x| < 1 — x_eu is already in (0, 0.9)
save(f"{d}/expected_atanh.bin",      torch.atanh(x_eu))

# Backward fixtures (dy = 1 so dx = d(op)/dx)
x_r = x_eu.clone().requires_grad_(True)
x_r.abs().sum().backward()
save(f"{d}/expected_abs_backward.bin", x_r.grad.detach())

x_r = x_eu.clone().requires_grad_(True)
(-x_r).sum().backward()
save(f"{d}/expected_neg_backward.bin", x_r.grad.detach())

x_r = x_eu.clone().requires_grad_(True)
x_r.sqrt().sum().backward()
save(f"{d}/expected_sqrt_backward.bin", x_r.grad.detach())

x_r = x_eu.clone().requires_grad_(True)
(1.0 / x_r).sum().backward()
save(f"{d}/expected_reciprocal_backward.bin", x_r.grad.detach())

x_r = x_eu.clone().requires_grad_(True)
x_r.exp().sum().backward()
save(f"{d}/expected_exp_backward.bin", x_r.grad.detach())

x_r = x_eu.clone().requires_grad_(True)
x_r.log().sum().backward()
save(f"{d}/expected_log_backward.bin", x_r.grad.detach())

x_r = x_eu.clone().requires_grad_(True)
torch.erf(x_r).sum().backward()
save(f"{d}/expected_erf_backward.bin", x_r.grad.detach())

x_r = x_eu.clone().requires_grad_(True)
torch.sin(x_r).sum().backward()
save(f"{d}/expected_sin_backward.bin", x_r.grad.detach())

x_r = x_eu.clone().requires_grad_(True)
torch.cos(x_r).sum().backward()
save(f"{d}/expected_cos_backward.bin", x_r.grad.detach())

x_r = x_eu.clone().requires_grad_(True)
torch.tan(x_r).sum().backward()
save(f"{d}/expected_tan_backward.bin", x_r.grad.detach())

x_r = x_eu.clone().requires_grad_(True)
torch.asin(x_r).sum().backward()
save(f"{d}/expected_asin_backward.bin", x_r.grad.detach())

x_r = x_eu.clone().requires_grad_(True)
torch.acos(x_r).sum().backward()
save(f"{d}/expected_acos_backward.bin", x_r.grad.detach())

x_r = x_eu.clone().requires_grad_(True)
torch.atan(x_r).sum().backward()
save(f"{d}/expected_atan_backward.bin", x_r.grad.detach())

x_r = x_eu.clone().requires_grad_(True)
torch.sinh(x_r).sum().backward()
save(f"{d}/expected_sinh_backward.bin", x_r.grad.detach())

x_r = x_eu.clone().requires_grad_(True)
torch.cosh(x_r).sum().backward()
save(f"{d}/expected_cosh_backward.bin", x_r.grad.detach())

x_r = x_eu.clone().requires_grad_(True)
torch.asinh(x_r).sum().backward()
save(f"{d}/expected_asinh_backward.bin", x_r.grad.detach())

x_r = x_acosh.clone().requires_grad_(True)
torch.acosh(x_r).sum().backward()
save(f"{d}/expected_acosh_backward.bin", x_r.grad.detach())

x_r = x_eu.clone().requires_grad_(True)
torch.atanh(x_r).sum().backward()
save(f"{d}/expected_atanh_backward.bin", x_r.grad.detach())

# IsNaN fixture: mix of NaN and normal values
x_nan = x_eu.clone()
x_nan[::4] = float('nan')  # every 4th element is NaN
save(f"{d}/x_with_nan.bin", x_nan)
expected_isnan = torch.zeros(N_EU)
expected_isnan[::4] = 1.0
save(f"{d}/expected_isnan.bin", expected_isnan)

# ── elemwise_binary ────────────────────────────────────────────────────────────
print("elemwise_binary")
d = os.path.join(BASE, "elemwise_binary")
os.makedirs(d, exist_ok=True)
N_EB = 1024
torch.manual_seed(42)
# Use values in [0.5, 2.0] to keep division, pow, log safe
a_eb = torch.empty(N_EB).uniform_(0.5, 2.0)
b_eb = torch.empty(N_EB).uniform_(0.5, 2.0)
dy_eb = torch.ones(N_EB)
# cond: alternating 0 and 1
cond_eb = torch.zeros(N_EB)
cond_eb[::2] = 1.0
save(f"{d}/a.bin", a_eb)
save(f"{d}/b.bin", b_eb)
save(f"{d}/dy.bin", dy_eb)
save(f"{d}/cond.bin", cond_eb)

# Forward fixtures
save(f"{d}/expected_mul.bin",           a_eb * b_eb)
save(f"{d}/expected_sub.bin",           a_eb - b_eb)
save(f"{d}/expected_div.bin",           a_eb / b_eb)
save(f"{d}/expected_pow.bin",           torch.pow(a_eb, b_eb))
save(f"{d}/expected_fmod.bin",          a_eb - torch.floor(a_eb / b_eb) * b_eb)
save(f"{d}/expected_min.bin",           torch.minimum(a_eb, b_eb))
save(f"{d}/expected_max.bin",           torch.maximum(a_eb, b_eb))
save(f"{d}/expected_mean.bin",          (a_eb + b_eb) / 2.0)
save(f"{d}/expected_sum.bin",           a_eb + b_eb)
save(f"{d}/expected_equal.bin",         (a_eb == b_eb).float())
save(f"{d}/expected_greater.bin",       (a_eb > b_eb).float())
save(f"{d}/expected_greater_equal.bin", (a_eb >= b_eb).float())
save(f"{d}/expected_less.bin",          (a_eb < b_eb).float())
save(f"{d}/expected_less_equal.bin",    (a_eb <= b_eb).float())
save(f"{d}/expected_where.bin",         torch.where(cond_eb.bool(), a_eb, b_eb))
save(f"{d}/expected_clip.bin",          torch.clamp(a_eb, -1.0, 1.0))

# Backward fixtures

# Mul backward: da = dy * b, db = dy * a
a_r = a_eb.clone().requires_grad_(True)
b_r = b_eb.clone().requires_grad_(True)
(a_r * b_r).sum().backward()
save(f"{d}/expected_mul_da.bin", a_r.grad.detach())
save(f"{d}/expected_mul_db.bin", b_r.grad.detach())

# Sub backward: da = dy, db = -dy
save(f"{d}/expected_sub_da.bin", dy_eb)
save(f"{d}/expected_sub_db.bin", -dy_eb)

# Div backward: da = dy/b, db = -a*dy/b^2
save(f"{d}/expected_div_da.bin", dy_eb / b_eb)
save(f"{d}/expected_div_db.bin", -(a_eb * dy_eb / (b_eb * b_eb)))

# Pow backward: da = b*a^(b-1)*dy, db = log(a)*a^b*dy
a_r = a_eb.clone().requires_grad_(True)
b_r = b_eb.clone().requires_grad_(True)
torch.pow(a_r, b_r).sum().backward()
save(f"{d}/expected_pow_da.bin", a_r.grad.detach())
save(f"{d}/expected_pow_db.bin", b_r.grad.detach())

# Min/Max backward
a_r = a_eb.clone().requires_grad_(True)
b_r = b_eb.clone().requires_grad_(True)
torch.minimum(a_r, b_r).sum().backward()
save(f"{d}/expected_min_da.bin", a_r.grad.detach())
save(f"{d}/expected_min_db.bin", b_r.grad.detach())

a_r = a_eb.clone().requires_grad_(True)
b_r = b_eb.clone().requires_grad_(True)
torch.maximum(a_r, b_r).sum().backward()
save(f"{d}/expected_max_da.bin", a_r.grad.detach())
save(f"{d}/expected_max_db.bin", b_r.grad.detach())

# Mean backward: da = db = dy/2
save(f"{d}/expected_mean_da.bin", dy_eb / 2.0)
save(f"{d}/expected_mean_db.bin", dy_eb / 2.0)

# Sum backward: da = db = dy
save(f"{d}/expected_sum_da.bin", dy_eb)
save(f"{d}/expected_sum_db.bin", dy_eb)

# Where backward: dx = where(cond, dy, 0), dy_in = where(cond, 0, dy)
save(f"{d}/expected_where_dx.bin",    torch.where(cond_eb.bool(), dy_eb, torch.zeros_like(dy_eb)))
save(f"{d}/expected_where_dy_in.bin", torch.where(cond_eb.bool(), torch.zeros_like(dy_eb), dy_eb))

# Clip backward: dx = dy where x in [-1, 1], else 0
in_range = (a_eb >= -1.0) & (a_eb <= 1.0)
save(f"{d}/expected_clip_backward.bin", torch.where(in_range, dy_eb, torch.zeros_like(dy_eb)))

# ── reduction ─────────────────────────────────────────────────────────────────
print("reduction")
d = os.path.join(BASE, "reduction")
os.makedirs(d, exist_ok=True)
OUTER_R, INNER_R = 32, 64
torch.manual_seed(42)
# Use positive values for log-based reductions
x_red = torch.empty(OUTER_R, INNER_R).uniform_(0.5, 2.0)
save(f"{d}/x.bin", x_red)

save(f"{d}/expected_reduce_sum.bin",        x_red.sum(dim=-1))
save(f"{d}/expected_reduce_mean.bin",       x_red.mean(dim=-1))
save(f"{d}/expected_reduce_max.bin",        x_red.max(dim=-1).values)
save(f"{d}/expected_reduce_min.bin",        x_red.min(dim=-1).values)
save(f"{d}/expected_reduce_prod.bin",       x_red.prod(dim=-1))
save(f"{d}/expected_reduce_l1.bin",         x_red.abs().sum(dim=-1))
save(f"{d}/expected_reduce_l2.bin",         (x_red ** 2).sum(dim=-1).sqrt())
save(f"{d}/expected_reduce_log_sum.bin",    torch.log(x_red.sum(dim=-1)))
save(f"{d}/expected_reduce_log_sum_exp.bin", torch.logsumexp(x_red, dim=-1))
save(f"{d}/expected_reduce_sum_square.bin", (x_red ** 2).sum(dim=-1))
save(f"{d}/expected_global_avg_pool.bin",   x_red.mean(dim=-1))
save(f"{d}/expected_global_max_pool.bin",   x_red.max(dim=-1).values)

# Cumulative ops: output same shape as input
save(f"{d}/expected_cum_sum.bin",  torch.cumsum(x_red, dim=-1))
save(f"{d}/expected_cum_prod.bin", torch.cumprod(x_red, dim=-1))

# ── extra_activations ─────────────────────────────────────────────────────────
print("extra_activations")
d = os.path.join(BASE, "extra_activations")
os.makedirs(d, exist_ok=True)
N_EA = 1024
LOG_SOFTMAX_ROWS_EA = 32
LOG_SOFTMAX_COLS_EA = 128
THRELU_ALPHA_EA = 1.0
SHRINK_LAMBD_EA = 0.5
SHRINK_BIAS_EA = 0.0
torch.manual_seed(42)

x_ea = torch.empty(N_EA).uniform_(-5, 5)
dy_ea = torch.ones(N_EA)
slope_ea = torch.empty(N_EA).uniform_(0.1, 1.0)
save(f"{d}/x.bin", x_ea)
save(f"{d}/dy.bin", dy_ea)
save(f"{d}/slope.bin", slope_ea)

# Swish forward: x * sigmoid(x)
save(f"{d}/expected_swish.bin", F.silu(x_ea))
# Swish backward
x_r = x_ea.clone().requires_grad_(True)
F.silu(x_r).sum().backward()
save(f"{d}/expected_swish_backward.bin", x_r.grad.detach())

# PRelu forward: max(0,x) + slope * min(0,x)
save(f"{d}/expected_prelu.bin", F.prelu(x_ea, slope_ea))
# PRelu backward
x_r = x_ea.clone().requires_grad_(True)
s_r = slope_ea.clone().requires_grad_(True)
F.prelu(x_r, s_r).sum().backward()
save(f"{d}/expected_prelu_dx.bin", x_r.grad.detach())
save(f"{d}/expected_prelu_dslope.bin", s_r.grad.detach())

# ThresholdedRelu forward: x if x > alpha else 0
save(f"{d}/expected_thresholded_relu.bin",
     torch.where(x_ea > THRELU_ALPHA_EA, x_ea, torch.zeros_like(x_ea)))
# backward: dy if x > alpha else 0
save(f"{d}/expected_thresholded_relu_backward.bin",
     torch.where(x_ea > THRELU_ALPHA_EA, dy_ea, torch.zeros_like(dy_ea)))

# Shrink forward: x - bias if x > lambd, x + bias if x < -lambd, else 0
def shrink_fwd(x, lambd, bias):
    return torch.where(x > lambd, x - bias,
           torch.where(x < -lambd, x + bias, torch.zeros_like(x)))

save(f"{d}/expected_shrink.bin", shrink_fwd(x_ea, SHRINK_LAMBD_EA, SHRINK_BIAS_EA))
# backward: dy if |x| > lambd else 0
save(f"{d}/expected_shrink_backward.bin",
     torch.where(x_ea.abs() > SHRINK_LAMBD_EA, dy_ea, torch.zeros_like(dy_ea)))

# LogSoftmax: 2D input [n_rows, n_cols]
x_2d = torch.empty(LOG_SOFTMAX_ROWS_EA, LOG_SOFTMAX_COLS_EA).uniform_(-3, 3)
dy_2d = torch.ones(LOG_SOFTMAX_ROWS_EA, LOG_SOFTMAX_COLS_EA)
save(f"{d}/x_2d.bin", x_2d)
save(f"{d}/dy_2d.bin", dy_2d)
y_lsm = F.log_softmax(x_2d, dim=-1)
save(f"{d}/expected_log_softmax.bin", y_lsm)
# backward
x_r = x_2d.clone().requires_grad_(True)
F.log_softmax(x_r, dim=-1).sum().backward()
save(f"{d}/expected_log_softmax_backward.bin", x_r.grad.detach())

print("\nDone — all fixtures generated.")
