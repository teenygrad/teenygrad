#!/usr/bin/env python3
"""Generate fixtures for detect_decode forward test.

Simulates dist2bbox decode for B=2, A=8400 (640px YOLO26 output).

Input:  raw_boxes (B,4,A) — ltrb distances
        anchor_x  (A,)
        anchor_y  (A,)
        strides   (A,)
Output: decoded   (B,4,A) — xywh world coords

Math (per anchor):
    x1 = anchor_x - dx1,  x2 = anchor_x + dx2
    y1 = anchor_y - dy1,  y2 = anchor_y + dy2
    cx = (x1+x2)/2 * stride
    cy = (y1+y2)/2 * stride
    w  = (x2-x1)   * stride
    h  = (y2-y1)   * stride
"""
import os
import numpy as np

np.random.seed(7)
BASE = os.path.dirname(os.path.abspath(__file__))

B = 2
# Use small A for the test (8 anchors — fast but tests the math)
A = 16

raw_boxes = np.random.rand(B, 4, A).astype(np.float32)

# Fake anchor grid: (cx, cy) in range [0.5, 20.5], strides in {8,16,32}
anchor_x = (np.arange(A) % 4 + 0.5).astype(np.float32)
anchor_y = (np.arange(A) // 4 + 0.5).astype(np.float32)
strides  = np.full(A, 8.0, dtype=np.float32)

dx1 = raw_boxes[:, 0, :]  # shape (B, A)
dy1 = raw_boxes[:, 1, :]
dx2 = raw_boxes[:, 2, :]
dy2 = raw_boxes[:, 3, :]

x1 = anchor_x - dx1
x2 = anchor_x + dx2
y1 = anchor_y - dy1
y2 = anchor_y + dy2

cx = (x1 + x2) * 0.5 * strides
cy = (y1 + y2) * 0.5 * strides
w  = (x2 - x1) * strides
h  = (y2 - y1) * strides

decoded = np.stack([cx, cy, w, h], axis=1)  # (B, 4, A)

def save(name, arr):
    path = os.path.join(BASE, name)
    arr.astype(np.float32).flatten(order='C').tofile(path)
    print(f"  {name:30s}  shape={list(arr.shape)}")

save("raw_boxes.bin",   raw_boxes)
save("anchor_x.bin",   anchor_x)
save("anchor_y.bin",   anchor_y)
save("strides.bin",    strides)
save("expected.bin",   decoded)
print("done")
