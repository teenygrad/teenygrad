/*
 * Copyright (c) 2026 Teenygrad.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

//! Fused dist2bbox decode Triton kernel.
//!
//! Converts raw LTRB distance predictions from a detection head into
//! XYWH world-coordinate boxes, fusing the dist2bbox conversion and
//! stride-scaling into a single kernel pass.
//!
//! Layout:
//!   - `boxes`:    `[B, 4, A]` — raw LTRB distances, channels: 0=dx1, 1=dy1, 2=dx2, 3=dy2
//!   - `anchor_x`: `[A]`       — anchor centre x per anchor
//!   - `anchor_y`: `[A]`       — anchor centre y per anchor
//!   - `strides`:  `[A]`       — stride scale per anchor
//!   - `out`:      `[B, 4, A]` — decoded XYWH boxes in world coordinates
//!
//! Parallelism: **one CTA per (batch, BLOCK_A-wide anchor tile)**.
//! Grid: `B * cdiv(A, BLOCK_A)` flat CTAs.
//!
//! Inference only — no backward pass is needed.

#![allow(non_snake_case)]

use teeny_macros::kernel;
use teeny_triton::triton::{
    types::{AddOffsets, Comparison},
    *,
};

/// Fused dist2bbox + stride-scale decode: LTRB distances → XYWH world coords.
///
/// Grid: `B * cdiv(A, BLOCK_A)` — one CTA per (batch element, anchor tile).
#[kernel]
pub fn detect_decode_forward<T: Triton, const BLOCK_A: i32>(
    boxes_ptr: T::Pointer<f32>,
    anchor_x_ptr: T::Pointer<f32>,
    anchor_y_ptr: T::Pointer<f32>,
    strides_ptr: T::Pointer<f32>,
    out_ptr: T::Pointer<f32>,
    _B: i32,
    A: i32,
) where
    T::I32Tensor: types::Tensor<i32, 1>,
    T::I32Tensor: Comparison<i32, BoolTensor = T::BoolTensor>,
    T::Pointer<f32>: AddOffsets<i32, 1, T::I32Tensor, Output = T::Tensor<T::Pointer<f32>>>,
{
    let a_tiles = T::cdiv(A, BLOCK_A);
    let pid_b   = T::program_id(Axis::X) / a_tiles;
    let a_tile  = T::program_id(Axis::X) % a_tiles;
    let a_start = a_tile * BLOCK_A;

    let a_offs = T::arange(0, BLOCK_A) + a_start;
    let mask   = a_offs.lt(A);

    let zeros = T::zeros::<f32>(&[BLOCK_A]);

    // Load per-anchor scalars.
    let anchor_x = T::load(anchor_x_ptr.add_offsets(a_offs), Some(mask), Some(zeros), &[], None, None, None, false);
    let anchor_y = T::load(anchor_y_ptr.add_offsets(a_offs), Some(mask), Some(zeros), &[], None, None, None, false);
    let strides  = T::load(strides_ptr.add_offsets(a_offs),  Some(mask), Some(zeros), &[], None, None, None, false);

    // Load LTRB distances from boxes[pid_b, ch, a_offs].
    // boxes layout: (B, 4, A) → channel ch at flat offset pid_b*4*A + ch*A + a_offs.
    let base = pid_b * 4 * A;
    let dx1 = T::load(boxes_ptr.add_offsets(a_offs + (base + 0 * A)), Some(mask), Some(zeros), &[], None, None, None, false);
    let dy1 = T::load(boxes_ptr.add_offsets(a_offs + (base + 1 * A)), Some(mask), Some(zeros), &[], None, None, None, false);
    let dx2 = T::load(boxes_ptr.add_offsets(a_offs + (base + 2 * A)), Some(mask), Some(zeros), &[], None, None, None, false);
    let dy2 = T::load(boxes_ptr.add_offsets(a_offs + (base + 3 * A)), Some(mask), Some(zeros), &[], None, None, None, false);

    // dist2bbox: x1y1 = anchor - lt, x2y2 = anchor + rb.
    let x1 = anchor_x - dx1;
    let x2 = anchor_x + dx2;
    let y1 = anchor_y - dy1;
    let y2 = anchor_y + dy2;

    // Convert to XYWH and scale by stride.
    let half    = T::full::<f32>(&[BLOCK_A], 0.5f32);
    let cx = (x1 + x2) * half * strides;
    let cy = (y1 + y2) * half * strides;
    let w  = (x2 - x1) * strides;
    let h  = (y2 - y1) * strides;

    // Store decoded boxes to out[pid_b, ch, a_offs].
    T::store(out_ptr.add_offsets(a_offs + (base + 0 * A)), cx, Some(mask), &[], None, None);
    T::store(out_ptr.add_offsets(a_offs + (base + 1 * A)), cy, Some(mask), &[], None, None);
    T::store(out_ptr.add_offsets(a_offs + (base + 2 * A)), w,  Some(mask), &[], None, None);
    T::store(out_ptr.add_offsets(a_offs + (base + 3 * A)), h,  Some(mask), &[], None, None);
}
