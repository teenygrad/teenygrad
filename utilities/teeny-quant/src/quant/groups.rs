/*
 * Copyright (c) 2026 teenygrad (https://teenygrad.org).
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

//! Row-major shape/group-id arithmetic shared by [`crate::quant::affine`] and
//! [`crate::quant::fp8`], both of which need to map each element of an N-d tensor to the
//! [`crate::quant::granularity::Granularity`] group that owns its scale.
//!
//! [`Granularity::PerChannel`] and [`Granularity::Group`] use genuinely different iteration
//! patterns, not just different parameters of the same one: `PerChannel { axis }` *reduces over
//! every other axis* (one group per index along `axis`, shared by all other elements at that
//! index -- `group_id = idx[axis]`). `Group { axis, group_size }` instead keeps every
//! combination of the *other* axes independent, and only subdivides along `axis` (the standard
//! GPTQ/AWQ per-row grouping). Treating `PerChannel` as `Group { group_size: 1 }` would silently
//! produce near-per-element groups instead (every other axis staying independent rather than
//! being reduced over) -- a real bug caught by this module's tests.

use crate::quant::granularity::Granularity;

fn row_major_strides(shape: &[usize]) -> Vec<usize> {
    let mut strides = vec![1usize; shape.len()];
    for i in (0..shape.len().saturating_sub(1)).rev() {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    strides
}

fn unravel(flat: usize, shape: &[usize], strides: &[usize]) -> Vec<usize> {
    shape
        .iter()
        .enumerate()
        .map(|(d, &s)| (flat / strides[d]) % s)
        .collect()
}

/// Maps a multi-index to a group id in `[0, num_groups)` for [`Granularity::Group`]: contiguous
/// `group_size`-runs along `axis`, with every combination of the other axes getting its own
/// independent set of groups (the standard per-row grouping used for `Linear`/`Conv` weight
/// quantization).
fn group_wise_id(idx: &[usize], shape: &[usize], axis: usize, group_size: usize) -> usize {
    let mut outer_id = 0usize;
    let mut multiplier = 1usize;
    for d in (0..shape.len()).rev() {
        if d == axis {
            continue;
        }
        outer_id += idx[d] * multiplier;
        multiplier *= shape[d];
    }
    let groups_along_axis = shape[axis].div_ceil(group_size);
    outer_id * groups_along_axis + idx[axis] / group_size
}

fn group_wise_num_groups(shape: &[usize], axis: usize, group_size: usize) -> usize {
    let outer: usize = shape
        .iter()
        .enumerate()
        .filter(|&(d, _)| d != axis)
        .map(|(_, &s)| s)
        .product();
    outer * shape[axis].div_ceil(group_size)
}

/// Computes the flat-index -> group-id table for a tensor of `shape` under `granularity`, and
/// the resulting group count.
pub(crate) fn assign_groups(shape: &[usize], granularity: Granularity) -> (Vec<u32>, usize) {
    let n: usize = shape.iter().product();
    match granularity {
        Granularity::PerTensor => (vec![0u32; n], 1),
        Granularity::PerChannel { axis } => {
            let strides = row_major_strides(shape);
            let groups = (0..n)
                .map(|flat| unravel(flat, shape, &strides)[axis] as u32)
                .collect();
            (groups, shape[axis])
        }
        Granularity::Group { axis, group_size } => {
            let strides = row_major_strides(shape);
            let groups = (0..n)
                .map(|flat| {
                    let idx = unravel(flat, shape, &strides);
                    group_wise_id(&idx, shape, axis, group_size) as u32
                })
                .collect();
            (groups, group_wise_num_groups(shape, axis, group_size))
        }
    }
}
