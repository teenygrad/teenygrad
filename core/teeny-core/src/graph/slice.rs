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

use alloc::{vec, vec::Vec};

/// How a `Slice` maps output coordinates onto its input, per dimension.
///
/// Produced by [`slice_plan`], which normalises ONNX `Slice`'s operands once so that
/// shape inference and kernel lowering cannot disagree about them. They are easy to
/// disagree about: starts and ends may be negative, are clamped differently depending
/// on the sign of the step, and the clamped start is what a kernel needs while only
/// the resulting extent is what shape inference needs.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SlicePlan {
    /// First input index read along each dimension, already normalised and clamped.
    pub starts: Vec<i64>,
    /// Input elements to advance along each dimension per output element. May be
    /// negative, which reverses that axis.
    pub steps: Vec<i64>,
    /// The output extent of each dimension.
    pub out_shape: Vec<usize>,
}

/// Normalises ONNX `Slice`'s operands against an input shape.
///
/// `axes` and `steps` are optional, defaulting to the leading axes and to 1 as ONNX
/// specifies. Dimensions no axis names are passed through whole. Returns `None` if the
/// operands disagree in length, name an axis outside the rank, or ask for a zero step,
/// each of which makes the op meaningless rather than empty.
pub fn slice_plan(
    in_shape: &[usize],
    starts: &[i64],
    ends: &[i64],
    axes: Option<&[i64]>,
    steps: Option<&[i64]>,
) -> Option<SlicePlan> {
    let rank = in_shape.len();
    let default_axes: Vec<i64> = (0..starts.len() as i64).collect();
    let axes = axes.unwrap_or(&default_axes);
    let default_steps: Vec<i64> = vec![1; starts.len()];
    let steps = steps.unwrap_or(&default_steps);
    if starts.len() != ends.len() || starts.len() != axes.len() || starts.len() != steps.len() {
        return None;
    }

    // Untouched dimensions are copied whole.
    let mut plan = SlicePlan {
        starts: vec![0; rank],
        steps: vec![1; rank],
        out_shape: in_shape.to_vec(),
    };

    for k in 0..starts.len() {
        let ax = axes[k];
        let ax = if ax < 0 { ax + rank as i64 } else { ax };
        if ax < 0 || ax as usize >= rank {
            return None;
        }
        let ax = ax as usize;
        let dim = in_shape[ax] as i64;
        let step = steps[k];
        if step == 0 {
            return None;
        }
        let mut s = if starts[k] < 0 {
            starts[k] + dim
        } else {
            starts[k]
        };
        let mut e = if ends[k] < 0 { ends[k] + dim } else { ends[k] };
        // A forward slice clamps into `[0, dim]`; a reverse one into `[-1, dim - 1]`,
        // because -1 is where it stops rather than an index it reads.
        let len = if step > 0 {
            s = s.clamp(0, dim);
            e = e.clamp(0, dim);
            (e - s + step - 1).max(0) / step
        } else {
            s = s.clamp(-1, dim - 1);
            e = e.clamp(-1, dim - 1);
            (s - e + (-step) - 1).max(0) / (-step)
        };
        plan.starts[ax] = s;
        plan.steps[ax] = step;
        plan.out_shape[ax] = len as usize;
    }

    Some(plan)
}
