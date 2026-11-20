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

use alloc::vec::Vec;

/// Which axes a reduction actually reduces, sorted and de-duplicated.
///
/// `axes` counts negative from the end, as ONNX does. An empty list means every axis,
/// unless `noop_with_empty_axes`, where it means none and the op is a copy. Returns
/// `None` when an entry is outside `-rank ..= rank - 1`, since a reduction over an axis
/// the tensor does not have is not a shape that can be guessed at.
///
/// A rank of zero gives an empty list: a scalar has nothing to reduce, and `axes` must
/// then be empty for the same reason.
pub fn reduce_axes(axes: &[i64], rank: usize, noop_with_empty_axes: bool) -> Option<Vec<usize>> {
    if axes.is_empty() {
        return Some(if noop_with_empty_axes {
            Vec::new()
        } else {
            (0..rank).collect()
        });
    }
    let mut out = Vec::with_capacity(axes.len());
    for &axis in axes {
        let resolved = if axis < 0 { axis + rank as i64 } else { axis };
        if resolved < 0 || resolved >= rank as i64 {
            return None;
        }
        out.push(resolved as usize);
    }
    out.sort_unstable();
    out.dedup();
    Some(out)
}

/// A permutation that moves `axes` to the end, keeping every other axis in order.
///
/// `None` when they are already the trailing axes, which is the layout a reduction
/// kernel that walks a flattened `[outer, inner]` view needs. The axes that stay keep
/// their relative order, so the reduced output's elements come out in the same order as
/// the recorded output shape describes, whether or not the reduced axes are kept as 1s.
pub fn reduce_transpose_perm(axes: &[usize], rank: usize) -> Option<Vec<usize>> {
    if axes.is_empty() || axes.len() > rank {
        return None;
    }
    // Already trailing: the axes are exactly the last `axes.len()` of them.
    if axes.iter().copied().eq(rank - axes.len()..rank) {
        return None;
    }
    let mut perm: Vec<usize> = (0..rank).filter(|a| !axes.contains(a)).collect();
    perm.extend_from_slice(axes);
    Some(perm)
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    #[test]
    fn negative_axes_count_from_the_end() {
        assert_eq!(reduce_axes(&[-1], 4, false), Some(vec![3]));
        assert_eq!(reduce_axes(&[-4], 4, false), Some(vec![0]));
        assert_eq!(reduce_axes(&[-2, 1], 4, false), Some(vec![1, 2]));
    }

    #[test]
    fn axes_come_back_sorted_and_deduplicated() {
        assert_eq!(reduce_axes(&[3, 1, 3, -3], 4, false), Some(vec![1, 3]));
    }

    #[test]
    fn an_empty_list_means_every_axis_or_none() {
        assert_eq!(reduce_axes(&[], 3, false), Some(vec![0, 1, 2]));
        assert_eq!(reduce_axes(&[], 3, true), Some(vec![]));
    }

    #[test]
    fn an_axis_outside_the_rank_is_refused() {
        assert_eq!(reduce_axes(&[4], 4, false), None);
        assert_eq!(reduce_axes(&[-5], 4, false), None);
    }

    #[test]
    fn a_scalar_has_nothing_to_reduce() {
        assert_eq!(reduce_axes(&[], 0, false), Some(vec![]));
        assert_eq!(reduce_axes(&[0], 0, false), None);
    }

    #[test]
    fn trailing_axes_need_no_transpose() {
        assert_eq!(reduce_transpose_perm(&[3], 4), None);
        assert_eq!(reduce_transpose_perm(&[2, 3], 4), None);
        assert_eq!(reduce_transpose_perm(&[0, 1, 2, 3], 4), None);
    }

    #[test]
    fn a_middle_axis_moves_to_the_end_and_the_rest_keep_their_order() {
        // FoundationStereo's normalization: [1, 8, 28, 120, 155] reducing axis 2.
        assert_eq!(reduce_transpose_perm(&[2], 5), Some(vec![0, 1, 3, 4, 2]));
        assert_eq!(reduce_transpose_perm(&[0], 3), Some(vec![1, 2, 0]));
        assert_eq!(reduce_transpose_perm(&[1, 2], 4), Some(vec![0, 3, 1, 2]));
    }

    #[test]
    fn nothing_to_move_gives_no_permutation() {
        assert_eq!(reduce_transpose_perm(&[], 4), None);
    }
}
