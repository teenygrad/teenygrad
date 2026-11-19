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

/// How many input elements one output element of a resize reads, per axis.
///
/// Nearest takes one, linear two and cubic four, and a resize over several axes takes
/// the product of them: a 3-D linear resize reads eight.
pub const fn resize_taps(mode: ResizeMode) -> usize {
    match mode {
        ResizeMode::Nearest(_) => 1,
        ResizeMode::Linear => 2,
        ResizeMode::Cubic => 4,
    }
}

/// How a nearest-mode resize rounds a coordinate between two elements.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResizeNearest {
    /// Towards zero, which is what a plain upsample asks for.
    Floor,
    /// Away from zero.
    Ceil,
    /// To the nearer element, a half going down. ONNX's default.
    RoundPreferFloor,
    /// To the nearer element, a half going up.
    RoundPreferCeil,
}

/// Which interpolation a resize uses.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResizeMode {
    /// One element per axis, chosen by a [`ResizeNearest`] rounding.
    Nearest(ResizeNearest),
    /// Two elements per axis, weighted by distance.
    Linear,
    /// Four elements per axis, by ONNX's cubic convolution with `a = -0.75`.
    Cubic,
}

/// How an output coordinate maps back to the input's.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResizeCoords {
    /// `x = (o + 0.5) / scale - 0.5`, ONNX's default.
    HalfPixel,
    /// `x = o * (in - 1) / (out - 1)`: the first and last elements line up exactly.
    AlignCorners,
    /// `x = o / scale`, which lines up the first element and lets the last drift.
    Asymmetric,
}

/// Why a resize cannot be planned.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ResizeError {
    /// An axis with no elements on one side or the other.
    EmptyAxis {
        /// The input's extent.
        input: usize,
        /// The output's extent.
        output: usize,
    },
}

impl core::fmt::Display for ResizeError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::EmptyAxis { input, output } => write!(
                f,
                "an axis of {input} cannot be resized to {output}: both need at least \
                 one element"
            ),
        }
    }
}

/// Which input elements each output element of one resized axis reads, and how much of
/// each.
///
/// One entry per `(output coordinate, tap)`, in that order, so entry `o * taps + t` is
/// tap `t` of output coordinate `o`. Indices are already clamped into the axis, which
/// is the edge behaviour ONNX specifies when a tap falls outside, and the weights are
/// left unnormalised, which is what `exclude_outside = 0` means.
///
/// Computing this on the host is what keeps the interpolation mode out of the kernel:
/// the kernel multiplies and adds what it is given and knows nothing about half-pixel
/// offsets or cubic coefficients.
#[derive(Debug, Clone, PartialEq)]
pub struct ResizeAxisPlan {
    /// Clamped input index per `(output coordinate, tap)`.
    pub indices: Vec<i32>,
    /// Weight per `(output coordinate, tap)`, in the same order.
    pub weights: Vec<f64>,
    /// Taps per output coordinate.
    pub taps: usize,
}

/// Plans one resized axis.
///
/// `scale` is the output extent over the input's, which ONNX uses directly when the
/// resize was given scales rather than sizes; passing `None` derives it from the
/// extents, which is what a resize given explicit sizes does.
pub fn resize_axis_plan(
    mode: ResizeMode,
    coords: ResizeCoords,
    in_ext: usize,
    out_ext: usize,
    scale: Option<f64>,
) -> Result<ResizeAxisPlan, ResizeError> {
    if in_ext == 0 || out_ext == 0 {
        return Err(ResizeError::EmptyAxis {
            input: in_ext,
            output: out_ext,
        });
    }
    let taps = resize_taps(mode);
    let scale = scale.unwrap_or(out_ext as f64 / in_ext as f64);
    let last = in_ext as i64 - 1;
    let clamp = |i: i64| -> i32 { i.clamp(0, last) as i32 };

    let mut indices = Vec::with_capacity(out_ext * taps);
    let mut weights = Vec::with_capacity(out_ext * taps);

    for o in 0..out_ext {
        // Where this output coordinate sits in the input, as a real number.
        let x = match coords {
            ResizeCoords::HalfPixel => (o as f64 + 0.5) / scale - 0.5,
            ResizeCoords::AlignCorners => {
                if out_ext == 1 {
                    0.0
                } else {
                    o as f64 * (in_ext as f64 - 1.0) / (out_ext as f64 - 1.0)
                }
            }
            ResizeCoords::Asymmetric => o as f64 / scale,
        };

        match mode {
            ResizeMode::Nearest(rounding) => {
                let i = match rounding {
                    ResizeNearest::Floor => floor_i64(x),
                    ResizeNearest::Ceil => libm_ceil(x) as i64,
                    // A half goes down, so it is `ceil(x - 0.5)` rather than
                    // `floor(x + 0.5)`, which would send it up.
                    ResizeNearest::RoundPreferFloor => libm_ceil(x - 0.5) as i64,
                    ResizeNearest::RoundPreferCeil => floor_i64(x + 0.5),
                };
                indices.push(clamp(i));
                weights.push(1.0);
            }
            ResizeMode::Linear => {
                let i0 = floor_i64(x);
                let frac = x - i0 as f64;
                indices.push(clamp(i0));
                weights.push(1.0 - frac);
                indices.push(clamp(i0 + 1));
                weights.push(frac);
            }
            ResizeMode::Cubic => {
                // ONNX's cubic convolution, with the default coefficient.
                const A: f64 = -0.75;
                let i1 = floor_i64(x);
                let s = x - i1 as f64;
                let w = [
                    ((A * (s + 1.0) - 5.0 * A) * (s + 1.0) + 8.0 * A) * (s + 1.0) - 4.0 * A,
                    ((A + 2.0) * s - (A + 3.0)) * s * s + 1.0,
                    ((A + 2.0) * (1.0 - s) - (A + 3.0)) * (1.0 - s) * (1.0 - s) + 1.0,
                    ((A * (2.0 - s) - 5.0 * A) * (2.0 - s) + 8.0 * A) * (2.0 - s) - 4.0 * A,
                ];
                for (t, weight) in w.iter().enumerate() {
                    indices.push(clamp(i1 - 1 + t as i64));
                    weights.push(*weight);
                }
            }
        }
    }

    Ok(ResizeAxisPlan {
        indices,
        weights,
        taps,
    })
}

/// `floor` for a value that is about to become an index, without `std`.
fn floor_i64(x: f64) -> i64 {
    let truncated = x as i64;
    if x < truncated as f64 {
        truncated - 1
    } else {
        truncated
    }
}

/// `ceil`, without `std`.
fn libm_ceil(x: f64) -> f64 {
    let truncated = x as i64;
    if x > truncated as f64 {
        (truncated + 1) as f64
    } else {
        truncated as f64
    }
}

#[cfg(test)]
mod tests {
    use super::{ResizeCoords, ResizeError, ResizeMode, ResizeNearest, resize_axis_plan};
    use alloc::vec;
    use alloc::vec::Vec;

    fn plan(
        mode: ResizeMode,
        coords: ResizeCoords,
        in_ext: usize,
        out_ext: usize,
    ) -> (Vec<i32>, Vec<f64>) {
        let p = resize_axis_plan(mode, coords, in_ext, out_ext, None).expect("plannable");
        (p.indices, p.weights)
    }

    /// Doubling with aligned corners: the ends land exactly on the ends, and every
    /// other output coordinate falls halfway between two inputs.
    #[test]
    fn linear_with_aligned_corners_pins_the_ends() {
        let (idx, w) = plan(ResizeMode::Linear, ResizeCoords::AlignCorners, 3, 5);
        // x = o * 2 / 4 = 0, 0.5, 1, 1.5, 2
        assert_eq!(idx, vec![0, 1, 0, 1, 1, 2, 1, 2, 2, 2]);
        let want = [1.0, 0.0, 0.5, 0.5, 1.0, 0.0, 0.5, 0.5, 1.0, 0.0];
        for (have, want) in w.iter().zip(want) {
            assert!((have - want).abs() < 1e-12, "{w:?}");
        }
    }

    /// Half-pixel puts the first output coordinate inside the first input element, so
    /// the taps clamp at the edge rather than running off it.
    #[test]
    fn linear_with_half_pixel_clamps_at_the_edges() {
        let (idx, w) = plan(ResizeMode::Linear, ResizeCoords::HalfPixel, 2, 4);
        // x = (o + 0.5) / 2 - 0.5 = -0.25, 0.25, 0.75, 1.25
        assert_eq!(idx, vec![0, 0, 0, 1, 0, 1, 1, 1]);
        // The first output's taps are index 0 twice, and its weights still sum to 1.
        assert!((w[0] + w[1] - 1.0).abs() < 1e-12, "{w:?}");
        assert!((w[2] - 0.75).abs() < 1e-12, "{w:?}");
        assert!((w[3] - 0.25).abs() < 1e-12, "{w:?}");
    }

    /// Nearest rounds a half down, which is ONNX's default and the opposite of what
    /// `floor(x + 0.5)` would do.
    #[test]
    fn nearest_rounds_a_half_down() {
        let (idx, w) = plan(
            ResizeMode::Nearest(ResizeNearest::RoundPreferFloor),
            ResizeCoords::Asymmetric,
            2,
            4,
        );
        // x = o / 2 = 0, 0.5, 1, 1.5 -> 0, 0, 1, 1
        assert_eq!(idx, vec![0, 0, 1, 1]);
        assert_eq!(w, vec![1.0, 1.0, 1.0, 1.0]);
    }

    /// The rounding is not a detail: at a whole scale factor, `floor` repeats each
    /// element and ONNX's default rounding does not, so reading one for the other
    /// shifts the whole axis.
    #[test]
    fn the_nearest_rounding_decides_whether_a_whole_factor_repeats() {
        let (floor, _) = plan(
            ResizeMode::Nearest(ResizeNearest::Floor),
            ResizeCoords::Asymmetric,
            3,
            12,
        );
        assert_eq!(floor, vec![0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2]);

        let (rounded, _) = plan(
            ResizeMode::Nearest(ResizeNearest::RoundPreferFloor),
            ResizeCoords::Asymmetric,
            3,
            12,
        );
        assert_eq!(rounded, vec![0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 2]);

        let (ceil, _) = plan(
            ResizeMode::Nearest(ResizeNearest::Ceil),
            ResizeCoords::Asymmetric,
            3,
            12,
        );
        assert_eq!(ceil, vec![0, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2]);
    }

    /// Cubic takes four taps whose weights sum to one, and at an exact integer
    /// coordinate it is the identity: weight 1 on that element and 0 on its neighbours.
    #[test]
    fn cubic_sums_to_one_and_is_exact_on_an_integer() {
        let p = resize_axis_plan(ResizeMode::Cubic, ResizeCoords::AlignCorners, 5, 9, None)
            .expect("plannable");
        assert_eq!(p.taps, 4);
        for o in 0..9 {
            let w = &p.weights[o * 4..o * 4 + 4];
            let sum: f64 = w.iter().sum();
            assert!(
                (sum - 1.0).abs() < 1e-12,
                "coordinate {o}: {w:?} sums to {sum}"
            );
        }
        // o = 2 maps to x = 1 exactly: the centre tap takes everything.
        let w = &p.weights[8..12];
        assert!((w[1] - 1.0).abs() < 1e-12, "{w:?}");
        assert!(
            w[0].abs() < 1e-12 && w[2].abs() < 1e-12 && w[3].abs() < 1e-12,
            "{w:?}"
        );
    }

    /// One output element from one input element is still a resize, and aligned
    /// corners would otherwise divide by zero.
    #[test]
    fn a_single_output_coordinate_is_the_first_input() {
        let (idx, w) = plan(ResizeMode::Linear, ResizeCoords::AlignCorners, 4, 1);
        assert_eq!(idx, vec![0, 1]);
        assert_eq!(w, vec![1.0, 0.0]);
    }

    #[test]
    fn an_empty_axis_is_refused() {
        assert_eq!(
            resize_axis_plan(ResizeMode::Linear, ResizeCoords::HalfPixel, 0, 4, None),
            Err(ResizeError::EmptyAxis {
                input: 0,
                output: 4
            })
        );
    }

    /// Downsampling: the taps still come from inside the axis, and the weights still
    /// sum to one, which is what makes an average rather than a dimming.
    #[test]
    fn downsampling_stays_inside_the_axis_and_sums_to_one() {
        let p = resize_axis_plan(ResizeMode::Linear, ResizeCoords::HalfPixel, 8, 3, None)
            .expect("plannable");
        for (i, &idx) in p.indices.iter().enumerate() {
            assert!((0..8).contains(&idx), "tap {i} is {idx}");
        }
        for o in 0..3 {
            let sum: f64 = p.weights[o * 2..o * 2 + 2].iter().sum();
            assert!((sum - 1.0).abs() < 1e-12, "coordinate {o} sums to {sum}");
        }
    }
}
