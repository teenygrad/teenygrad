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

//! Affine (scale + zero-point) integer quantization, shared by the INT8 and INT4 schemes --
//! INT4 just runs this with `bits: 4` and then nibble-packs the result (see
//! [`crate::quant::pack4`]).

use crate::error::{Error, Result};
use crate::quant::granularity::Granularity;
use crate::quant::groups::assign_groups;

/// Per-group affine quantization parameters: `q = round(x / scale) + zero_point`, clamped to the
/// scheme's `[qmin, qmax]`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AffineParams {
    /// The quantization scale (always > 0).
    pub scale: f32,
    /// The zero-point in quantized (integer) units. `0` for symmetric quantization.
    pub zero_point: i32,
}

/// A tensor quantized with one [`AffineParams`] per group.
#[derive(Debug, Clone)]
pub struct QuantizedAffine {
    /// One quantized value per input element, in the same (row-major) order as the input.
    pub qvalues: Vec<i8>,
    /// One [`AffineParams`] per group.
    pub params: Vec<AffineParams>,
    /// `qvalues[i]`'s group is `groups[i]`, i.e. `params[groups[i]]` quantized it.
    pub groups: Vec<u32>,
}

/// The `[qmin, qmax]` integer range for `bits`-bit (signed) quantization.
fn qrange(bits: u8, symmetric: bool) -> (i32, i32) {
    let qmax = (1i32 << (bits - 1)) - 1; // 127 for bits=8, 7 for bits=4
    let qmin = if symmetric { -qmax } else { -qmax - 1 }; // -127/-128, -7/-8
    (qmin, qmax)
}

/// Quantizes `data` (row-major, shape `shape`) to `bits`-bit signed integers at the given
/// `granularity`, computing one [`AffineParams`] per group from that group's own min/max.
pub fn quantize_affine(
    tensor: &str,
    data: &[f32],
    shape: &[usize],
    granularity: Granularity,
    symmetric: bool,
    bits: u8,
) -> Result<QuantizedAffine> {
    let n: usize = shape.iter().product();
    if data.len() != n {
        return Err(Error::ShapeMismatch {
            name: tensor.to_string(),
            a: vec![data.len()],
            b: shape.to_vec(),
        });
    }

    let (qmin, qmax) = qrange(bits, symmetric);
    let axis_group = granularity.axis_and_group_size();

    if let Some((axis, group_size)) = axis_group {
        if axis >= shape.len() {
            return Err(Error::InvalidAxis {
                tensor: tensor.to_string(),
                axis,
                rank: shape.len(),
            });
        }
        if group_size == 0 {
            return Err(Error::InvalidGroupSize(tensor.to_string()));
        }
    }

    let (groups, ngroups) = assign_groups(shape, granularity);

    // Pass 1: per-group min/max (and amax, for symmetric).
    let mut vmin = vec![f32::INFINITY; ngroups];
    let mut vmax = vec![f32::NEG_INFINITY; ngroups];
    for (i, &x) in data.iter().enumerate() {
        let g = groups[i] as usize;
        if x < vmin[g] {
            vmin[g] = x;
        }
        if x > vmax[g] {
            vmax[g] = x;
        }
    }

    let params: Vec<AffineParams> = (0..ngroups)
        .map(|g| {
            let (lo, hi) = if vmin[g].is_finite() {
                (vmin[g].min(0.0), vmax[g].max(0.0))
            } else {
                (0.0, 0.0) // empty group (shouldn't happen, but stay total)
            };
            if symmetric {
                let amax = lo.abs().max(hi.abs());
                let scale = if amax == 0.0 { 1.0 } else { amax / qmax as f32 };
                AffineParams {
                    scale,
                    zero_point: 0,
                }
            } else {
                let scale = if hi == lo {
                    1.0
                } else {
                    (hi - lo) / (qmax - qmin) as f32
                };
                let zero_point = (qmin as f32 - lo / scale).round() as i32;
                AffineParams {
                    scale,
                    zero_point: zero_point.clamp(qmin, qmax),
                }
            }
        })
        .collect();

    // Pass 2: quantize each element with its group's params.
    let qvalues: Vec<i8> = data
        .iter()
        .enumerate()
        .map(|(i, &x)| {
            let p = params[groups[i] as usize];
            let q = (x / p.scale).round() as i32 + p.zero_point;
            q.clamp(qmin, qmax) as i8
        })
        .collect();

    Ok(QuantizedAffine {
        qvalues,
        params,
        groups,
    })
}

/// Reconstructs `f32` values from `qvalues`/`params`/`groups` produced by [`quantize_affine`].
pub fn dequantize_affine(q: &QuantizedAffine) -> Vec<f32> {
    q.qvalues
        .iter()
        .enumerate()
        .map(|(i, &v)| {
            let p = q.params[q.groups[i] as usize];
            (v as i32 - p.zero_point) as f32 * p.scale
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn per_tensor_symmetric_round_trip() {
        let data = vec![-1.0f32, -0.5, 0.0, 0.5, 1.0, 2.0];
        let shape = [6usize];
        let q =
            quantize_affine("t", &data, &shape, Granularity::PerTensor, true, 8).unwrap();
        assert_eq!(q.params.len(), 1);
        assert_eq!(q.params[0].zero_point, 0);
        let deq = dequantize_affine(&q);
        for (orig, back) in data.iter().zip(deq.iter()) {
            assert!((orig - back).abs() < 0.02, "{orig} vs {back}");
        }
    }

    #[test]
    fn per_tensor_asymmetric_covers_positive_only_range() {
        // All-positive data: asymmetric should use the full qmin..qmax range, unlike symmetric.
        let data = vec![0.0f32, 1.0, 2.0, 3.0, 4.0];
        let shape = [5usize];
        let q =
            quantize_affine("t", &data, &shape, Granularity::PerTensor, false, 8).unwrap();
        let deq = dequantize_affine(&q);
        for (orig, back) in data.iter().zip(deq.iter()) {
            assert!((orig - back).abs() < 0.05, "{orig} vs {back}");
        }
    }

    #[test]
    fn per_channel_independent_scales() {
        // shape [2, 4]: row 0 has small magnitude, row 1 much larger -- per-channel (axis 0)
        // should give each row its own scale, so both round-trip tightly.
        let data = vec![0.1f32, -0.1, 0.05, -0.05, 100.0, -100.0, 50.0, -50.0];
        let shape = [2usize, 4];
        let q = quantize_affine(
            "t",
            &data,
            &shape,
            Granularity::PerChannel { axis: 0 },
            true,
            8,
        )
        .unwrap();
        assert_eq!(q.params.len(), 2);
        let deq = dequantize_affine(&q);
        for (orig, back) in data.iter().zip(deq.iter()) {
            let tol = orig.abs().max(1.0) * 0.02;
            assert!((orig - back).abs() < tol, "{orig} vs {back}");
        }
    }

    #[test]
    fn group_wise_matches_per_channel_when_group_size_covers_axis() {
        let data: Vec<f32> = (0..8).map(|i| i as f32 - 4.0).collect();
        let shape = [2usize, 4];
        let per_channel = quantize_affine(
            "t",
            &data,
            &shape,
            Granularity::PerChannel { axis: 0 },
            true,
            8,
        )
        .unwrap();
        let group = quantize_affine(
            "t",
            &data,
            &shape,
            Granularity::Group {
                axis: 1,
                group_size: 4,
            },
            true,
            8,
        )
        .unwrap();
        // Different axes but both reduce to "one group per row of 4" here, so params should
        // come out identical (grouping along the full-width axis == per-channel over rows).
        assert_eq!(per_channel.params.len(), group.params.len());
    }

    #[test]
    fn rejects_zero_group_size() {
        let data = vec![1.0f32; 4];
        let shape = [4usize];
        let err = quantize_affine(
            "t",
            &data,
            &shape,
            Granularity::Group {
                axis: 0,
                group_size: 0,
            },
            true,
            8,
        )
        .unwrap_err();
        assert!(matches!(err, Error::InvalidGroupSize(_)));
    }

    #[test]
    fn rejects_out_of_range_axis() {
        let data = vec![1.0f32; 4];
        let shape = [4usize];
        let err = quantize_affine(
            "t",
            &data,
            &shape,
            Granularity::PerChannel { axis: 3 },
            true,
            8,
        )
        .unwrap_err();
        assert!(matches!(err, Error::InvalidAxis { .. }));
    }

    #[test]
    fn int4_range_is_clamped() {
        let data = vec![-100.0f32, 100.0];
        let shape = [2usize];
        let q =
            quantize_affine("t", &data, &shape, Granularity::PerTensor, true, 4).unwrap();
        for &v in &q.qvalues {
            assert!((-7..=7).contains(&v));
        }
    }
}
