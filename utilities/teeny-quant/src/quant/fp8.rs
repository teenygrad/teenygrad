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

//! `F8_E4M3`/`F8_E5M2` weight quantization.
//!
//! Both are natively supported `safetensors` dtypes (see the pinned `safetensors` 0.7's
//! `Dtype::F8_E4M3`/`Dtype::F8_E5M2`), so unlike INT4 (see [`crate::quant::pack4`]) no packing
//! convention is needed -- this module only has to do the bit-level `f32 <-> f8` conversion and
//! per-group `amax`-based scale computation, since the crate depends on neither the `half` crate
//! (f16/bf16 only, no f8) nor any other f8 implementation.
//!
//! Encoding follows the OCP FP8 spec: `E4M3` is the "FN" variant (no infinities; the single
//! exponent=`1111`/mantissa=`111` bit pattern is reserved for NaN, freeing up the rest of that
//! exponent for finite values up to `448`). `E5M2` is IEEE-754-like (has infinities). Rounding is
//! round-to-nearest-even; out-of-range magnitudes saturate (to `448`/`57344` for `E4M3`/`E5M2`
//! respectively, or to infinity for `E5M2`, which has one). **Subnormal outputs are flushed to
//! zero** rather than rounded into the target format's subnormal range -- an accepted
//! simplification for weight quantization, where values within about one ULP of the smallest
//! normal (`2^-9` for `E4M3`, `2^-16` for `E5M2`) are negligible relative to the scale factor
//! applied before conversion. Decoding handles subnormals and NaN/Inf fully, since it also needs
//! to correctly read back bytes this module didn't itself produce.

use crate::quant::granularity::Granularity;
use crate::quant::groups::assign_groups;

/// Which OCP FP8 encoding to target.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Fp8Variant {
    /// 4 exponent bits, 3 mantissa bits, no infinities (`safetensors` `F8_E4M3`).
    E4M3,
    /// 5 exponent bits, 2 mantissa bits, IEEE-like (`safetensors` `F8_E5M2`).
    E5M2,
}

impl Fp8Variant {
    fn mantissa_bits(self) -> u32 {
        match self {
            Fp8Variant::E4M3 => 3,
            Fp8Variant::E5M2 => 2,
        }
    }

    fn exp_bits(self) -> u32 {
        match self {
            Fp8Variant::E4M3 => 4,
            Fp8Variant::E5M2 => 5,
        }
    }

    fn bias(self) -> i32 {
        match self {
            Fp8Variant::E4M3 => 7,
            Fp8Variant::E5M2 => 15,
        }
    }

    fn has_infinity(self) -> bool {
        matches!(self, Fp8Variant::E5M2)
    }

    /// Largest unbiased exponent used by a *finite* value in this format.
    fn max_finite_unbiased_exp(self) -> i32 {
        match self {
            Fp8Variant::E4M3 => 8, // biased 15, mantissa < 0b111 (448 = 1.75 * 2^8)
            Fp8Variant::E5M2 => 15, // biased 30, mantissa <= 0b11 (57344 = 1.75 * 2^15)
        }
    }

    fn max_finite_value(self) -> f32 {
        match self {
            Fp8Variant::E4M3 => 448.0,
            Fp8Variant::E5M2 => 57344.0,
        }
    }

    fn exp_mask(self) -> u8 {
        ((1u32 << self.exp_bits()) - 1) as u8
    }
}

/// Encodes `x` as an `f8` byte in `variant`. See the module docs for rounding/saturation
/// behavior.
pub fn f32_to_f8(x: f32, variant: Fp8Variant) -> u8 {
    let bits = x.to_bits();
    let sign: u8 = ((bits >> 31) & 1) as u8;

    if x == 0.0 {
        return sign << 7;
    }
    if x.is_nan() {
        let mantissa_bits = variant.mantissa_bits();
        let all_ones_mantissa = ((1u32 << mantissa_bits) - 1) as u8;
        return (sign << 7) | (variant.exp_mask() << mantissa_bits) | all_ones_mantissa;
    }

    let m_bits = variant.mantissa_bits();
    let bias = variant.bias();
    let max_unbiased = variant.max_finite_unbiased_exp();
    let min_unbiased = 1 - bias; // smallest *normal* exponent in the target format

    let saturate = || -> u8 {
        if variant.has_infinity() {
            (sign << 7) | (variant.exp_mask() << m_bits)
        } else {
            // E4M3FN: no infinity -- saturate to the largest finite value (exp=1111, mantissa=110).
            (sign << 7) | (variant.exp_mask() << m_bits) | (((1u32 << m_bits) - 2) as u8)
        }
    };

    if x.is_infinite() {
        return saturate();
    }

    let exp_field_f32 = (bits >> 23) & 0xFF;
    let unbiased_exp = exp_field_f32 as i32 - 127;
    let mantissa_f32 = bits & 0x007F_FFFF;

    if unbiased_exp > max_unbiased {
        return saturate();
    }
    if unbiased_exp < min_unbiased {
        // Too small even for the target format's smallest normal -- flush to zero (see module
        // docs: subnormal outputs aren't supported).
        return sign << 7;
    }

    // 24-bit significand: implicit leading 1 + the 23 mantissa bits.
    let significand = (1u32 << 23) | mantissa_f32;
    let shift = 23 - m_bits;

    // Round-to-nearest-even at `shift`.
    let half = 1u32 << (shift - 1);
    let mask = (1u32 << shift) - 1;
    let remainder = significand & mask;
    let mut rounded = significand >> shift;
    if remainder > half || (remainder == half && (rounded & 1) == 1) {
        rounded += 1;
    }

    let implicit_bit = 1u32 << m_bits;
    let mut exp_out = unbiased_exp;
    if rounded == (implicit_bit << 1) {
        // Rounding carried all the way through the mantissa: bump the exponent instead.
        rounded = implicit_bit;
        exp_out += 1;
    }
    let mantissa_field = rounded - implicit_bit;

    if exp_out > max_unbiased {
        return saturate();
    }

    let biased_exp = (exp_out + bias) as u8;
    (sign << 7) | (biased_exp << m_bits) | (mantissa_field as u8)
}

/// Decodes an `f8` byte (`variant`) back to `f32`. Unlike [`f32_to_f8`], this fully handles
/// subnormals, since it may be asked to decode bytes this module didn't itself produce.
pub fn f8_to_f32(byte: u8, variant: Fp8Variant) -> f32 {
    let m_bits = variant.mantissa_bits();
    let bias = variant.bias();
    let exp_mask = variant.exp_mask();

    let sign = (byte >> 7) & 1;
    let sign_f: f32 = if sign == 1 { -1.0 } else { 1.0 };
    let biased_exp = (byte >> m_bits) & exp_mask;
    let mantissa = (byte & ((1u32 << m_bits) - 1) as u8) as u32;

    if variant == Fp8Variant::E4M3 && biased_exp == exp_mask && mantissa == (1 << m_bits) - 1 {
        return f32::NAN;
    }
    if variant == Fp8Variant::E5M2 && biased_exp == exp_mask {
        return if mantissa == 0 {
            sign_f * f32::INFINITY
        } else {
            f32::NAN
        };
    }

    if biased_exp == 0 {
        if mantissa == 0 {
            return sign_f * 0.0;
        }
        let val = (mantissa as f32) / ((1u32 << m_bits) as f32) * 2f32.powi(1 - bias);
        return sign_f * val;
    }

    let unbiased = biased_exp as i32 - bias;
    let val = (1.0 + (mantissa as f32) / ((1u32 << m_bits) as f32)) * 2f32.powi(unbiased);
    sign_f * val
}

/// An `f32` tensor quantized to `f8` bytes, one [`f32`] scale per group (the tensor is scaled by
/// `1 / scale` before conversion, matching the affine schemes' convention of storing a
/// multiplicative dequantization scale).
#[derive(Debug, Clone)]
pub struct QuantizedFp8 {
    /// One `f8`-encoded byte per input element, same order as the input.
    pub qvalues: Vec<u8>,
    /// One scale per group.
    pub scales: Vec<f32>,
    /// `qvalues[i]`'s group is `groups[i]`.
    pub groups: Vec<u32>,
    /// Which variant `qvalues` was encoded with.
    pub variant: Fp8Variant,
}

/// Quantizes `data` (row-major, shape `shape`) to `variant`, scaling each group so its `amax`
/// maps to the format's largest finite value.
pub fn quantize_fp8(
    data: &[f32],
    shape: &[usize],
    granularity: Granularity,
    variant: Fp8Variant,
) -> QuantizedFp8 {
    let (groups, ngroups) = assign_groups(shape, granularity);

    let mut amax = vec![0f32; ngroups];
    for (i, &x) in data.iter().enumerate() {
        let g = groups[i] as usize;
        amax[g] = amax[g].max(x.abs());
    }

    let target_max = variant.max_finite_value();
    let scales: Vec<f32> = amax
        .iter()
        .map(|&a| if a == 0.0 { 1.0 } else { a / target_max })
        .collect();

    let qvalues: Vec<u8> = data
        .iter()
        .enumerate()
        .map(|(i, &x)| {
            let scale = scales[groups[i] as usize];
            f32_to_f8(x / scale, variant)
        })
        .collect();

    QuantizedFp8 {
        qvalues,
        scales,
        groups,
        variant,
    }
}

/// Reconstructs `f32` values from a [`QuantizedFp8`].
pub fn dequantize_fp8(q: &QuantizedFp8) -> Vec<f32> {
    q.qvalues
        .iter()
        .enumerate()
        .map(|(i, &v)| f8_to_f32(v, q.variant) * q.scales[q.groups[i] as usize])
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn round_trip(x: f32, variant: Fp8Variant, tol: f32) {
        let byte = f32_to_f8(x, variant);
        let back = f8_to_f32(byte, variant);
        assert!(
            (x - back).abs() <= tol,
            "{x:?} -> {byte:#04x} -> {back:?} (variant {variant:?})"
        );
    }

    #[test]
    fn zero_and_signs_round_trip_exactly() {
        for variant in [Fp8Variant::E4M3, Fp8Variant::E5M2] {
            assert_eq!(f8_to_f32(f32_to_f8(0.0, variant), variant), 0.0);
            round_trip(1.0, variant, 0.0);
            round_trip(-1.0, variant, 0.0);
            round_trip(2.0, variant, 0.0);
            round_trip(-2.0, variant, 0.0);
        }
    }

    #[test]
    fn e4m3_max_finite_round_trips_exactly() {
        round_trip(448.0, Fp8Variant::E4M3, 0.0);
        round_trip(-448.0, Fp8Variant::E4M3, 0.0);
    }

    #[test]
    fn e4m3_overflow_saturates_to_max_finite() {
        let byte = f32_to_f8(1.0e6, Fp8Variant::E4M3);
        assert_eq!(f8_to_f32(byte, Fp8Variant::E4M3), 448.0);
        let byte = f32_to_f8(f32::INFINITY, Fp8Variant::E4M3);
        assert_eq!(f8_to_f32(byte, Fp8Variant::E4M3), 448.0);
    }

    #[test]
    fn e5m2_max_finite_round_trips_exactly() {
        round_trip(57344.0, Fp8Variant::E5M2, 0.0);
    }

    #[test]
    fn e5m2_overflow_saturates_to_infinity() {
        let byte = f32_to_f8(1.0e6, Fp8Variant::E5M2);
        assert!(f8_to_f32(byte, Fp8Variant::E5M2).is_infinite());
    }

    #[test]
    fn small_values_flush_to_zero() {
        // Well below E4M3's smallest normal (2^-6).
        let byte = f32_to_f8(1.0e-10, Fp8Variant::E4M3);
        assert_eq!(f8_to_f32(byte, Fp8Variant::E4M3), 0.0);
    }

    #[test]
    fn mid_range_round_trips_within_quantization_error() {
        for variant in [Fp8Variant::E4M3, Fp8Variant::E5M2] {
            for x in [0.1f32, 0.3, 1.5, 3.7, -12.25, 100.0] {
                let byte = f32_to_f8(x, variant);
                let back = f8_to_f32(byte, variant);
                let tol = x.abs() * 0.2 + 0.01; // f8 has very few mantissa bits
                assert!((x - back).abs() < tol, "{x} -> {back} (variant {variant:?})");
            }
        }
    }

    #[test]
    fn per_channel_quantization_uses_independent_scales() {
        let data = vec![0.1f32, -0.1, 200.0, -200.0];
        let shape = [2usize, 2];
        let q = quantize_fp8(
            &data,
            &shape,
            Granularity::PerChannel { axis: 0 },
            Fp8Variant::E4M3,
        );
        assert_eq!(q.scales.len(), 2);
        let deq = dequantize_fp8(&q);
        for (orig, back) in data.iter().zip(deq.iter()) {
            let tol = orig.abs() * 0.3 + 0.01;
            assert!((orig - back).abs() < tol, "{orig} vs {back}");
        }
    }
}
