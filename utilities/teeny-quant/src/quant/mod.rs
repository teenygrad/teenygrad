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

//! Weight quantization primitives: INT8/INT4 affine quantization (`affine`, `pack4`) and
//! `F8_E4M3`/`F8_E5M2` (`fp8`), each parameterized by a [`Granularity`].

pub mod affine;
pub mod fp8;
pub mod granularity;
pub(crate) mod groups;
pub mod pack4;

pub use affine::{AffineParams, QuantizedAffine, dequantize_affine, quantize_affine};

/// Computes the flat-index -> group-id table for a tensor of `shape` under `granularity`, and
/// the resulting group count. Quantizing functions in this module compute and use this
/// internally; it's exposed so callers reconstructing a quantized tensor (e.g.
/// [`crate::validate`], or an external consumer of `teeny-quant`'s output) can recompute the
/// same element -> group mapping without duplicating the grouping logic.
pub fn compute_groups(shape: &[usize], granularity: Granularity) -> (Vec<u32>, usize) {
    groups::assign_groups(shape, granularity)
}
pub use fp8::{Fp8Variant, QuantizedFp8, dequantize_fp8, f8_to_f32, f32_to_f8, quantize_fp8};
pub use granularity::Granularity;

/// Which quantization scheme to apply to a tensor.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Scheme {
    /// 8-bit affine integer quantization.
    Int8 {
        /// Symmetric (`zero_point = 0`) vs asymmetric.
        symmetric: bool,
    },
    /// 4-bit affine integer quantization, nibble-packed into `U8` (see [`pack4`]).
    Int4 {
        /// Symmetric (`zero_point = 0`) vs asymmetric.
        symmetric: bool,
    },
    /// 8-bit floating point.
    Fp8 {
        /// Which OCP FP8 encoding.
        variant: Fp8Variant,
    },
}

impl Scheme {
    /// The compressed-tensors `weights.type` field for this scheme.
    pub fn type_name(self) -> &'static str {
        match self {
            Scheme::Int8 { .. } | Scheme::Int4 { .. } => "int",
            Scheme::Fp8 { .. } => "float",
        }
    }

    /// The compressed-tensors `weights.num_bits` field for this scheme.
    pub fn num_bits(self) -> u8 {
        match self {
            Scheme::Int8 { .. } => 8,
            Scheme::Int4 { .. } => 4,
            Scheme::Fp8 { .. } => 8,
        }
    }

    /// Whether this scheme is symmetric (`zero_point = 0` / no zero-point tensor written).
    pub fn is_symmetric(self) -> bool {
        match self {
            Scheme::Int8 { symmetric } | Scheme::Int4 { symmetric } => symmetric,
            // FP8 quantization here is amax-scaled with no zero-point, i.e. always symmetric.
            Scheme::Fp8 { .. } => true,
        }
    }

    /// A short, stable name used in CLI output and file names (e.g. `int8`, `int4`, `fp8_e4m3`).
    pub fn short_name(self) -> String {
        match self {
            Scheme::Int8 { .. } => "int8".to_string(),
            Scheme::Int4 { .. } => "int4".to_string(),
            Scheme::Fp8 {
                variant: Fp8Variant::E4M3,
            } => "fp8_e4m3".to_string(),
            Scheme::Fp8 {
                variant: Fp8Variant::E5M2,
            } => "fp8_e5m2".to_string(),
        }
    }
}
