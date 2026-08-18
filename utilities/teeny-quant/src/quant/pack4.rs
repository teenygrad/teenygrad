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

//! INT4 nibble packing.
//!
//! safetensors 0.7 has no native `I4`/`U4` dtype, so packed INT4 tensors are stored as plain
//! `U8`. **Packing layout** (this crate's own convention -- not bit-for-bit compatible with any
//! particular GPTQ/AWQ int32-packing scheme, which vary across implementations and versions):
//! two consecutive elements in row-major order share one byte, the first in the low nibble and
//! the second in the high nibble, each a 4-bit two's-complement value in `-8..=7`. A tensor with
//! an odd element count gets one trailing byte whose high nibble is unused padding (`0`). The
//! packed tensor's logical shape and element count are recorded separately in the
//! `quantization_config` metadata (see [`crate::format`]) since the packed `U8` tensor's own
//! shape (`[ceil(n / 2)]`) doesn't reflect it.

/// Packs 4-bit two's-complement values (each expected to be in `-8..=7`; out-of-range bits above
/// the low nibble are silently dropped) two-per-byte, low nibble first.
pub fn pack_i4(values: &[i8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(values.len().div_ceil(2));
    for pair in values.chunks(2) {
        let lo = (pair[0] as u8) & 0x0F;
        let hi = pair.get(1).map(|&v| (v as u8) & 0x0F).unwrap_or(0);
        out.push(lo | (hi << 4));
    }
    out
}

/// Inverse of [`pack_i4`]: unpacks `n` sign-extended 4-bit values from `bytes`.
pub fn unpack_i4(bytes: &[u8], n: usize) -> Vec<i8> {
    fn sign_extend(nibble: u8) -> i8 {
        if nibble >= 8 {
            nibble as i8 - 16
        } else {
            nibble as i8
        }
    }

    let mut out = Vec::with_capacity(n);
    for &byte in bytes {
        if out.len() >= n {
            break;
        }
        out.push(sign_extend(byte & 0x0F));
        if out.len() < n {
            out.push(sign_extend((byte >> 4) & 0x0F));
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trips_full_int4_range() {
        let values: Vec<i8> = (-8..=7).collect();
        let packed = pack_i4(&values);
        assert_eq!(packed.len(), values.len().div_ceil(2));
        let unpacked = unpack_i4(&packed, values.len());
        assert_eq!(values, unpacked);
    }

    #[test]
    fn odd_length_pads_trailing_nibble() {
        let values = vec![-8i8, 7, -1];
        let packed = pack_i4(&values);
        assert_eq!(packed.len(), 2);
        let unpacked = unpack_i4(&packed, values.len());
        assert_eq!(values, unpacked);
    }

    #[test]
    fn empty_round_trips() {
        let values: Vec<i8> = vec![];
        let packed = pack_i4(&values);
        assert!(packed.is_empty());
        assert!(unpack_i4(&packed, 0).is_empty());
    }
}
