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

use alloc::{collections::BTreeMap, vec::Vec};

use super::{DtypeRepr, Shape};

/// A constant tensor's actual values, carried alongside the graph.
///
/// The graph IR records a constant's dtype and shape but not its contents, since
/// most constants are weights that are bound separately. Shape plumbing is the
/// exception: an exporter-emitted reshape target or pad list *is* its value, and a
/// lowering pass that cannot read it cannot lower the ops that consume it. A
/// [`ConstantMap`] supplies those values, keyed by node index.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ConstantTensor {
    /// Element type.
    pub dtype: DtypeRepr,
    /// Tensor shape.
    pub shape: Shape,
    /// Little-endian element bytes, in row-major order.
    pub bytes: Vec<u8>,
}

/// Why a [`ConstantTensor`]'s bytes cannot be read as the element type asked for.
#[derive(thiserror::Error, Debug, Clone, PartialEq, Eq)]
pub enum ConstantReadError {
    /// The shape has a dynamic dimension, so the element count is not known.
    #[error("constant has a dynamic dimension in shape {shape:?}, so its element count is unknown")]
    DynamicShape {
        /// The offending shape.
        shape: Shape,
    },
    /// The shape's element count does not fit in a `usize`.
    #[error("constant shape {shape:?} has more elements than fit in a usize")]
    ElementCountOverflow {
        /// The offending shape.
        shape: Shape,
    },
    /// The dtype cannot be converted to the element type asked for.
    #[error("cannot read a {dtype:?} constant as {target}")]
    UnsupportedDtype {
        /// The constant's dtype.
        dtype: DtypeRepr,
        /// The element type asked for.
        target: &'static str,
    },
    /// There are fewer bytes than the shape and dtype call for.
    #[error("{dtype:?} constant of shape {shape:?} needs {needed} bytes but has {actual}")]
    TooFewBytes {
        /// The constant's dtype.
        dtype: DtypeRepr,
        /// The constant's shape.
        shape: Shape,
        /// Bytes the shape and dtype call for.
        needed: usize,
        /// Bytes present.
        actual: usize,
    },
    /// A `u64` element is above `i64::MAX`.
    #[error("u64 constant element {index} is {value}, which exceeds i64::MAX")]
    OutOfRange {
        /// The element's position, in row-major order.
        index: usize,
        /// The element's value.
        value: u64,
    },
}

impl ConstantTensor {
    /// Decodes the tensor as `i64`s, which is how shape vectors, axes, indices and pad
    /// lists are consumed.
    ///
    /// Fails for a float tensor, whose elements are not integers, for a `u64` element
    /// above `i64::MAX`, for a dynamic shape, or if the bytes are short for the shape.
    pub fn to_i64_vec(&self) -> Result<Vec<i64>, ConstantReadError> {
        let n = self.element_count()?;
        match self.dtype {
            DtypeRepr::Bool | DtypeRepr::U8 => self.decode(n, |[b]: [u8; 1]| b as i64),
            DtypeRepr::I8 => self.decode(n, |c| i8::from_le_bytes(c) as i64),
            DtypeRepr::I16 => self.decode(n, |c| i16::from_le_bytes(c) as i64),
            DtypeRepr::U16 => self.decode(n, |c| u16::from_le_bytes(c) as i64),
            DtypeRepr::I32 => self.decode(n, |c| i32::from_le_bytes(c) as i64),
            DtypeRepr::U32 => self.decode(n, |c| u32::from_le_bytes(c) as i64),
            DtypeRepr::I64 => self.decode(n, i64::from_le_bytes),
            DtypeRepr::U64 => self
                .decode(n, u64::from_le_bytes)?
                .into_iter()
                .enumerate()
                .map(|(index, value)| {
                    i64::try_from(value).map_err(|_| ConstantReadError::OutOfRange { index, value })
                })
                .collect(),
            DtypeRepr::F16 | DtypeRepr::BF16 | DtypeRepr::F32 | DtypeRepr::F64 => {
                Err(ConstantReadError::UnsupportedDtype {
                    dtype: self.dtype,
                    target: "i64",
                })
            }
        }
    }

    /// Decodes the tensor as `f64`s, which is how a scalar bound or scale is consumed.
    ///
    /// Integers are widened rather than refused: ONNX types a `Clip` bound to match
    /// its input, so the same operand position carries a float for a float tensor and
    /// an integer for an integer one, and a caller asking for a bound wants the value
    /// either way. Integers above 2^53 lose precision in the widening. F16 and BF16
    /// are refused, since nothing here decodes them.
    pub fn to_f64_vec(&self) -> Result<Vec<f64>, ConstantReadError> {
        match self.dtype {
            DtypeRepr::F32 => self.decode(self.element_count()?, |c| f32::from_le_bytes(c) as f64),
            DtypeRepr::F64 => self.decode(self.element_count()?, f64::from_le_bytes),
            DtypeRepr::F16 | DtypeRepr::BF16 => Err(ConstantReadError::UnsupportedDtype {
                dtype: self.dtype,
                target: "f64",
            }),
            // Every integer and bool case is already handled by `to_i64_vec`.
            _ => Ok(self.to_i64_vec()?.into_iter().map(|i| i as f64).collect()),
        }
    }

    /// Decodes the first `n` little-endian `W`-byte elements with `f`.
    fn decode<const W: usize, T>(
        &self,
        n: usize,
        f: impl Fn([u8; W]) -> T,
    ) -> Result<Vec<T>, ConstantReadError> {
        let (chunks, _) = self.bytes.as_chunks::<W>();
        if chunks.len() < n {
            return Err(ConstantReadError::TooFewBytes {
                dtype: self.dtype,
                shape: self.shape.clone(),
                needed: n.saturating_mul(W),
                actual: self.bytes.len(),
            });
        }
        Ok(chunks[..n].iter().map(|&c| f(c)).collect())
    }

    /// The number of elements, or why it is not known.
    fn element_count(&self) -> Result<usize, ConstantReadError> {
        self.shape.iter().try_fold(1usize, |acc, &d| {
            let d = d.ok_or_else(|| ConstantReadError::DynamicShape {
                shape: self.shape.clone(),
            })?;
            acc.checked_mul(d)
                .ok_or_else(|| ConstantReadError::ElementCountOverflow {
                    shape: self.shape.clone(),
                })
        })
    }

    /// The number of elements, if the shape is fully static and its product fits in
    /// a `usize`.
    pub fn len(&self) -> Option<usize> {
        self.element_count().ok()
    }

    /// Whether the tensor has no elements.
    pub fn is_empty(&self) -> bool {
        self.len() == Some(0)
    }
}

/// Materialised constant values, keyed by the index of the [`GraphNode`](super::GraphNode) that
/// produces them.
pub type ConstantMap = BTreeMap<usize, ConstantTensor>;

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::{string::ToString, vec};

    fn constant(dtype: DtypeRepr, shape: &[Option<usize>], bytes: Vec<u8>) -> ConstantTensor {
        ConstantTensor {
            dtype,
            shape: shape.to_vec(),
            bytes,
        }
    }

    fn le_bytes<const W: usize>(values: impl IntoIterator<Item = [u8; W]>) -> Vec<u8> {
        values.into_iter().flatten().collect()
    }

    #[test]
    fn every_integer_dtype_widens_to_i64() {
        let cases = [
            (DtypeRepr::Bool, vec![1u8, 0], vec![1i64, 0]),
            (DtypeRepr::U8, vec![255, 7], vec![255, 7]),
            (DtypeRepr::I8, vec![0xff, 7], vec![-1, 7]),
            (
                DtypeRepr::I16,
                le_bytes([(-300i16).to_le_bytes(), 2i16.to_le_bytes()]),
                vec![-300, 2],
            ),
            (
                DtypeRepr::U16,
                le_bytes([60_000u16.to_le_bytes(), 2u16.to_le_bytes()]),
                vec![60_000, 2],
            ),
            (
                DtypeRepr::I32,
                le_bytes([(-70_000i32).to_le_bytes(), 2i32.to_le_bytes()]),
                vec![-70_000, 2],
            ),
            (
                DtypeRepr::U32,
                le_bytes([4_000_000_000u32.to_le_bytes(), 2u32.to_le_bytes()]),
                vec![4_000_000_000, 2],
            ),
            (
                DtypeRepr::I64,
                le_bytes([i64::MIN.to_le_bytes(), 2i64.to_le_bytes()]),
                vec![i64::MIN, 2],
            ),
            (
                DtypeRepr::U64,
                le_bytes([(i64::MAX as u64).to_le_bytes(), 2u64.to_le_bytes()]),
                vec![i64::MAX, 2],
            ),
        ];
        for (dtype, bytes, want) in cases {
            let t = constant(dtype, &[Some(2)], bytes);
            assert_eq!(t.to_i64_vec(), Ok(want), "{dtype:?}");
        }
    }

    /// A `u64` above `i64::MAX` used to wrap to a negative number, which in a shape or
    /// pad list reads as a nonsense dimension rather than an error.
    #[test]
    fn a_u64_above_i64_max_is_out_of_range_not_wrapped() {
        let t = constant(
            DtypeRepr::U64,
            &[Some(2)],
            le_bytes([1u64.to_le_bytes(), u64::MAX.to_le_bytes()]),
        );
        assert_eq!(
            t.to_i64_vec(),
            Err(ConstantReadError::OutOfRange {
                index: 1,
                value: u64::MAX,
            })
        );
        // The f64 reader goes through the i64 one, so it refuses the same element.
        assert!(matches!(
            t.to_f64_vec(),
            Err(ConstantReadError::OutOfRange { index: 1, .. })
        ));
    }

    #[test]
    fn floats_are_refused_as_i64() {
        for dtype in [
            DtypeRepr::F16,
            DtypeRepr::BF16,
            DtypeRepr::F32,
            DtypeRepr::F64,
        ] {
            let t = constant(dtype, &[Some(1)], vec![0; 8]);
            assert_eq!(
                t.to_i64_vec(),
                Err(ConstantReadError::UnsupportedDtype {
                    dtype,
                    target: "i64"
                }),
            );
        }
    }

    #[test]
    fn floats_and_integers_read_as_f64_but_half_floats_do_not() {
        let f32s = constant(
            DtypeRepr::F32,
            &[Some(2)],
            le_bytes([1.5f32.to_le_bytes(), (-2.0f32).to_le_bytes()]),
        );
        assert_eq!(f32s.to_f64_vec(), Ok(vec![1.5, -2.0]));

        let f64s = constant(DtypeRepr::F64, &[Some(1)], 0.25f64.to_le_bytes().to_vec());
        assert_eq!(f64s.to_f64_vec(), Ok(vec![0.25]));

        let ints = constant(DtypeRepr::I32, &[Some(1)], (-3i32).to_le_bytes().to_vec());
        assert_eq!(ints.to_f64_vec(), Ok(vec![-3.0]));

        for dtype in [DtypeRepr::F16, DtypeRepr::BF16] {
            let t = constant(dtype, &[Some(1)], vec![0; 2]);
            assert_eq!(
                t.to_f64_vec(),
                Err(ConstantReadError::UnsupportedDtype {
                    dtype,
                    target: "f64"
                }),
            );
        }
    }

    #[test]
    fn short_bytes_are_refused_with_the_counts() {
        let t = constant(DtypeRepr::I32, &[Some(2), Some(2)], vec![0; 12]);
        assert_eq!(
            t.to_i64_vec(),
            Err(ConstantReadError::TooFewBytes {
                dtype: DtypeRepr::I32,
                shape: vec![Some(2), Some(2)],
                needed: 16,
                actual: 12,
            })
        );
    }

    #[test]
    fn a_dynamic_shape_is_refused_and_has_no_len() {
        let t = constant(DtypeRepr::I64, &[None, Some(2)], vec![0; 16]);
        assert_eq!(
            t.to_i64_vec(),
            Err(ConstantReadError::DynamicShape {
                shape: vec![None, Some(2)],
            })
        );
        assert_eq!(t.len(), None);
    }

    #[test]
    fn an_overflowing_element_count_is_refused() {
        let t = constant(DtypeRepr::U8, &[Some(usize::MAX), Some(2)], vec![]);
        assert_eq!(
            t.to_i64_vec(),
            Err(ConstantReadError::ElementCountOverflow {
                shape: vec![Some(usize::MAX), Some(2)],
            })
        );
        assert_eq!(t.len(), None);
    }

    /// A rank-0 tensor is a scalar: one element, not none.
    #[test]
    fn a_scalar_has_one_element() {
        let t = constant(DtypeRepr::I64, &[], 42i64.to_le_bytes().to_vec());
        assert_eq!(t.len(), Some(1));
        assert_eq!(t.to_i64_vec(), Ok(vec![42]));
    }

    #[test]
    fn errors_name_the_values_involved() {
        let e = ConstantReadError::TooFewBytes {
            dtype: DtypeRepr::I32,
            shape: vec![Some(4)],
            needed: 16,
            actual: 12,
        };
        assert_eq!(
            e.to_string(),
            "I32 constant of shape [Some(4)] needs 16 bytes but has 12"
        );
    }
}
