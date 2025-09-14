/*
 * Copyright (c) 2025 Teenygrad. All rights reserved.
 *
 * This program is free software: you can redistribute it and/or modify it
 * under the terms of the GNU General Public License as published by the
 * Free Software Foundation, either version 3 of the License, or (at your
 * option) any later version.
 *
 * This program is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program. If not, see <https://www.gnu.org/licenses/>.
 */

use crate::{
    error::Error,
    fxgraph::{
        dtype::DType,
        shape::{Shape, SymInt},
        tensor::Tensor,
    },
};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TyTensor {
    pub dtype: DType,
    pub shape: Vec<SymInt>,
}

impl TyTensor {
    pub fn new(tensor: &Tensor) -> Self {
        Self {
            dtype: tensor.dtype,
            shape: tensor.shape.shape.clone(),
        }
    }

    pub fn new_from_dtype_and_shape(dtype: DType, shape: &Shape) -> Self {
        Self {
            dtype,
            shape: shape.shape.clone(),
        }
    }

    pub fn rank(&self) -> usize {
        self.shape.len()
    }

    pub fn dtype_compatible(&self, other: &TyTensor) -> bool {
        self.dtype.promote(&other.dtype).is_ok()
    }

    pub fn broadcast(&self, other: &TyTensor) -> Result<TyTensor, Error> {
        if self.rank() < other.rank() {
            return other.broadcast(self);
        }

        let mut result_shape = Vec::with_capacity(self.rank());

        let shape_a = &self.shape;
        let shape_b = {
            let mut v = vec![SymInt::from(1); self.rank() - other.rank()];
            v.extend_from_slice(&other.shape);
            v
        };

        for (dim_a, dim_b) in shape_a.iter().zip(shape_b.iter()) {
            match (dim_a, dim_b) {
                (SymInt::Int(1), SymInt::Int(a)) => {
                    result_shape.push(SymInt::Int(*a));
                }
                (SymInt::Int(a), SymInt::Int(1)) => {
                    result_shape.push(SymInt::Int(*a));
                }
                (SymInt::Int(a), SymInt::Int(b)) => {
                    if a == b {
                        result_shape.push(SymInt::Int(*a));
                    } else {
                        return Err(Error::InvalidTensorBroadcast(format!(
                            "Cannot broadcast dimensions: {:?} and {:?}",
                            self.shape, other.shape
                        )));
                    }
                }
                _ => {
                    return Err(Error::InvalidTensorBroadcast(format!(
                        "Symbolic dimensions are not supported for broadcasting: {:?} and {:?}",
                        self.shape, other.shape
                    )));
                }
            }
        }

        let dtype = self.dtype.promote(&other.dtype)?;
        let ty_tensor = TyTensor {
            dtype,
            shape: result_shape,
        };

        Ok(ty_tensor)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_valid_broadcast() {
        let a = TyTensor::new_from_dtype_and_shape(DType::F32, &Shape::new(&[3.into(), 4.into()]));
        let b = TyTensor::new_from_dtype_and_shape(
            DType::F32,
            &Shape::new(&[2.into(), 3.into(), 4.into()]),
        );

        let c = a.broadcast(&b).unwrap();
        assert_eq!(c.shape, vec![2.into(), 3.into(), 4.into()]);
    }

    #[test]
    fn test_invalid_broadcast() {
        let a = TyTensor::new_from_dtype_and_shape(DType::F32, &Shape::new(&[3.into(), 4.into()]));
        let b = TyTensor::new_from_dtype_and_shape(
            DType::F32,
            &Shape::new(&[2.into(), 3.into(), 5.into()]),
        );

        let c = a.broadcast(&b);
        assert!(c.is_err());
    }
}
