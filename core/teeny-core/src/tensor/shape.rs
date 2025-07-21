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

use std::ops::Index;

/// A type-level tuple for representing tensor shapes
pub trait Shape {
    const RANK: usize;
    type Dims: AsRef<[usize]> + Clone + Index<usize, Output = usize>;

    fn dims(&self) -> Self::Dims;

    fn broadcast(&self, other: &Self) -> Self;
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DynamicShape {
    pub dims: Vec<usize>,
}

impl DynamicShape {
    pub fn new(dims: &[usize]) -> Self {
        Self {
            dims: dims.to_vec(),
        }
    }
}

impl<T: AsRef<[usize]>> From<T> for DynamicShape {
    fn from(dims: T) -> Self {
        Self {
            dims: dims.as_ref().to_vec(),
        }
    }
}

impl Shape for DynamicShape {
    const RANK: usize = 0; // This will be determined at runtime
    type Dims = Vec<usize>;

    fn dims(&self) -> Self::Dims {
        self.dims.clone()
    }

    fn broadcast(&self, other: &Self) -> Self {
        let max_rank = self.dims.len().max(other.dims.len());
        let mut result = Vec::new();

        for i in 0..max_rank {
            let lhs_dim = self
                .dims
                .get(self.dims.len().wrapping_sub(1).wrapping_sub(i))
                .copied()
                .unwrap_or(1);
            let rhs_dim = other
                .dims
                .get(other.dims.len().wrapping_sub(1).wrapping_sub(i))
                .copied()
                .unwrap_or(1);

            if lhs_dim == rhs_dim || lhs_dim == 1 || rhs_dim == 1 {
                result.push(lhs_dim.max(rhs_dim));
            } else {
                panic!("Shapes {self:?} and {other:?} are not broadcastable",);
            }
        }

        result.reverse();
        Self { dims: result }
    }
}

#[macro_export]
macro_rules! shape {
    ($($dim:expr),*) => {
        $crate::tensor::shape::DynamicShape {
            dims: vec![$($dim),*],
        }
    };
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_broadcast() {
        let shape1 = DynamicShape::new(&[1, 2, 3]);
        let shape2 = DynamicShape::new(&[4, 5]);

        let result = shape1.broadcast(&shape2);
        assert_eq!(result.dims, vec![4, 5, 3]);
    }
}
