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

    fn unsqueeze(&self, axis: isize) -> Self;

    fn permute(&self, dims: &[isize]) -> Self;
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DynamicShape {
    dims: Vec<usize>,
}

impl DynamicShape {
    pub fn new(dims: &[usize]) -> Self {
        Self {
            dims: dims.to_vec(),
        }
    }

    pub fn dims(&self) -> &[usize] {
        &self.dims
    }

    pub fn last(&self) -> usize {
        self.dims[self.dims.len() - 1]
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

    fn unsqueeze(&self, axis: isize) -> Self {
        let mut dims = self.dims.clone();
        match axis {
            -1 => dims.push(1),
            1 => dims.insert(0, 1),
            _ => panic!("Invalid axis: {axis}"),
        }

        DynamicShape::new(&dims)
    }

    fn permute(&self, dims: &[isize]) -> Self {
        let mut result = Vec::new();
        for dim in dims {
            result.push(self.dims[*dim as usize]);
        }
        Self { dims: result }
    }
}

impl Index<usize> for DynamicShape {
    type Output = usize;

    fn index(&self, index: usize) -> &Self::Output {
        &self.dims[index]
    }
}

// Implement Index for ranges
impl Index<std::ops::Range<usize>> for DynamicShape {
    type Output = [usize];

    fn index(&self, index: std::ops::Range<usize>) -> &Self::Output {
        &self.dims[index]
    }
}

impl Index<std::ops::RangeFrom<usize>> for DynamicShape {
    type Output = [usize];

    fn index(&self, index: std::ops::RangeFrom<usize>) -> &Self::Output {
        &self.dims[index]
    }
}

impl Index<std::ops::RangeTo<usize>> for DynamicShape {
    type Output = [usize];

    fn index(&self, index: std::ops::RangeTo<usize>) -> &Self::Output {
        &self.dims[index]
    }
}

impl Index<std::ops::RangeFull> for DynamicShape {
    type Output = [usize];

    fn index(&self, _index: std::ops::RangeFull) -> &Self::Output {
        &self.dims
    }
}

#[macro_export]
macro_rules! shape {
    ($($dim:expr),*) => {
        $crate::graph::shape::DynamicShape::new(&[$($dim),*])
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

    #[test]
    fn test_array_access() {
        let shape = DynamicShape::new(&[10, 20, 30, 40]);

        // Single index access
        assert_eq!(shape[0], 10);
        assert_eq!(shape[1], 20);
        assert_eq!(shape[2], 30);
        assert_eq!(shape[3], 40);

        // Range access
        assert_eq!(shape[0..2], [10, 20]);
        assert_eq!(shape[1..], [20, 30, 40]);
        assert_eq!(shape[..3], [10, 20, 30]);
        assert_eq!(shape[..], [10, 20, 30, 40]);
    }
}
