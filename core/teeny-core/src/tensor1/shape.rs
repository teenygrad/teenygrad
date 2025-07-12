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
    fn dims() -> Self::Dims;
}

/// A 0-dimensional tensor (scalar)
pub struct ScalarShape;
impl Shape for ScalarShape {
    const RANK: usize = 0;
    type Dims = [usize; 0];
    fn dims() -> Self::Dims {
        []
    }
}

/// A 1-dimensional tensor
pub struct Dim1<const N: usize>;
impl<const N: usize> Shape for Dim1<N> {
    const RANK: usize = 1;
    type Dims = [usize; 1];
    fn dims() -> Self::Dims {
        [N]
    }
}

/// A 2-dimensional tensor
pub struct Dim2<const M: usize, const N: usize>;
impl<const M: usize, const N: usize> Shape for Dim2<M, N> {
    const RANK: usize = 2;
    type Dims = [usize; 2];
    fn dims() -> Self::Dims {
        [M, N]
    }
}

/// A 3-dimensional tensor
pub struct Dim3<const M: usize, const N: usize, const P: usize>;
impl<const M: usize, const N: usize, const P: usize> Shape for Dim3<M, N, P> {
    const RANK: usize = 3;
    type Dims = [usize; 3];
    fn dims() -> Self::Dims {
        [M, N, P]
    }
}

/// A dynamically sized tensor shape
pub struct DynamicShape {
    pub dims: Vec<usize>,
}

impl DynamicShape {
    pub fn new(dims: Vec<usize>) -> Self {
        Self { dims }
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

    fn dims() -> Self::Dims {
        Vec::new() // This will be overridden at runtime
    }
}
