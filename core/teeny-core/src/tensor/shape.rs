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

#[cfg(feature = "ndarray")]
impl From<DynamicShape> for ndarray::Shape<ndarray::IxDyn> {
    fn from(shape: DynamicShape) -> Self {
        use ndarray::IntoDimension;

        ndarray::Shape::from(&shape.dims.into_dimension())
    }
}

// #[cfg(feature = "ndarray")]
// impl From<ScalarShape> for ndarray::Shape<ndarray::Ix0> {
//     fn from(_: ScalarShape) -> Self {
//         ndarray::Shape::from(&ScalarShape::dims())
//     }
// }

// #[cfg(feature = "ndarray")]
// impl<const N: usize> From<Dim1<N>> for ndarray::IxDyn {
//     fn from(_: Dim1<N>) -> Self {
//         ndarray::IxDyn(&[N])
//     }
// }
