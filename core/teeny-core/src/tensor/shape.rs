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
}

/// A dynamically sized tensor shape
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
}

#[macro_export]
macro_rules! shape {
    ($($dim:expr),*) => {
        DynamicShape {
            dims: vec![$($dim),*],
        }
    };
}
