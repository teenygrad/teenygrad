/*
 * Copyright (C) 2025 Teenygrad. All rights reserved.
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

use crate::types::NumericType;

/// A type-level tuple for representing tensor shapes
pub trait Shape {
    const RANK: usize;
    type Dims: AsRef<[usize]>;
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

impl Shape for DynamicShape {
    const RANK: usize = 0; // This will be determined at runtime
    type Dims = Vec<usize>;
    fn dims() -> Self::Dims {
        Vec::new() // This will be overridden at runtime
    }
}

/// The Tensor trait now takes a shape parameter
pub trait Tensor<S: Shape> {
    type Element: NumericType;

    /// Get the shape of the tensor
    fn shape() -> S::Dims {
        S::dims()
    }

    /// Get the rank of the tensor
    fn rank() -> usize {
        S::RANK
    }
}

// Example implementation for a concrete tensor type
pub struct DenseTensor<S: Shape, T: NumericType> {
    pub data: Vec<T::RustType>,
    pub _shape: std::marker::PhantomData<S>,
}

impl<S: Shape, T: NumericType> DenseTensor<S, T> {
    pub fn new(data: Vec<T::RustType>) -> Self {
        Self {
            data,
            _shape: std::marker::PhantomData,
        }
    }
}

impl<S: Shape, T: NumericType> Tensor<S> for DenseTensor<S, T> {
    type Element = T;
}

/// Trait for broadcasting shapes
pub trait BroadcastTo<Target: Shape> {
    type Output: Shape;
}

/// Implement broadcasting for same shapes (no broadcast needed)
impl<S: Shape> BroadcastTo<S> for S {
    type Output = S;
}

/// Broadcast 1D to 2D by repeating along first dimension
impl<const N: usize, const M: usize> BroadcastTo<Dim2<M, N>> for Dim1<N> {
    type Output = Dim2<M, N>;
}

/// Broadcast 1D to 3D by repeating along first two dimensions
impl<const N: usize, const M: usize, const P: usize> BroadcastTo<Dim3<M, P, N>> for Dim1<N> {
    type Output = Dim3<M, P, N>;
}

/// Broadcast 2D to 3D by repeating along first dimension
impl<const M: usize, const N: usize, const P: usize> BroadcastTo<Dim3<P, M, N>> for Dim2<M, N> {
    type Output = Dim3<P, M, N>;
}

/// Extension trait for broadcasting operations
pub trait Broadcast {
    type Shape: Shape;
    type Element: NumericType;

    /// Broadcast this tensor to match the target shape
    fn broadcast_to<Target: Shape>(self) -> DenseTensor<Target, Self::Element>
    where
        Self::Shape: BroadcastTo<Target, Output = Target>;
}

impl<S: Shape, T: NumericType> Broadcast for DenseTensor<S, T>
where
    T::RustType: Clone,
{
    type Shape = S;
    type Element = T;

    fn broadcast_to<Target: Shape>(self) -> DenseTensor<Target, T>
    where
        S: BroadcastTo<Target, Output = Target>,
    {
        // For now, we'll just clone the data
        // In a real implementation, you'd need to handle the actual broadcasting logic
        DenseTensor {
            data: self.data,
            _shape: std::marker::PhantomData,
        }
    }
}

/// Common shape for broadcasting two shapes
pub trait BroadcastShape<Rhs: Shape> {
    type Output: Shape;
}

impl<S: Shape> BroadcastShape<S> for S {
    type Output = S;
}

// Example of how to use broadcasting in operations
pub trait Add<Rhs> {
    type Output;
    fn add(self, rhs: Rhs) -> Self::Output;
}

impl<S1: Shape, S2: Shape, T: NumericType> Add<DenseTensor<S2, T>> for DenseTensor<S1, T>
where
    S1: BroadcastShape<S2>,
    S2: BroadcastShape<S1>,
    <S1 as BroadcastShape<S2>>::Output: Shape,
    T::RustType: Clone + std::ops::Add<Output = T::RustType>,
{
    type Output = DenseTensor<<S1 as BroadcastShape<S2>>::Output, T>;

    fn add(self, _rhs: DenseTensor<S2, T>) -> Self::Output {
        // Broadcast both tensors to the same shape and then add
        let broadcasted_self = self.broadcast_to();

        // In a real implementation, you'd need to handle the actual addition
        // This is just a placeholder
        DenseTensor {
            data: broadcasted_self.data,
            _shape: std::marker::PhantomData,
        }
    }
}
