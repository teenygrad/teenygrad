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

use std::ops::{Mul, MulAssign};

use crate::{
    tensor::{Shape, Tensor},
    types::NumericType,
};

// Tensor * Tensor operations (element-wise multiplication)
impl<S: Shape, T: NumericType> Mul<Box<dyn Tensor<S, Element = T>>>
    for Box<dyn Tensor<S, Element = T>>
{
    type Output = Box<dyn Tensor<S, Element = T>>;

    fn mul(self, rhs: Box<dyn Tensor<S, Element = T>>) -> Self::Output {
        // Create a computation graph node for element-wise multiplication
        Box::new(MulNode { lhs: self, rhs })
    }
}

// Tensor * Scalar operations
impl<S: Shape, T: NumericType> Mul<T::RustType> for Box<dyn Tensor<S, Element = T>> {
    type Output = Box<dyn Tensor<S, Element = T>>;

    fn mul(self, rhs: T::RustType) -> Self::Output {
        // Create a computation graph node for scalar multiplication
        Box::new(ScalarMulNode {
            tensor: self,
            scalar: rhs,
        })
    }
}

// Scalar * Tensor operations
impl<S: Shape, T: NumericType> Mul<Box<dyn Tensor<S, Element = T>>> for T::RustType {
    type Output = Box<dyn Tensor<S, Element = T>>;

    fn mul(self, rhs: Box<dyn Tensor<S, Element = T>>) -> Self::Output {
        // Create a computation graph node for scalar multiplication (commutative)
        Box::new(ScalarMulNode {
            tensor: rhs,
            scalar: self,
        })
    }
}

// MulAssign implementations for in-place operations
impl<S: Shape, T: NumericType> MulAssign<Box<dyn Tensor<S, Element = T>>>
    for Box<dyn Tensor<S, Element = T>>
{
    fn mul_assign(&mut self, rhs: Box<dyn Tensor<S, Element = T>>) {
        *self = Box::new(MulNode {
            lhs: std::mem::replace(self, Box::new(EmptyNode(std::marker::PhantomData))),
            rhs,
        });
    }
}

impl<S: Shape, T: NumericType> MulAssign<T::RustType> for Box<dyn Tensor<S, Element = T>> {
    fn mul_assign(&mut self, rhs: T::RustType) {
        *self = Box::new(ScalarMulNode {
            tensor: std::mem::replace(self, Box::new(EmptyNode(std::marker::PhantomData))),
            scalar: rhs,
        });
    }
}

// Computation graph nodes for lazy evaluation

/// Node representing element-wise tensor multiplication
struct MulNode<S: Shape, T: NumericType> {
    lhs: Box<dyn Tensor<S, Element = T>>,
    rhs: Box<dyn Tensor<S, Element = T>>,
}

impl<S: Shape, T: NumericType> Tensor<S> for MulNode<S, T> {
    type Element = T;

    fn shape(&self) -> S::Dims {
        // Both tensors should have the same shape for element-wise multiplication
        self.lhs.shape()
    }

    fn reshape(&self, shape: &[isize]) -> Box<dyn Tensor<S, Element = T>> {
        Box::new(MulNode {
            lhs: self.lhs.reshape(shape),
            rhs: self.rhs.reshape(shape),
        })
    }

    fn shape_of(&self, index: isize) -> usize {
        self.lhs.shape_of(index)
    }

    fn stride(&self, index: isize) -> usize {
        self.lhs.stride(index)
    }

    fn rank(&self) -> usize {
        self.lhs.rank()
    }

    fn device(&self) -> crate::device::Device {
        self.lhs.device()
    }

    fn to_device(&self, device: crate::device::Device) -> Box<dyn Tensor<S, Element = T>> {
        Box::new(MulNode {
            lhs: self.lhs.to_device(device),
            rhs: self.rhs.to_device(device),
        })
    }
}

/// Node representing scalar multiplication
struct ScalarMulNode<S: Shape, T: NumericType> {
    tensor: Box<dyn Tensor<S, Element = T>>,
    scalar: T::RustType,
}

impl<S: Shape, T: NumericType> Tensor<S> for ScalarMulNode<S, T> {
    type Element = T;

    fn shape(&self) -> S::Dims {
        self.tensor.shape()
    }

    fn reshape(&self, shape: &[isize]) -> Box<dyn Tensor<S, Element = T>> {
        Box::new(ScalarMulNode {
            tensor: self.tensor.reshape(shape),
            scalar: self.scalar.clone(),
        })
    }

    fn shape_of(&self, index: isize) -> usize {
        self.tensor.shape_of(index)
    }

    fn stride(&self, index: isize) -> usize {
        self.tensor.stride(index)
    }

    fn rank(&self) -> usize {
        self.tensor.rank()
    }

    fn device(&self) -> crate::device::Device {
        self.tensor.device()
    }

    fn to_device(&self, device: crate::device::Device) -> Box<dyn Tensor<S, Element = T>> {
        Box::new(ScalarMulNode {
            tensor: self.tensor.to_device(device),
            scalar: self.scalar.clone(),
        })
    }
}

/// Empty node for placeholder purposes
struct EmptyNode<S: Shape, T: NumericType>(std::marker::PhantomData<(S, T)>);

impl<S: Shape, T: NumericType> Tensor<S> for EmptyNode<S, T> {
    type Element = T;

    fn shape(&self) -> S::Dims {
        unimplemented!("EmptyNode should not be used for actual operations")
    }

    fn reshape(&self, _shape: &[isize]) -> Box<dyn Tensor<S, Element = T>> {
        unimplemented!("EmptyNode should not be used for actual operations")
    }

    fn shape_of(&self, _index: isize) -> usize {
        unimplemented!("EmptyNode should not be used for actual operations")
    }

    fn stride(&self, _index: isize) -> usize {
        unimplemented!("EmptyNode should not be used for actual operations")
    }

    fn rank(&self) -> usize {
        unimplemented!("EmptyNode should not be used for actual operations")
    }

    fn device(&self) -> crate::device::Device {
        unimplemented!("EmptyNode should not be used for actual operations")
    }

    fn to_device(&self, _device: crate::device::Device) -> Box<dyn Tensor<S, Element = T>> {
        unimplemented!("EmptyNode should not be used for actual operations")
    }
}
