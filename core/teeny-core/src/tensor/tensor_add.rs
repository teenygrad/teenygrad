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

use std::ops::{Add, AddAssign};

use crate::{
    tensor::{Shape, Tensor},
    types::NumericType,
};

// Tensor + Tensor operations (trait objects)
impl<S: Shape, T: NumericType> Add<Box<dyn Tensor<S, Element = T>>>
    for Box<dyn Tensor<S, Element = T>>
{
    type Output = Box<dyn Tensor<S, Element = T>>;

    fn add(self, rhs: Box<dyn Tensor<S, Element = T>>) -> Self::Output {
        // Create a computation graph node for addition
        // The actual computation would be deferred until evaluation
        Box::new(AddNode { lhs: self, rhs })
    }
}

// Tensor + Scalar operations
impl<S: Shape, T: NumericType> Add<T::RustType> for Box<dyn Tensor<S, Element = T>> {
    type Output = Box<dyn Tensor<S, Element = T>>;

    fn add(self, rhs: T::RustType) -> Self::Output {
        // Create a computation graph node for scalar addition
        Box::new(ScalarAddNode {
            tensor: self,
            scalar: rhs,
        })
    }
}

// Scalar + Tensor operations
impl<S: Shape, T: NumericType> Add<Box<dyn Tensor<S, Element = T>>> for T::RustType {
    type Output = Box<dyn Tensor<S, Element = T>>;

    fn add(self, rhs: Box<dyn Tensor<S, Element = T>>) -> Self::Output {
        // Create a computation graph node for scalar addition (commutative)
        Box::new(ScalarAddNode {
            tensor: rhs,
            scalar: self,
        })
    }
}

// AddAssign implementations for in-place operations
impl<S: Shape, T: NumericType> AddAssign<Box<dyn Tensor<S, Element = T>>>
    for Box<dyn Tensor<S, Element = T>>
{
    fn add_assign(&mut self, rhs: Box<dyn Tensor<S, Element = T>>) {
        // For in-place operations, we might need to evaluate immediately
        // or create a special in-place computation node
        *self = Box::new(AddNode {
            lhs: std::mem::replace(self, Box::new(EmptyNode(std::marker::PhantomData))),
            rhs,
        });
    }
}

impl<S: Shape, T: NumericType> AddAssign<T::RustType> for Box<dyn Tensor<S, Element = T>> {
    fn add_assign(&mut self, rhs: T::RustType) {
        *self = Box::new(ScalarAddNode {
            tensor: std::mem::replace(self, Box::new(EmptyNode(std::marker::PhantomData))),
            scalar: rhs,
        });
    }
}

// Computation graph nodes for lazy evaluation

/// Node representing tensor addition
struct AddNode<S: Shape, T: NumericType> {
    lhs: Box<dyn Tensor<S, Element = T>>,
    rhs: Box<dyn Tensor<S, Element = T>>,
}

impl<S: Shape, T: NumericType> Tensor<S> for AddNode<S, T> {
    type Element = T;

    fn shape(&self) -> S::Dims {
        // Both tensors should have the same shape for addition
        self.lhs.shape()
    }

    fn reshape(&self, shape: &[isize]) -> Box<dyn Tensor<S, Element = T>> {
        // Reshape both operands and create a new addition node
        Box::new(AddNode {
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
        // For now, use the device of the left operand
        // In a real implementation, you might want to handle device placement more carefully
        self.lhs.device()
    }

    fn to_device(&self, device: crate::device::Device) -> Box<dyn Tensor<S, Element = T>> {
        Box::new(AddNode {
            lhs: self.lhs.to_device(device),
            rhs: self.rhs.to_device(device),
        })
    }
}

/// Node representing scalar addition
struct ScalarAddNode<S: Shape, T: NumericType> {
    tensor: Box<dyn Tensor<S, Element = T>>,
    scalar: T::RustType,
}

impl<S: Shape, T: NumericType> Tensor<S> for ScalarAddNode<S, T> {
    type Element = T;

    fn shape(&self) -> S::Dims {
        self.tensor.shape()
    }

    fn reshape(&self, shape: &[isize]) -> Box<dyn Tensor<S, Element = T>> {
        Box::new(ScalarAddNode {
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
        Box::new(ScalarAddNode {
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
