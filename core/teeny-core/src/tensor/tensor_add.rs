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

// Tensor + Tensor operations using references
impl<'a, 'b, S: Shape, T: NumericType> Add<&'b Box<dyn Tensor<S, Element = T> + 'static>>
    for &'a Box<dyn Tensor<S, Element = T> + 'static>
{
    type Output = Box<dyn Tensor<S, Element = T> + 'static>;

    fn add(self, rhs: &'b Box<dyn Tensor<S, Element = T> + 'static>) -> Self::Output {
        // Create a computation graph node for addition
        // The actual computation would be deferred until evaluation
        Box::new(AddNode {
            lhs: self.as_ref(),
            rhs: rhs.as_ref(),
        })
    }
}

// Tensor + Scalar operations using references
impl<'a, S: Shape, T: NumericType> Add<T::RustType>
    for &'a Box<dyn Tensor<S, Element = T> + 'static>
{
    type Output = Box<dyn Tensor<S, Element = T> + 'static>;

    fn add(self, rhs: T::RustType) -> Self::Output {
        // Create a computation graph node for scalar addition
        Box::new(ScalarAddNode {
            tensor: self.as_ref(),
            scalar: rhs,
        })
    }
}

// Scalar + Tensor operations using references
impl<'a, S: Shape, T: NumericType> Add<&'a Box<dyn Tensor<S, Element = T> + 'static>>
    for T::RustType
{
    type Output = Box<dyn Tensor<S, Element = T> + 'static>;

    fn add(self, rhs: &'a Box<dyn Tensor<S, Element = T> + 'static>) -> Self::Output {
        // Create a computation graph node for scalar addition (commutative)
        Box::new(ScalarAddNode {
            tensor: rhs.as_ref(),
            scalar: self,
        })
    }
}

// AddAssign implementations for in-place operations
impl<S: Shape, T: NumericType> AddAssign<&Box<dyn Tensor<S, Element = T> + 'static>>
    for Box<dyn Tensor<S, Element = T> + 'static>
{
    fn add_assign(&mut self, rhs: &Box<dyn Tensor<S, Element = T> + 'static>) {
        // For in-place operations, we might need to evaluate immediately
        // or create a special in-place computation node
        *self = Box::new(AddNode {
            lhs: self.as_ref(),
            rhs: rhs.as_ref(),
        });
    }
}

impl<S: Shape, T: NumericType> AddAssign<T::RustType>
    for Box<dyn Tensor<S, Element = T> + 'static>
{
    fn add_assign(&mut self, rhs: T::RustType) {
        *self = Box::new(ScalarAddNode {
            tensor: self.as_ref(),
            scalar: rhs,
        });
    }
}

// Computation graph nodes for lazy evaluation

/// Node representing tensor addition
struct AddNode<'a, S: Shape, T: NumericType> {
    lhs: &'a dyn Tensor<S, Element = T>,
    rhs: &'a dyn Tensor<S, Element = T>,
}

impl<'a, S: Shape, T: NumericType> Tensor<S> for AddNode<'a, S, T> {
    type Element = T;

    fn shape(&self) -> S::Dims {
        // Both tensors should have the same shape for addition
        self.lhs.shape()
    }

    fn reshape(&self, shape: &[isize]) -> Box<dyn Tensor<S, Element = T> + 'static> {
        // Reshape both operands and create a new addition node
        Box::new(AddNode {
            lhs: &*self.lhs.reshape(shape),
            rhs: &*self.rhs.reshape(shape),
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

    fn to_device(
        &self,
        device: crate::device::Device,
    ) -> Box<dyn Tensor<S, Element = T> + 'static> {
        Box::new(AddNode {
            lhs: &*self.lhs.to_device(device.clone()),
            rhs: &*self.rhs.to_device(device),
        })
    }
}

/// Node representing scalar addition
struct ScalarAddNode<'a, S: Shape, T: NumericType> {
    tensor: &'a dyn Tensor<S, Element = T>,
    scalar: T::RustType,
}

impl<'a, S: Shape, T: NumericType> Tensor<S> for ScalarAddNode<'a, S, T> {
    type Element = T;

    fn shape(&self) -> S::Dims {
        self.tensor.shape()
    }

    fn reshape(&self, shape: &[isize]) -> Box<dyn Tensor<S, Element = T> + 'static> {
        Box::new(ScalarAddNode {
            tensor: &*self.tensor.reshape(shape),
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

    fn to_device(
        &self,
        device: crate::device::Device,
    ) -> Box<dyn Tensor<S, Element = T> + 'static> {
        Box::new(ScalarAddNode {
            tensor: &*self.tensor.to_device(device),
            scalar: self.scalar.clone(),
        })
    }
}
