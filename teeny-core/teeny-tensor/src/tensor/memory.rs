/*
 * Copyright (C) 2025 SpinorML Ltd.
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

use alloc::{boxed::Box, vec::Vec};

use crate::tensor::{ElementType, Tensor};

#[derive(Clone)]
pub struct MemoryTensor<T> {
    pub element_type: ElementType,
    pub shape: Vec<i64>,
    pub data: Vec<T>,
}

impl<T: Clone + 'static> Tensor<T> for MemoryTensor<T> {
    fn element_type(&self) -> &ElementType {
        &self.element_type
    }

    fn shape(&self) -> &[i64] {
        &self.shape
    }

    fn data(&self) -> &[T] {
        &self.data
    }

    fn reshape(&mut self, shape: Vec<i64>) -> Box<dyn Tensor<T>> {
        Box::new(MemoryTensor::new(
            self.element_type.clone(),
            shape,
            self.data.clone(),
        ))
    }

    fn dot(&self, _other: &dyn Tensor<T>) -> Box<dyn Tensor<T>> {
        todo!()
    }

    fn relu(&self) -> Box<dyn Tensor<T>> {
        todo!()
    }

    fn log_softmax(&self) -> Box<dyn Tensor<T>> {
        todo!()
    }

    fn zeroes(&self, _shape: super::Shape) -> Box<dyn Tensor<T>> {
        todo!()
    }

    fn ones(&self, _shape: super::Shape) -> Box<dyn Tensor<T>> {
        todo!()
    }

    fn empty(&self, _shape: super::Shape) -> Box<dyn Tensor<T>> {
        todo!()
    }

    fn full(&self, _shape: super::Shape, _value: T) -> Box<dyn Tensor<T>> {
        todo!()
    }

    fn rand(&self, _shape: super::Shape) -> Box<dyn Tensor<T>> {
        todo!()
    }

    fn randn(&self, _shape: super::Shape) -> Box<dyn Tensor<T>> {
        todo!()
    }

    fn arange(&self, _start: T, _end: T, _step: T) -> Box<dyn Tensor<T>> {
        todo!()
    }

    fn linspace(&self, _start: T, _end: T, _steps: i64) -> Box<dyn Tensor<T>> {
        todo!()
    }

    fn eye(&self, _shape: super::Shape) -> Box<dyn Tensor<T>> {
        todo!()
    }

    fn diag(&self, _shape: super::Shape) -> Box<dyn Tensor<T>> {
        todo!()
    }
}

impl<T> MemoryTensor<T> {
    pub fn new(element_type: ElementType, shape: Vec<i64>, data: Vec<T>) -> Self {
        Self {
            element_type,
            shape,
            data,
        }
    }
}

impl<T> Default for MemoryTensor<T> {
    fn default() -> Self {
        Self::new(ElementType::FP16, Vec::new(), Vec::new())
    }
}
