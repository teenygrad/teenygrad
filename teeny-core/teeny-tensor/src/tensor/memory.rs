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

use crate::tensor::Tensor;

#[derive(Clone)]
pub struct MemoryTensor<T> {
    pub shape: Vec<i64>,
    pub data: Vec<T>,
}

impl<T: Clone + 'static> Tensor<T> for MemoryTensor<T> {
    fn shape(&self) -> &[i64] {
        &self.shape
    }

    fn reshape(&mut self, _shape: &[i64]) -> Box<dyn Tensor<T>> {
        todo!()
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
}

impl<T> MemoryTensor<T> {
    pub fn new() -> Self {
        Self {
            shape: Vec::new(),
            data: Vec::new(),
        }
    }

    pub fn with_data(shape: &[i64], data: Vec<T>) -> Self {
        Self {
            shape: shape.to_vec(),
            data,
        }
    }

    pub fn randn(_shape: &[i64]) -> Self {
        todo!()
    }
}

impl<T> Default for MemoryTensor<T> {
    fn default() -> Self {
        Self::new()
    }
}
