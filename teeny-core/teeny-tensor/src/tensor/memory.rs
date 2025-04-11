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

use alloc::vec::Vec;

use crate::tensor::{ElementType, Tensor};

pub struct MemoryTensor<T> {
    pub element_type: ElementType,
    pub shape: Vec<i64>,
    pub data: Vec<T>,
}

impl<T> Tensor<T> for MemoryTensor<T> {
    fn element_type(&self) -> &ElementType {
        &self.element_type
    }

    fn shape(&self) -> &[i64] {
        &self.shape
    }

    fn data(&self) -> &[T] {
        &self.data
    }

    fn reshape(&mut self, shape: Vec<i64>) -> &mut Self {
        self.shape = shape;
        self
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
