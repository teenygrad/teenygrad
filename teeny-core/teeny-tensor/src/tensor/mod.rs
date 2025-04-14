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

pub mod memory;

#[derive(Clone)]
pub enum ElementType {
    FP16,
}

pub type Shape = Vec<i64>;
pub trait Tensor<T: Clone> {
    fn element_type(&self) -> &ElementType;
    fn shape(&self) -> &[i64];
    fn data(&self) -> &[T];

    fn reshape(&mut self, shape: Shape) -> Box<dyn Tensor<T>>;

    fn dot(&self, other: &dyn Tensor<T>) -> Box<dyn Tensor<T>>;
    fn relu(&self) -> Box<dyn Tensor<T>>;
    fn log_softmax(&self) -> Box<dyn Tensor<T>>;

    fn zeroes(&self, shape: Shape) -> Box<dyn Tensor<T>>;
    fn ones(&self, shape: Shape) -> Box<dyn Tensor<T>>;
    fn empty(&self, shape: Shape) -> Box<dyn Tensor<T>>;
    fn full(&self, shape: Shape, value: T) -> Box<dyn Tensor<T>>;
    fn rand(&self, shape: Shape) -> Box<dyn Tensor<T>>;
    fn randn(&self, shape: Shape) -> Box<dyn Tensor<T>>;
    fn arange(&self, start: T, end: T, step: T) -> Box<dyn Tensor<T>>;
    fn linspace(&self, start: T, end: T, steps: i64) -> Box<dyn Tensor<T>>;
    fn eye(&self, shape: Shape) -> Box<dyn Tensor<T>>;
    fn diag(&self, shape: Shape) -> Box<dyn Tensor<T>>;
}
