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

use std::fmt::Display;

use crate::{
    tensor::{DenseTensor, DynamicShape, Shape},
    types::NumericType,
};

pub mod kernel;
pub use kernel::TritonKernel;

pub struct ConstExpr<T: Display>(pub T);

pub enum Triton<S: Shape, T: NumericType> {
    Arange {
        start: T,
        end: T,
        step: T,
        _shape: std::marker::PhantomData<S>,
    },
}

pub fn program_id(_id: i32) -> i32 {
    unimplemented!("Only used for type checking")
}

pub struct Block<S: Shape, T: NumericType> {
    pub base: DenseTensor<DynamicShape, T>,
    pub shape: S,
    pub strides: S,
    pub offsets: S,
    pub block_shape: S,
    pub order: S,
}

pub fn arange<S: Shape, T: NumericType>(
    _start: T,
    _end: T,
    _step: T,
) -> DenseTensor<DynamicShape, T> {
    unimplemented!("Only used for type checking")
}

pub fn zeros<S: Shape, T: NumericType>(_shape: S) -> DenseTensor<DynamicShape, T> {
    unimplemented!("Only used for type checking")
}

pub fn load<S: Shape, T: NumericType>(_offsets: S) -> DenseTensor<DynamicShape, T> {
    unimplemented!("Only used for type checking")
}
