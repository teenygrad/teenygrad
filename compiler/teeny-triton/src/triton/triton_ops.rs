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

// use std::{fmt::Display, ops::Div};

// use teeny_core::{
//     tensorx::types::NumericType,
//     tensorx::{Shape, Tensor, shape::DynamicShape},
// };

// pub struct ConstExpr<T: Display>(pub T);

// pub enum Triton<S: Shape, T: NumericType> {
//     Arange {
//         start: T,
//         end: T,
//         step: T,
//         _shape: std::marker::PhantomData<S>,
//     },
// }

// pub fn program_id(_id: usize) -> usize {
//     unimplemented!("Only used for type checking")
// }

// pub struct Block<'a, T: NumericType> {
//     pub base: &'a Tensor,
//     pub shape: DynamicShape,
//     pub strides: DynamicShape,
//     pub offsets: DynamicShape,
//     pub block_shape: DynamicShape,
//     pub order: DynamicShape,
//     pub _marker: std::marker::PhantomData<T>,
// }

// pub fn arange<T: NumericType>(_start: T, _end: T, _step: T) -> Tensor {
//     unimplemented!("Only used for type checking")
// }

// pub fn zeros<S: Shape, T: NumericType>(_shape: S) -> Tensor {
//     unimplemented!("Only used for type checking")
// }

// pub fn append_axis<T: NumericType>(_tensor: &Tensor) -> Tensor {
//     unimplemented!("Only used for type checking")
// }

// pub fn prepend_axis<T: NumericType>(_tensor: &Tensor) -> Tensor {
//     unimplemented!("Only used for type checking")
// }

// pub fn load<T: NumericType>(_offsets: &Tensor) -> Tensor {
//     unimplemented!("Only used for type checking")
// }

// pub fn floor_div<T: Div<Output = U>, U>(_a: T, _b: T) -> U {
//     unimplemented!("Only used for type checking")
// }

// pub fn make_block_ptr<T: NumericType>(_block: Block<T>) -> Tensor {
//     unimplemented!("Only used for type checking")
// }

// pub fn dot<T: NumericType>(_a: &Tensor, _b: &Tensor) -> Tensor {
//     unimplemented!("Only used for type checking")
// }

// pub fn maximum<T: NumericType>(_a: &Tensor, _b: &Tensor) -> Tensor {
//     unimplemented!("Only used for type checking")
// }

// pub fn sub<T: NumericType>(_a: &Tensor, _b: &Tensor) -> Tensor {
//     unimplemented!("Only used for type checking")
// }

// pub fn max<T: NumericType>(_a: &Tensor, _b: f32) -> Tensor {
//     unimplemented!("Only used for type checking")
// }

// pub fn exp2<T: NumericType>(_a: &Tensor) -> Tensor {
//     unimplemented!("Only used for type checking")
// }

// pub fn log2<T: NumericType>(_a: &Tensor) -> Tensor {
//     unimplemented!("Only used for type checking")
// }

// pub fn sum<T: NumericType>(_a: &Tensor, _axis: usize) -> Tensor {
//     unimplemented!("Only used for type checking")
// }

// pub fn advance<T: NumericType>(_tensor: &Tensor, _offsets: DynamicShape) -> Tensor {
//     unimplemented!("Only used for type checking")
// }

// pub fn store<T: NumericType>(_tensor: &Tensor, _offsets: &Tensor) {
//     unimplemented!("Only used for type checking")
// }
