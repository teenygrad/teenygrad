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

use crate::dtype::DtypeEnum;
use crate::error::Result;
use crate::tensor::shape::DynamicShape;

pub mod add;
pub mod and;
pub mod arange;
pub mod cat;
pub mod cos;
pub mod cumsum;
pub mod diff;
pub mod div;
pub mod dot;
pub mod eq;
pub mod exp;
pub mod expand;
pub mod index;
pub mod inverse;
pub mod isneginf;
pub mod leq;
pub mod log;
pub mod mean;
pub mod mult;
pub mod neg;
pub mod neq;
pub mod ones;
pub mod or;
pub mod pad;
pub mod pow;
pub mod powi;
pub mod randn;
pub mod relu;
pub mod rsqrt;
pub mod safetensor;
pub mod scalar;
pub mod sigmoid;
pub mod sin;
pub mod slice;
pub mod sqrt;
pub mod sub;
pub mod tensor;
pub mod to_dtype;
pub mod transpose;
pub mod unsqueeze;
pub mod vmap;
pub mod r#where;
pub mod zeros;

pub trait Op {
    fn shape(&self) -> Result<DynamicShape>;
    fn dtype(&self) -> DtypeEnum;
}

// use std::fmt::Debug;

// use crate::tensorx::{TensorData, ValueRef};
// pub trait TensorOp: Debug {
//     fn is_param(&self) -> bool {
//         false
//     }

//     fn eval(&self, dependencies: &[ValueRef]) -> TensorData;

//     fn backward(
//         &self,
//         dependencies: &[ValueRef],
//         grad: &ndarray::ArrayBase<ndarray::OwnedRepr<f32>, ndarray::IxDyn>,
//     );
// }
