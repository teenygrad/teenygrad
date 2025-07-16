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

pub mod add;
pub mod arange;
pub mod div;
pub mod exp;
pub mod log;
pub mod mean;
pub mod mult;
pub mod neg;
pub mod randn;
pub mod relu;
pub mod scalar;
pub mod sigmoid;
pub mod sub;
pub mod zeros;

// pub mod add_op;
// pub mod bias_op;
// pub mod input_op;
// pub mod log_op;
// pub mod mean_op;
// pub mod mult_op;
// pub mod param_op;
// pub mod relu_op;
// pub mod sigmoid_op;
// pub mod sub_op;
// pub mod transpose_op;

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
