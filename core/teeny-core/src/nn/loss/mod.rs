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

// use crate::tensorx::{
//     Tensor,
//     value::{ValueRef, toposort_graph},
// };

pub mod bce_loss;

// pub trait LossFn {
//     fn compute(&self, _y_pred: &Tensor, _y_target: &Tensor) -> Loss;
// }

// #[derive(Debug, Clone)]
// pub struct Loss {
//     pub params: Vec<ValueRef>,
//     pub loss: Tensor,
// }

// impl Loss {
//     pub fn new(loss: Tensor) -> Self {
//         loss.eval();

//         // Get the shape of the loss tensor and create a gradient of ones with the same shape
//         let loss_shape: Vec<usize> = {
//             let loss_ref = loss.value.borrow();
//             loss_ref.data.as_ref().unwrap().shape().to_vec()
//         };
//         let grad_data = ndarray::Array::ones(loss_shape).into_dyn();
//         loss.value.borrow_mut().grad = Some(grad_data);

//         let params = toposort_graph(&loss.value);

//         Self { params, loss }
//     }
// }

// impl Loss {
//     pub fn backward(&mut self) {
//         for param in self.params.iter().rev() {
//             param.borrow_mut().backward();
//         }
//     }
// }
