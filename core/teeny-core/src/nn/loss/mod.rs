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

use crate::{dtype::Dtype, graph::NodeRef};

pub mod bce_loss;

#[derive(Debug, Clone)]
pub struct Loss<'data, N: Dtype> {
    pub params: Vec<NodeRef<'data, N>>,
    pub loss: NodeRef<'data, N>,
}

impl<'data, N: Dtype> Loss<'data, N> {
    pub fn new(loss: NodeRef<'data, N>) -> Self {
        Self {
            params: vec![],
            loss,
        }
    }
}

pub trait LossFn<'data, N: Dtype>: Sized {
    fn compute(&self, p: NodeRef<'data, N>, y: NodeRef<'data, N>) -> Loss<'data, N>;
}

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
