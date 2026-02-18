/*
 * Copyright (c) 2026 Teenygrad.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

use crate::graph::NodeRef;

pub mod bce_loss;

#[derive(Debug, Clone)]
pub struct Loss<'data> {
    pub params: Vec<NodeRef<'data>>,
    pub loss: NodeRef<'data>,
}

impl<'data> Loss<'data> {
    pub fn new(loss: NodeRef<'data>) -> Self {
        Self {
            params: vec![],
            loss,
        }
    }
}

pub trait LossFn<'data>: Sized {
    fn compute(&self, p: NodeRef<'data>, y: NodeRef<'data>) -> Loss<'data>;
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
