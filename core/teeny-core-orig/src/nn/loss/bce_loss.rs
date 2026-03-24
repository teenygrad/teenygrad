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

use crate::{
    graph::{NodeRef, log},
    nn::loss::{Loss, LossFn},
};

#[derive(Debug, Clone, Default)]
pub struct BCELoss {}

impl BCELoss {
    pub fn new() -> Self {
        Self {}
    }
}

impl<'data> LossFn<'data> for BCELoss {
    fn compute(&self, p: NodeRef<'data>, y: NodeRef<'data>) -> Loss<'data> {
        let one: NodeRef<'data> = 1.0.into();
        let bce_loss = -(&y * log(p.clone()) + (&one - y) * log(&one - p));

        Loss::new(bce_loss)
    }
}

// #[cfg(test)]
// mod tests {
//     use ndarray::array;

//     use super::*;

//     #[test]
//     fn test_bce_loss() {
//         // Create prediction and target tensors using the correct constructor
//         let a: Tensor = array![[0.5, 0.5], [0.5, 0.5]].into();
//         let b: Tensor = array![[1.0, 2.0], [2.0, 3.0]].into();

//         let c = &a * &b;
//         let d = &c * &a + &b;
//         let t: Tensor = array![[0.5, 0.5], [0.5, 0.5]].into();

//         d.eval();

//         let bce = BCELoss::new();
//         let mut loss = bce.compute(&d, &t);

//         loss.backward();

//         assert_eq!(
//             format!("{:?}", loss.loss.value.borrow().data.as_ref().unwrap()),
//             "[[-0.3465736, -0.25541282],\n [-0.3465736, -0.25541282]], shape=[2, 2], strides=[2, 1], layout=Cc (0x5), dynamic ndim=2"
//         );
//     }
// }
