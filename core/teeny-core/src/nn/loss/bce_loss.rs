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
