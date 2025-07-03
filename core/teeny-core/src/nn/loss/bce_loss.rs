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

use ndarray::array;

use crate::{
    nn::loss::{Loss, LossFn},
    tensor::{Tensor, log, tensor_ops::loss_op, value::topsort_graph},
};

#[derive(Debug, Clone, Default)]
pub struct BCELoss {}

impl BCELoss {
    pub fn new() -> Self {
        Self {}
    }
}

impl LossFn for BCELoss {
    fn compute(&self, p: &Tensor, y: &Tensor) -> Loss {
        let bce_loss = -(y * log(p.clone()) + (1.0 - y) * log(1.0 - p.clone()));
        let p1 = loss_op::loss(p.clone());
        p1.value.borrow_mut().grad = Some(array![1.0].into_dyn());
        p1.value.borrow_mut().eval();

        Loss {
            params: topsort_graph(&p1.value),
            loss: bce_loss,
        }
    }
}

#[cfg(test)]
mod tests {
    use ndarray::array;

    use super::*;

    #[test]
    fn test_bce_loss() {
        // Create prediction and target tensors using the correct constructor
        let a: Tensor = array![0.5, 0.5].into();
        let b: Tensor = array![1.0, 0.0].into();

        let c = &a * &b;
        let d = &c * &a + &b;
        let t: Tensor = array![0.5, 0.5].into();

        let bce = BCELoss::new();
        let mut loss = bce.compute(&d, &t);

        loss.backward();

        println!("Loss: {:?}", loss.loss);
        println!("Params: {:?}", loss.params);

        todo!()
    }
}
