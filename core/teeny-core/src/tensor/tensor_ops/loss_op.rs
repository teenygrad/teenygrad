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

use std::{cell::RefCell, rc::Rc};

use ndarray::array;

use crate::tensor::{Tensor, TensorData, Value, ValueRef, tensor_ops::TensorOp};

#[derive(Debug, Clone)]
pub struct LossOp;

impl TensorOp for LossOp {
    fn eval(&self, dependencies: &[ValueRef]) -> TensorData {
        assert_eq!(dependencies.len(), 1);

        dependencies[0].borrow_mut().eval();
        dependencies[0].borrow().data.clone().unwrap()
    }

    fn backward(&self, dependencies: &[ValueRef], _grad: &TensorData) {
        let grad = array![1.0].into_dyn();

        dependencies
            .iter()
            .for_each(|v| v.borrow_mut().accumulate_grad(&grad));
    }
}

impl LossOp {
    pub fn wrap(x: Tensor) -> Tensor {
        let requires_grad = x.value.borrow().requires_grad;

        let value = Rc::new(RefCell::new(Value::new(
            rand::random::<f32>() as usize,
            None,
            Box::new(LossOp),
            vec![x.value.clone()],
            requires_grad,
        )));

        Tensor {
            value,
            shape: x.shape.clone(),
        }
    }
}

#[cfg(test)]
mod tests {
    use ndarray::array;

    use super::*;

    #[test]
    fn test_loss_op() {
        let a: Tensor = array![[1.0, 2.0], [3.0, 4.0]].into();
        let b: Tensor = array![[5.0, 6.0], [7.0, 8.0]].into();

        let c = &a * &b;
        let d = &c * &a + &b;
        let e = LossOp::wrap(d);

        e.eval();
        println!("E: {:?}", e);

        e.backward();

        // assert_eq!(a.gradients(), vec![TensorData::ones(vec![2, 2])]);
        // assert_eq!(b.gradients(), vec![TensorData::ones(vec![2, 2])]);
    }
}
