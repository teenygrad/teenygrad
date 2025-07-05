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

use crate::tensor::{Tensor, TensorData, Value, ValueRef, tensor_ops::TensorOp};

#[derive(Debug, Clone)]
pub struct SigmoidOp;

impl TensorOp for SigmoidOp {
    fn eval(&self, dependencies: &[ValueRef]) -> TensorData {
        assert_eq!(dependencies.len(), 1);

        // Ensure the input is evaluated
        dependencies[0].borrow_mut().eval();

        dependencies[0]
            .borrow()
            .data
            .as_ref()
            .unwrap()
            .map(|v| 1.0 / (1.0 + (-v).exp()))
    }

    fn backward(&self, dependencies: &[ValueRef], grad: &TensorData) {
        assert_eq!(dependencies.len(), 1);
        let mut input = dependencies[0].borrow_mut();

        let input_data = input.data.as_ref().unwrap();
        let sigmoid_grad = grad
            * input_data.map(|v| {
                let sigmoid = 1.0 / (1.0 + (-v).exp());
                sigmoid * (1.0 - sigmoid)
            });
        input.accumulate_grad(&sigmoid_grad);
    }
}

impl Tensor {
    pub fn sigmoid(&self) -> Tensor {
        let requires_grad = self.value.borrow().requires_grad;

        let value = Rc::new(RefCell::new(Value::new(
            None,
            Box::new(SigmoidOp),
            vec![self.value.clone()],
            requires_grad,
        )));

        Tensor { value }
    }
}

pub fn sigmoid(x: &Tensor) -> Tensor {
    x.sigmoid()
}

#[cfg(test)]
mod tests {
    use ndarray::array;

    use super::*;
    use crate::{nn::loss::Loss, tensor::Tensor};

    #[test]
    fn test_sigmoid_backprop() {
        // Test with a simple 2x2 tensor
        let a: Tensor = array![[1.0, 2.0], [-1.0, 0.0]].into();
        let b: Tensor = array![[0.5, -0.5], [1.0, -1.0]].into();

        let c = &a * &b;
        let d = sigmoid(&c);

        let mut loss = Loss::new(d.clone());
        loss.backward();

        assert_eq!(
            c.value.borrow().data,
            Some(array![[0.5, -1.0], [-1.0, 0.0]].into_dyn())
        );

        assert_eq!(
            format!("{:?}", d.value.borrow().data),
            "Some([[0.62245935, 0.26894143],\n [0.26894143, 0.5]], shape=[2, 2], strides=[2, 1], layout=Cc (0x5), dynamic ndim=2)"
        );

        assert_eq!(
            format!("{:?}", a.grad()),
            "Some([[0.117501855, -0.09830597],\n [0.19661194, -0.25]], shape=[2, 2], strides=[2, 1], layout=Cc (0x5), dynamic ndim=2)"
        );
        assert_eq!(
            format!("{:?}", b.grad()),
            "Some([[0.23500371, 0.39322388],\n [-0.19661194, 0.0]], shape=[2, 2], strides=[2, 1], layout=Cc (0x5), dynamic ndim=2)"
        );
        assert_eq!(
            format!("{:?}", c.grad()),
            "Some([[0.23500371, 0.19661194],\n [0.19661194, 0.25]], shape=[2, 2], strides=[2, 1], layout=Cc (0x5), dynamic ndim=2)"
        );
    }

    #[test]
    fn test_sigmoid_edge_cases() {
        // Test with very large positive and negative values
        let large_pos: Tensor = array![10.0].into();
        let large_neg: Tensor = array![-10.0].into();
        let zero: Tensor = array![0.0].into();

        let sigmoid_pos = sigmoid(&large_pos);
        let sigmoid_neg = sigmoid(&large_neg);
        let sigmoid_zero = sigmoid(&zero);

        sigmoid_pos.eval();
        sigmoid_neg.eval();
        sigmoid_zero.eval();

        // sigmoid(10) should be very close to 1
        let pos_val = sigmoid_pos.value.borrow().data.as_ref().unwrap()[0];
        assert!(pos_val > 0.999);

        // sigmoid(-10) should be very close to 0
        let neg_val = sigmoid_neg.value.borrow().data.as_ref().unwrap()[0];
        assert!(neg_val < 0.001);

        // sigmoid(0) should be exactly 0.5
        let zero_val = sigmoid_zero.value.borrow().data.as_ref().unwrap()[0];
        assert!((zero_val - 0.5).abs() < 1e-6);
    }
}
