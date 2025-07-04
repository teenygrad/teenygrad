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
        dependencies[0]
            .borrow()
            .data
            .as_ref()
            .unwrap()
            .map(|v| 1.0 / (1.0 + (-v).exp()))
    }

    fn backward(&self, dependencies: &[ValueRef], grad: &TensorData) {
        if !dependencies.is_empty() && dependencies[0].borrow().requires_grad {
            let input = dependencies[0].borrow();
            let input_val = input.data.as_ref().unwrap();
            let sigmoid_val = 1.0 / (1.0 + (-input_val).exp());
            let sigmoid_grad = grad * sigmoid_val.clone() * (1.0 - sigmoid_val);
            dependencies[0].borrow_mut().accumulate_grad(&sigmoid_grad);
        }
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

        Tensor {
            value,
            shape: self.shape.clone(),
        }
    }
}
