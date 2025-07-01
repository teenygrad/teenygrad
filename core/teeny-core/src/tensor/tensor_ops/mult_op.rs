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
pub struct MultOp;

impl TensorOp for MultOp {
    fn eval(&self, dependencies: &[ValueRef]) -> TensorData {
        assert_eq!(dependencies.len(), 2);
        dependencies[0].borrow().data.as_ref().unwrap()
            * dependencies[1].borrow().data.as_ref().unwrap()
    }

    fn backward(&self, dependencies: &[ValueRef], grad: &TensorData) {
        if dependencies.len() >= 2 {
            if dependencies[0].borrow().requires_grad {
                dependencies[0].borrow_mut().accumulate_grad(grad);
            }
            if dependencies[1].borrow().requires_grad {
                dependencies[1].borrow_mut().accumulate_grad(grad);
            }
        }
    }
}

impl Tensor {
    pub fn mult(&self, other: &Tensor) -> Tensor {
        // Matrix multiplication that returns a new tensor with graph nodes
        // For simplicity, we'll assume 2D matrices
        assert_eq!(self.shape.len(), 2, "matmul requires 2D tensors");
        assert_eq!(other.shape.len(), 2, "matmul requires 2D tensors");
        assert_eq!(self.shape[1], other.shape[0], "matmul dimension mismatch");

        let requires_grad = self.value.borrow().requires_grad || other.value.borrow().requires_grad;

        let value = Rc::new(RefCell::new(Value::new(
            rand::random::<f32>() as usize,
            None,
            Box::new(MultOp),
            vec![self.value.clone(), other.value.clone()],
            requires_grad,
        )));

        Tensor {
            value,
            shape: self.shape.clone(),
        }
    }
}
