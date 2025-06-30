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

use crate::tensor::{Tensor, Value, ValueRef, tensor_ops::TensorOp};

#[derive(Debug, Clone)]
pub struct MultOp;

impl TensorOp for MultOp {
    fn backward(&self, dependencies: &[ValueRef], grad: f32) {
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

        let rows = self.shape[0];
        let cols = other.shape[1];
        let mut result_values = Vec::with_capacity(rows * cols);

        for i in 0..rows {
            for j in 0..cols {
                let mut dependencies = Vec::new();

                for k in 0..self.shape[1] {
                    let a_idx = i * self.shape[1] + k;
                    let b_idx = k * other.shape[1] + j;

                    dependencies.push(self.values[a_idx].clone());
                    dependencies.push(other.values[b_idx].clone());
                }

                let result_value = Value::new(
                    rand::random::<f32>() as usize,
                    None,
                    Box::new(MultOp),
                    dependencies,
                    self.values.iter().any(|v| v.borrow().requires_grad)
                        || other.values.iter().any(|v| v.borrow().requires_grad),
                );

                result_values.push(Rc::new(RefCell::new(result_value)));
            }
        }

        Tensor {
            values: result_values,
            shape: vec![rows, cols],
        }
    }
}
