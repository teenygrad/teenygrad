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
pub struct AddOp;

impl TensorOp for AddOp {
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
    pub fn add(&self, other: &Tensor) -> Tensor {
        assert_eq!(self.shape, other.shape, "Shape mismatch in addition");

        let mut result_values = Vec::with_capacity(self.values.len());

        for (a, b) in self.values.iter().zip(other.values.iter()) {
            let result_value = Value::new(
                rand::random::<f32>() as usize,
                None,
                Box::new(AddOp),
                vec![a.clone(), b.clone()],
                a.borrow().requires_grad || b.borrow().requires_grad,
            );

            result_values.push(Rc::new(RefCell::new(result_value)));
        }

        Tensor {
            values: result_values,
            shape: self.shape.clone(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_add_op() {
        let a = Tensor::new(vec![2, 2], true);
        let b = Tensor::new(vec![2, 2], true);

        let c = a.add(&b);

        assert_eq!(c.shape, vec![2, 2]);
    }
}
