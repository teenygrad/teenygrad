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
pub struct MeanOp;

impl TensorOp for MeanOp {
    fn backward(&self, dependencies: &[ValueRef], grad: f32) {
        let n = dependencies.len() as f32;
        let grad_per_element = grad / n;
        for dep in dependencies {
            if dep.borrow().requires_grad {
                dep.borrow_mut().accumulate_grad(grad_per_element);
            }
        }
    }
}

impl Tensor {
    pub fn mean(&self) -> Tensor {
        let mut result_values = Vec::with_capacity(1);

        let result_value = Value::new(
            rand::random::<f32>() as usize,
            None,
            Box::new(MeanOp),
            self.values.clone(),
            self.values.iter().any(|v| v.borrow().requires_grad),
        );

        result_values.push(Rc::new(RefCell::new(result_value)));

        Tensor {
            values: result_values,
            shape: vec![1],
        }
    }
}
