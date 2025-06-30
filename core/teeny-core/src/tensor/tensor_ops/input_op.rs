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
pub struct InputOp;

impl TensorOp for InputOp {
    fn backward(&self, dependencies: &[ValueRef], grad: f32) {
        for dep in dependencies {
            if dep.borrow().requires_grad {
                dep.borrow_mut().accumulate_grad(grad);
            }
        }
    }
}

impl Tensor {
    pub fn new(shape: Vec<usize>, requires_grad: bool) -> Self {
        let size = shape.iter().product();
        let mut values = Vec::with_capacity(size);

        for _ in 0..size {
            values.push(Rc::new(RefCell::new(Value::new(
                rand::random::<f32>() as usize,
                None,
                Box::new(InputOp),
                Vec::new(),
                requires_grad,
            ))));
        }

        Tensor { values, shape }
    }
}
