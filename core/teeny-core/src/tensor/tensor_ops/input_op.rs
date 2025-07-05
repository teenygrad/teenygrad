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
pub struct InputOp;

impl TensorOp for InputOp {
    fn is_input(&self) -> bool {
        true
    }

    fn eval(&self, _dependencies: &[ValueRef]) -> TensorData {
        unreachable!("InputOp should never be evaluated")
    }

    fn backward(&self, dependencies: &[ValueRef], grad: &TensorData) {
        for dep in dependencies {
            if dep.borrow().requires_grad {
                dep.borrow_mut().accumulate_grad(grad);
            }
        }
    }
}

impl Tensor {
    pub fn new(data: TensorData, requires_grad: bool) -> Self {
        let value = Rc::new(RefCell::new(Value::new(
            Some(data),
            Box::new(InputOp),
            Vec::new(),
            requires_grad,
        )));

        Tensor { value }
    }
}
