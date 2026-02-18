/*
 * Copyright (c) 2026 Teenygrad.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

use std::{cell::RefCell, rc::Rc};

use crate::tensorx::{Tensor, TensorData, Value, ValueRef, tensor_ops::TensorOp};

#[derive(Debug, Clone)]
pub struct InputOp;

impl TensorOp for InputOp {
    fn is_param(&self) -> bool {
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
