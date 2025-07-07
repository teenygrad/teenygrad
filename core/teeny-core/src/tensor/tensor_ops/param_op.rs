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
pub struct ParamOp;

impl TensorOp for ParamOp {
    fn is_param(&self) -> bool {
        true
    }

    fn eval(&self, _dependencies: &[ValueRef]) -> TensorData {
        unreachable!("ParamOp should never be evaluated")
    }

    fn backward(&self, _dependencies: &[ValueRef], _grad: &TensorData) {
        // do nothing
    }
}

impl Tensor {
    pub fn new_param(data: TensorData) -> Self {
        let value = Rc::new(RefCell::new(Value::new(
            Some(data),
            Box::new(ParamOp),
            Vec::new(),
            true,
        )));

        Tensor { value }
    }
}
