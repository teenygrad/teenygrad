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

use crate::tensor::{TensorData, ValueRef, tensor_ops::TensorOp};

#[derive(Debug, Clone)]
pub struct LogOp;

impl TensorOp for LogOp {
    fn eval(&self, dependencies: &[ValueRef]) -> TensorData {
        assert_eq!(dependencies.len(), 1);
        dependencies[0].borrow().data.as_ref().unwrap().log(10.0)
    }

    fn backward(&self, dependencies: &[ValueRef], _grad: &TensorData) {
        if !dependencies.is_empty() && dependencies[0].borrow().requires_grad {
            let _input_val = dependencies[0].borrow().data.as_ref().unwrap();
            todo!("Fixme")
            // let log_grad = if input_val > 0.0 {
            //     grad / input_val
            // } else {
            //     // array![0.0]
            // };
            // dependencies[0].borrow_mut().accumulate_grad(log_grad);
        }
    }
}
