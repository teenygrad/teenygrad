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

        dependencies[0].borrow_mut().eval();
        dependencies[0].borrow().data.as_ref().unwrap().ln()
    }

    fn backward(&self, dependencies: &[ValueRef], grad: &TensorData) {
        assert_eq!(dependencies.len(), 1);
        let mut a = dependencies[0].borrow_mut();

        let grad_a = grad / a.data.clone().unwrap();
        a.accumulate_grad(&grad_a);
    }
}
