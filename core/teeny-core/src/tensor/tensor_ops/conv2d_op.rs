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

use crate::tensor::{ValueRef, tensor_ops::TensorOp};

#[derive(Debug, Clone)]
pub struct Conv2dOp;

impl TensorOp for Conv2dOp {
    fn backward(&self, dependencies: &[ValueRef], grad: f32) {
        if !dependencies.is_empty() && dependencies[0].borrow().requires_grad {
            dependencies[0].borrow_mut().accumulate_grad(grad);
        }
    }
}
