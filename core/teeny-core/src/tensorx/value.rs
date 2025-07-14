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

use std::{cell::RefCell, collections::HashSet, rc::Rc};

use uuid::Uuid;

use crate::tensorx::tensor_ops::TensorOp;

pub type TensorData = ndarray::ArrayBase<ndarray::OwnedRepr<f32>, ndarray::IxDyn>;

/// Value represents either an input or the result of an operation
#[derive(Debug)]
pub struct Value {
    pub id: String,
    pub data: Option<TensorData>, // Concrete value if computed
    pub operation: Box<dyn TensorOp>,
    pub dependencies: Vec<ValueRef>,
    // Autodifferentiation fields
    pub grad: Option<TensorData>, // Gradient with respect to this value``
    pub requires_grad: bool,      // Whether this value needs gradients
    pub retain_grad: bool,
}

/// Reference-counted pointer to a Value
pub type ValueRef = Rc<RefCell<Value>>;

impl Value {
    /// Create a new value with autodifferentiation support
    pub fn new(
        data: Option<TensorData>,
        operation: Box<dyn TensorOp>,
        dependencies: Vec<ValueRef>,
        requires_grad: bool,
    ) -> Self {
        let id = Uuid::new_v4().to_string();
        let shape = data.as_ref().map(|d| d.shape().to_vec());
        let grad = shape.map(ndarray::Array::zeros);

        Value {
            id,
            data,
            operation,
            dependencies,
            grad,
            requires_grad,
            retain_grad: false,
        }
    }

    /// Accumulate gradient (for handling multiple paths in computation graph)
    pub fn accumulate_grad(&mut self, grad: &TensorData) {
        if let Some(g) = self.grad.as_mut() {
            if Self::is_1d(grad.shape()) {
                let g1 = grad[0];
                *g += g1;
            } else if Self::is_1d(g.shape()) {
                let g1 = g[0];
                *g = grad + g1;
            } else {
                *g += grad;
            }
        } else {
            self.grad = Some(grad.clone());
        }
    }

    /// Clear the gradients of this value and all its dependencies
    pub fn zero_grad(&mut self) {
        self.grad = None;
        self.dependencies
            .iter()
            .for_each(|v| v.borrow_mut().zero_grad());
    }

    /// Backward pass for this value
    pub fn backward(&self) {
        if self.requires_grad {
            self.operation
                .backward(&self.dependencies, self.grad.as_ref().unwrap());
        }
    }

    pub fn eval(&mut self) {
        if self.operation.is_param() {
            return;
        }

        self.data = Some(self.operation.eval(&self.dependencies));
    }

    pub fn is_1d(shape: &[usize]) -> bool {
        shape.len() == 1
    }
}

pub fn toposort_graph(value: &ValueRef) -> Vec<ValueRef> {
    let mut sorted = Vec::new();
    let mut visited = HashSet::<String>::new();

    fn visit(value: &ValueRef, sorted: &mut Vec<ValueRef>, visited: &mut HashSet<String>) {
        if visited.contains(&value.borrow().id) {
            return;
        }

        visited.insert(value.borrow().id.clone());

        for dependency in &value.borrow().dependencies {
            visit(dependency, sorted, visited);
        }

        sorted.push(value.clone());
    }

    visit(value, &mut sorted, &mut visited);
    sorted
}
