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

use std::cell::RefCell;
use std::rc::Rc;

pub mod shape;
pub mod tensor_ops;
pub mod value;

pub use shape::*;

use crate::tensor::tensor_ops::input_op::InputOp;
use crate::tensor::tensor_ops::log_op::LogOp;
use crate::tensor::value::{TensorData, Value, ValueRef};

/// A tensor in our computation graph
#[derive(Debug, Clone)]
pub struct Tensor {
    pub value: ValueRef,
    pub shape: Vec<usize>,
}

impl Tensor {
    /// Zero all gradients in the tensor
    pub fn zero_grad(&self) {
        self.value.borrow_mut().zero_grad();
    }

    pub fn eval(&self) -> TensorData {
        self.value.borrow_mut().eval();
        self.value.borrow().data.clone().unwrap()
    }

    /// Backward pass through the entire tensor
    pub fn backward(&self) {
        let value = self.value.borrow_mut();
        value.backward();
    }

    /// Get gradients for all values in the tensor
    pub fn grad(&self) -> Option<TensorData> {
        self.value.borrow().grad.clone()
    }

    /// Update values using gradients (for optimization)
    pub fn update(&mut self, learning_rate: f32) {
        let grad = self.value.borrow().grad.as_ref().unwrap().clone();

        if let Some(ref mut data) = self.value.borrow_mut().data {
            *data = learning_rate * grad;
        }
    }
}

pub fn log(x: Tensor) -> Tensor {
    let requires_grad = x.value.borrow().requires_grad;

    let value = Rc::new(RefCell::new(Value::new(
        None,
        Box::new(LogOp),
        vec![x.value.clone()],
        requires_grad,
    )));

    Tensor {
        value,
        shape: x.shape.clone(),
    }
}

impl<D: ndarray::Dimension> From<ndarray::ArrayBase<ndarray::ViewRepr<&f32>, D>> for Tensor {
    fn from(array: ndarray::ArrayBase<ndarray::ViewRepr<&f32>, D>) -> Self {
        let shape = array.shape().to_vec();
        let data = array.to_owned().into_dyn();
        let value = Rc::new(RefCell::new(Value::new(
            Some(data),
            Box::new(InputOp),
            Vec::new(),
            true,
        )));

        Tensor { value, shape }
    }
}

impl<D: ndarray::Dimension> From<ndarray::ArrayBase<ndarray::OwnedRepr<f32>, D>> for Tensor {
    fn from(array: ndarray::ArrayBase<ndarray::OwnedRepr<f32>, D>) -> Self {
        let shape = array.shape().to_vec();
        let value = Rc::new(RefCell::new(Value::new(
            Some(array.to_owned().into_dyn()),
            Box::new(InputOp),
            Vec::new(),
            true,
        )));

        Tensor { value, shape }
    }
}

impl<D: ndarray::Dimension> From<ndarray::ArrayBase<ndarray::CowRepr<'_, f32>, D>> for Tensor {
    fn from(array: ndarray::ArrayBase<ndarray::CowRepr<'_, f32>, D>) -> Self {
        let shape = array.shape().to_vec();
        let value = Rc::new(RefCell::new(Value::new(
            Some(array.to_owned().into_dyn()),
            Box::new(InputOp),
            Vec::new(),
            true,
        )));

        Tensor { value, shape }
    }
}

#[cfg(test)]
mod tests {
    use ndarray::array;

    use super::*;

    #[test]
    fn test_autodiff_basic() {
        // Create input tensors
        let x: Tensor = array![[2.0, 3.0], [4.0, 5.0]].into();
        let y: Tensor = array![[1.0, 2.0], [3.0, 4.0]].into();

        // Create computation graph: z = (x + y) * 2 + relu(x)
        let z1 = x + y; // x + y
        let z2 = z1.relu(); // relu(x + y)
        let z3 = z2.mean(); // mean(relu(x + y))

        // zero gradients
        z3.zero_grad();

        // Backward pass
        z3.backward();
    }

    #[test]
    fn test_autodiff_optimization() {
        // Simple optimization example: minimize f(x) = x^2 + 2x + 1
        let x_shape = vec![1];
        let mut x = Tensor::new(ndarray::Array::zeros(x_shape), true);
        x.value.borrow_mut().data = Some(ndarray::Array::from_elem(vec![1], 3.0)); // Start at x = 3

        let learning_rate = 0.1;

        for step in 0..10 {
            // Zero gradients
            x.zero_grad();

            // Forward pass: f(x) = x^2 + 2x + 1
            let x_squared = &x * &x; // x^2
            let two_x = 2.0 * &x; // 2x
            let x_sq_plus_2x = x_squared + two_x; // x^2 + 2x
            let one_shape = vec![1];
            let one = Tensor::new(ndarray::Array::zeros(one_shape), true);
            one.value.borrow_mut().data = Some(ndarray::Array::from_elem(vec![1], 1.0));
            let loss = x_sq_plus_2x + one; // x^2 + 2x + 1

            // Backward pass
            loss.backward();

            // Update parameters
            x.update(learning_rate);

            let current_x = x.value.borrow().data.as_ref().unwrap().clone();
            println!(
                "Step {}: x = {:.4}, loss = {:.4}",
                step,
                current_x,
                loss.value.borrow().data.as_ref().unwrap()
            );
        }

        // After optimization, x should be close to -1 (the minimum of f(x) = x^2 + 2x + 1)
        let final_x = x.value.borrow().data.as_ref().unwrap().clone();
        assert!(final_x.iter().any(|&v| (v - (-1.0)).abs() < 0.1));
    }
}
