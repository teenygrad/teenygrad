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

pub use shape::*;

use crate::tensor::tensor_ops::TensorOp;
use crate::tensor::tensor_ops::input_op::InputOp;

pub type TensorData = ndarray::ArrayBase<ndarray::OwnedRepr<f32>, ndarray::IxDyn>;

/// Value represents either an input or the result of an operation
#[derive(Debug)]
pub struct Value {
    pub id: usize,
    pub data: Option<TensorData>, // Concrete value if computed
    pub operation: Box<dyn TensorOp>,
    pub dependencies: Vec<ValueRef>,
    // Autodifferentiation fields
    pub grad: Option<TensorData>, // Gradient with respect to this value``
    pub requires_grad: bool,      // Whether this value needs gradients
}

/// Reference-counted pointer to a Value
pub type ValueRef = Rc<RefCell<Value>>;

/// A tensor in our computation graph
#[derive(Debug, Clone)]
pub struct Tensor {
    pub value: ValueRef,
    pub shape: Vec<usize>,
}

impl Value {
    /// Create a new value with autodifferentiation support
    pub fn new(
        id: usize,
        data: Option<TensorData>,
        operation: Box<dyn TensorOp>,
        dependencies: Vec<ValueRef>,
        requires_grad: bool,
    ) -> Self {
        let shape = data.as_ref().map(|d| d.shape().to_vec());
        let grad = shape.map(ndarray::Array::zeros);

        Value {
            id,
            data,
            operation,
            dependencies,
            grad,
            requires_grad,
        }
    }

    /// Accumulate gradient (for handling multiple paths in computation graph)
    pub fn accumulate_grad(&mut self, grad: &TensorData) {
        if let Some(g) = self.grad.as_mut() {
            *g += grad;
        }
    }

    /// Clear the gradient
    pub fn zero_grad(&mut self) {
        self.grad = None;
    }

    /// Backward pass for this value
    pub fn backward(&self) {
        self.operation
            .backward(&self.dependencies, self.grad.as_ref().unwrap());
    }

    pub fn eval(&mut self) {
        if self.operation.is_input() {
            return;
        }

        self.data = Some(self.operation.eval(&self.dependencies));
    }
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
        let mut value = self.value.borrow_mut();
        let shape = value.data.as_ref().unwrap().shape();

        // Start with gradient of 1.0 for the output tensor
        value.grad = Some(ndarray::Array::ones(shape));

        // Perform backward pass for each value
        value.backward();
    }

    /// Get gradients for all values in the tensor
    pub fn gradients(&self) -> TensorData {
        self.value.borrow().grad.as_ref().unwrap().clone()
    }

    /// Update values using gradients (for optimization)
    pub fn update(&mut self, learning_rate: f32) {
        let grad = self.value.borrow().grad.as_ref().unwrap().clone();

        if let Some(ref mut data) = self.value.borrow_mut().data {
            *data = learning_rate * grad;
        }
    }
}

impl<D: ndarray::Dimension> From<ndarray::ArrayBase<ndarray::ViewRepr<&f32>, D>> for Tensor {
    fn from(array: ndarray::ArrayBase<ndarray::ViewRepr<&f32>, D>) -> Self {
        let shape = array.shape().to_vec();
        let data = array.to_owned().into_dyn();
        let value = Rc::new(RefCell::new(Value {
            id: rand::random::<f32>() as usize,
            data: Some(data),
            operation: Box::new(InputOp),
            dependencies: Vec::new(),
            grad: Some(TensorData::zeros(shape.clone())),
            requires_grad: true,
        }));

        Tensor { value, shape }
    }
}

impl<D: ndarray::Dimension> From<ndarray::ArrayBase<ndarray::OwnedRepr<f32>, D>> for Tensor {
    fn from(array: ndarray::ArrayBase<ndarray::OwnedRepr<f32>, D>) -> Self {
        let shape = array.shape().to_vec();
        let value = Rc::new(RefCell::new(Value {
            id: rand::random::<f32>() as usize,
            data: Some(array.to_owned().into_dyn()),
            operation: Box::new(InputOp),
            dependencies: Vec::new(),
            grad: Some(TensorData::zeros(array.shape().to_vec())),
            requires_grad: true,
        }));

        Tensor { value, shape }
    }
}

impl<D: ndarray::Dimension> From<ndarray::ArrayBase<ndarray::CowRepr<'_, f32>, D>> for Tensor {
    fn from(array: ndarray::ArrayBase<ndarray::CowRepr<'_, f32>, D>) -> Self {
        let shape = array.shape().to_vec();
        let value = Rc::new(RefCell::new(Value {
            id: rand::random::<f32>() as usize,
            data: Some(array.to_owned().into_dyn()),
            operation: Box::new(InputOp),
            dependencies: Vec::new(),
            grad: Some(TensorData::zeros(array.shape().to_vec())),
            requires_grad: true,
        }));

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
