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

/// Value represents either an input or the result of an operation
#[derive(Debug)]
pub struct Value {
    pub id: usize,
    pub data: Option<f32>, // Concrete value if computed
    pub operation: Box<dyn TensorOp>,
    pub dependencies: Vec<ValueRef>,
    // Autodifferentiation fields
    pub grad: f32,           // Gradient with respect to this value
    pub requires_grad: bool, // Whether this value needs gradients
}

/// Reference-counted pointer to a Value
pub type ValueRef = Rc<RefCell<Value>>;

/// A tensor in our computation graph
#[derive(Debug, Clone)]
pub struct Tensor {
    pub values: Vec<ValueRef>,
    pub shape: Vec<usize>,
}

impl Value {
    /// Create a new value with autodifferentiation support
    pub fn new(
        id: usize,
        data: Option<f32>,
        operation: Box<dyn TensorOp>,
        dependencies: Vec<ValueRef>,
        requires_grad: bool,
    ) -> Self {
        Value {
            id,
            data,
            operation,
            dependencies,
            grad: 0.0,
            requires_grad,
        }
    }

    /// Accumulate gradient (for handling multiple paths in computation graph)
    pub fn accumulate_grad(&mut self, grad: f32) {
        self.grad += grad;
    }

    /// Clear the gradient
    pub fn zero_grad(&mut self) {
        self.grad = 0.0;
    }

    /// Backward pass for this value
    pub fn backward(&self) {
        self.operation.backward(&self.dependencies, self.grad);
    }
}

impl Tensor {
    /// Zero all gradients in the tensor
    pub fn zero_grad(&self) {
        for value in &self.values {
            value.borrow_mut().zero_grad();
        }
    }

    /// Backward pass through the entire tensor
    pub fn backward(&self) {
        // Start with gradient of 1.0 for the output tensor
        for value in &self.values {
            let mut value_mut = value.borrow_mut();
            if value_mut.requires_grad {
                value_mut.grad = 1.0;
            }
        }

        // Perform backward pass for each value
        for value in &self.values {
            value.borrow().backward();
        }
    }

    /// Get gradients for all values in the tensor
    pub fn gradients(&self) -> Vec<f32> {
        self.values.iter().map(|v| v.borrow().grad).collect()
    }

    /// Update values using gradients (for optimization)
    pub fn update(&mut self, learning_rate: f32) {
        for value in &self.values {
            let mut value_mut = value.borrow_mut();
            if value_mut.requires_grad {
                let grad = value_mut.grad;

                if let Some(ref mut data) = value_mut.data {
                    *data -= learning_rate * grad;
                }
            }
        }
    }
}

impl<D: ndarray::Dimension> From<ndarray::ArrayBase<ndarray::ViewRepr<&f32>, D>> for Tensor {
    fn from(array: ndarray::ArrayBase<ndarray::ViewRepr<&f32>, D>) -> Self {
        let shape = array.shape().to_vec();
        let values = array
            .iter()
            .map(|v| {
                Rc::new(RefCell::new(Value {
                    id: rand::random::<f32>() as usize,
                    data: Some(*v),
                    operation: Box::new(InputOp),
                    dependencies: Vec::new(),
                    grad: 0.0,
                    requires_grad: true,
                }))
            })
            .collect();

        Tensor { values, shape }
    }
}

impl<D: ndarray::Dimension> From<ndarray::ArrayBase<ndarray::OwnedRepr<f32>, D>> for Tensor {
    fn from(array: ndarray::ArrayBase<ndarray::OwnedRepr<f32>, D>) -> Self {
        let shape = array.shape().to_vec();
        let values = array
            .iter()
            .map(|v| {
                Rc::new(RefCell::new(Value {
                    id: rand::random::<f32>() as usize,
                    data: Some(*v),
                    operation: Box::new(InputOp),
                    dependencies: Vec::new(),
                    grad: 0.0,
                    requires_grad: true,
                }))
            })
            .collect();

        Tensor { values, shape }
    }
}

impl<D: ndarray::Dimension> From<ndarray::ArrayBase<ndarray::CowRepr<'_, f32>, D>> for Tensor {
    fn from(array: ndarray::ArrayBase<ndarray::CowRepr<'_, f32>, D>) -> Self {
        let shape = array.shape().to_vec();
        let values = array
            .iter()
            .map(|v| {
                Rc::new(RefCell::new(Value {
                    id: rand::random::<f32>() as usize,
                    data: Some(*v),
                    operation: Box::new(InputOp),
                    dependencies: Vec::new(),
                    grad: 0.0,
                    requires_grad: true,
                }))
            })
            .collect();

        Tensor { values, shape }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_autodiff_basic() {
        // Create input tensors
        let x = Tensor::new(vec![2, 2], true);
        let y = Tensor::new(vec![2, 2], true);

        // Set some initial values
        x.values[0].borrow_mut().data = Some(2.0);
        x.values[1].borrow_mut().data = Some(3.0);
        x.values[2].borrow_mut().data = Some(4.0);
        x.values[3].borrow_mut().data = Some(5.0);

        y.values[0].borrow_mut().data = Some(1.0);
        y.values[1].borrow_mut().data = Some(2.0);
        y.values[2].borrow_mut().data = Some(3.0);
        y.values[3].borrow_mut().data = Some(4.0);

        // Create computation graph: z = (x + y) * 2 + relu(x)
        let z1 = x.add(&y); // x + y
        let z2 = z1.relu(); // relu(x + y)
        let z3 = z2.mean(); // mean(relu(x + y))

        // Zero gradients before backward pass
        x.zero_grad();
        y.zero_grad();

        // Backward pass
        z3.backward();

        // Check that gradients were computed
        let x_grads = x.gradients();
        let y_grads = y.gradients();

        println!("X gradients: {:?}", x_grads);
        println!("Y gradients: {:?}", y_grads);

        // Gradients should be non-zero
        assert!(x_grads.iter().any(|&g| g != 0.0));
        assert!(y_grads.iter().any(|&g| g != 0.0));
    }

    #[test]
    fn test_autodiff_optimization() {
        // Simple optimization example: minimize f(x) = x^2 + 2x + 1
        let mut x = Tensor::new(vec![1], true);
        x.values[0].borrow_mut().data = Some(3.0); // Start at x = 3

        let learning_rate = 0.1;

        for step in 0..10 {
            // Zero gradients
            x.zero_grad();

            // Forward pass: f(x) = x^2 + 2x + 1
            let x_squared = x.mult(&x); // x^2
            let two_x = x.add(&x); // 2x
            let x_sq_plus_2x = x_squared.add(&two_x); // x^2 + 2x
            let one = Tensor::new(vec![1], true);
            one.values[0].borrow_mut().data = Some(1.0);
            let loss = x_sq_plus_2x.add(&one); // x^2 + 2x + 1

            // Backward pass
            loss.backward();

            // Update parameters
            x.update(learning_rate);

            let current_x = x.values[0].borrow().data.unwrap();
            println!(
                "Step {}: x = {:.4}, loss = {:.4}",
                step,
                current_x,
                loss.values[0].borrow().data.unwrap_or(0.0)
            );
        }

        // After optimization, x should be close to -1 (the minimum of f(x) = x^2 + 2x + 1)
        let final_x = x.values[0].borrow().data.unwrap();
        assert!((final_x - (-1.0)).abs() < 0.1);
    }
}
