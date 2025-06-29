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

pub use shape::*;

/// Value represents either an input or the result of an operation
#[derive(Debug, Clone)]
pub struct Value {
    pub id: usize,
    pub data: Option<f32>, // Concrete value if computed
    pub operation: Operation,
    pub dependencies: Vec<ValueRef>,
    // Autodifferentiation fields
    pub grad: f32,           // Gradient with respect to this value
    pub requires_grad: bool, // Whether this value needs gradients
}

/// Reference-counted pointer to a Value
pub type ValueRef = Rc<RefCell<Value>>;

/// Operations that can be performed
#[derive(Debug, Clone)]
pub enum Operation {
    Input,
    Add,
    Sub,
    Transpose,
    Multiply,
    MatrixMultiply,
    Convolution2D,
    ReLU,
    Sigmoid,
    Log,
    Mean,
    // Other operations...
}

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
        operation: Operation,
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
        match self.operation {
            Operation::Add => {
                if self.dependencies.len() >= 2 {
                    let grad = self.grad;
                    if self.dependencies[0].borrow().requires_grad {
                        self.dependencies[0].borrow_mut().accumulate_grad(grad);
                    }
                    if self.dependencies[1].borrow().requires_grad {
                        self.dependencies[1].borrow_mut().accumulate_grad(grad);
                    }
                }
            }
            Operation::Sub => {
                if self.dependencies.len() >= 2 {
                    let grad = self.grad;
                    if self.dependencies[0].borrow().requires_grad {
                        self.dependencies[0].borrow_mut().accumulate_grad(grad);
                    }
                    if self.dependencies[1].borrow().requires_grad {
                        self.dependencies[1].borrow_mut().accumulate_grad(-grad);
                    }
                }
            }
            Operation::ReLU => {
                if !self.dependencies.is_empty() {
                    let grad = self.grad;
                    if self.dependencies[0].borrow().requires_grad {
                        let input_val = self.dependencies[0].borrow().data.unwrap_or(0.0);
                        let relu_grad = if input_val > 0.0 { grad } else { 0.0 };
                        self.dependencies[0].borrow_mut().accumulate_grad(relu_grad);
                    }
                }
            }
            Operation::Sigmoid => {
                if !self.dependencies.is_empty() {
                    let grad = self.grad;
                    if self.dependencies[0].borrow().requires_grad {
                        let input_val = self.dependencies[0].borrow().data.unwrap_or(0.0);
                        let sigmoid_val = 1.0 / (1.0 + (-input_val).exp());
                        let sigmoid_grad = grad * sigmoid_val * (1.0 - sigmoid_val);
                        self.dependencies[0]
                            .borrow_mut()
                            .accumulate_grad(sigmoid_grad);
                    }
                }
            }
            Operation::Log => {
                if !self.dependencies.is_empty() {
                    let grad = self.grad;
                    if self.dependencies[0].borrow().requires_grad {
                        let input_val = self.dependencies[0].borrow().data.unwrap_or(0.0);
                        let log_grad = if input_val > 0.0 {
                            grad / input_val
                        } else {
                            0.0
                        };
                        self.dependencies[0].borrow_mut().accumulate_grad(log_grad);
                    }
                }
            }
            Operation::Mean => {
                let grad = self.grad;
                let n = self.dependencies.len() as f32;
                let grad_per_element = grad / n;

                for dep in &self.dependencies {
                    if dep.borrow().requires_grad {
                        dep.borrow_mut().accumulate_grad(grad_per_element);
                    }
                }
            }
            Operation::Transpose => {
                if !self.dependencies.is_empty() {
                    let grad = self.grad;
                    if self.dependencies[0].borrow().requires_grad {
                        self.dependencies[0].borrow_mut().accumulate_grad(grad);
                    }
                }
            }
            Operation::MatrixMultiply => {
                // Simplified matrix multiplication gradient
                // In practice, this would need more sophisticated logic
                let grad = self.grad;
                for dep in &self.dependencies {
                    if dep.borrow().requires_grad {
                        dep.borrow_mut().accumulate_grad(grad);
                    }
                }
            }
            _ => {
                // For other operations, just propagate gradients
                let grad = self.grad;
                for dep in &self.dependencies {
                    if dep.borrow().requires_grad {
                        dep.borrow_mut().accumulate_grad(grad);
                    }
                }
            }
        }
    }
}

impl Tensor {
    pub fn new(shape: Vec<usize>) -> Self {
        let size = shape.iter().product();
        let mut values = Vec::with_capacity(size);

        for _ in 0..size {
            values.push(Rc::new(RefCell::new(Value::new(
                rand::random::<f32>() as usize,
                None,
                Operation::Input,
                Vec::new(),
                true, // Default to requiring gradients
            ))));
        }

        Tensor { values, shape }
    }

    /// Create a tensor that doesn't require gradients (for inputs that don't need to be optimized)
    pub fn new_no_grad(shape: Vec<usize>) -> Self {
        let size = shape.iter().product();
        let mut values = Vec::with_capacity(size);

        for _ in 0..size {
            values.push(Rc::new(RefCell::new(Value::new(
                rand::random::<f32>() as usize,
                None,
                Operation::Input,
                Vec::new(),
                false,
            ))));
        }

        Tensor { values, shape }
    }

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

    pub fn add(&self, other: &Tensor) -> Tensor {
        assert_eq!(self.shape, other.shape, "Shape mismatch in addition");

        let mut result_values = Vec::with_capacity(self.values.len());

        for (a, b) in self.values.iter().zip(other.values.iter()) {
            let result_value = Value::new(
                rand::random::<f32>() as usize,
                None,
                Operation::Add,
                vec![a.clone(), b.clone()],
                a.borrow().requires_grad || b.borrow().requires_grad,
            );

            result_values.push(Rc::new(RefCell::new(result_value)));
        }

        Tensor {
            values: result_values,
            shape: self.shape.clone(),
        }
    }

    pub fn sub(&self, other: &Tensor) -> Tensor {
        assert_eq!(self.shape, other.shape, "Shape mismatch in subtraction");

        let mut result_values = Vec::with_capacity(self.values.len());

        for (a, b) in self.values.iter().zip(other.values.iter()) {
            let result_value = Value::new(
                rand::random::<f32>() as usize,
                None,
                Operation::Sub,
                vec![a.clone(), b.clone()],
                a.borrow().requires_grad || b.borrow().requires_grad,
            );

            result_values.push(Rc::new(RefCell::new(result_value)));
        }

        Tensor {
            values: result_values,
            shape: self.shape.clone(),
        }
    }

    pub fn mult(&self, other: &Tensor) -> Tensor {
        // Matrix multiplication that returns a new tensor with graph nodes
        // For simplicity, we'll assume 2D matrices
        assert_eq!(self.shape.len(), 2, "matmul requires 2D tensors");
        assert_eq!(other.shape.len(), 2, "matmul requires 2D tensors");
        assert_eq!(self.shape[1], other.shape[0], "matmul dimension mismatch");

        let rows = self.shape[0];
        let cols = other.shape[1];
        let mut result_values = Vec::with_capacity(rows * cols);

        for i in 0..rows {
            for j in 0..cols {
                let mut dependencies = Vec::new();

                for k in 0..self.shape[1] {
                    let a_idx = i * self.shape[1] + k;
                    let b_idx = k * other.shape[1] + j;

                    dependencies.push(self.values[a_idx].clone());
                    dependencies.push(other.values[b_idx].clone());
                }

                let result_value = Value::new(
                    rand::random::<f32>() as usize,
                    None,
                    Operation::MatrixMultiply,
                    dependencies,
                    self.values.iter().any(|v| v.borrow().requires_grad)
                        || other.values.iter().any(|v| v.borrow().requires_grad),
                );

                result_values.push(Rc::new(RefCell::new(result_value)));
            }
        }

        Tensor {
            values: result_values,
            shape: vec![rows, cols],
        }
    }

    pub fn relu(&self) -> Tensor {
        let mut result_values = Vec::with_capacity(self.values.len());

        for value in &self.values {
            let result_value = Value::new(
                rand::random::<f32>() as usize,
                None,
                Operation::ReLU,
                vec![value.clone()],
                value.borrow().requires_grad,
            );

            result_values.push(Rc::new(RefCell::new(result_value)));
        }

        Tensor {
            values: result_values,
            shape: self.shape.clone(),
        }
    }

    pub fn sigmoid(&self) -> Tensor {
        let mut result_values = Vec::with_capacity(self.values.len());

        for value in &self.values {
            let result_value = Value::new(
                rand::random::<f32>() as usize,
                None,
                Operation::Sigmoid,
                vec![value.clone()],
                value.borrow().requires_grad,
            );

            result_values.push(Rc::new(RefCell::new(result_value)));
        }

        Tensor {
            values: result_values,
            shape: self.shape.clone(),
        }
    }

    pub fn transpose(&self) -> Tensor {
        let mut result_values = Vec::with_capacity(self.values.len());

        for value in &self.values {
            let result_value = Value::new(
                rand::random::<f32>() as usize,
                None,
                Operation::Transpose,
                vec![value.clone()],
                value.borrow().requires_grad,
            );

            result_values.push(Rc::new(RefCell::new(result_value)));
        }

        Tensor {
            values: result_values,
            shape: self.shape.clone(),
        }
    }

    pub fn log(&self) -> Tensor {
        let mut result_values = Vec::with_capacity(self.values.len());

        for value in &self.values {
            let result_value = Value::new(
                rand::random::<f32>() as usize,
                None,
                Operation::Log,
                vec![value.clone()],
                value.borrow().requires_grad,
            );

            result_values.push(Rc::new(RefCell::new(result_value)));
        }

        Tensor {
            values: result_values,
            shape: self.shape.clone(),
        }
    }

    pub fn mean(&self) -> Tensor {
        let mut result_values = Vec::with_capacity(1);

        let result_value = Value::new(
            rand::random::<f32>() as usize,
            None,
            Operation::Mean,
            self.values.clone(),
            self.values.iter().any(|v| v.borrow().requires_grad),
        );

        result_values.push(Rc::new(RefCell::new(result_value)));

        Tensor {
            values: result_values,
            shape: vec![1],
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
                    operation: Operation::Input,
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
                    operation: Operation::Input,
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
                    operation: Operation::Input,
                    dependencies: Vec::new(),
                    grad: 0.0,
                    requires_grad: true,
                }))
            })
            .collect();

        Tensor { values, shape }
    }
}

/// Example usage of the autodifferentiation system
pub fn autodiff_example() {
    println!("=== Teenygrad Autodifferentiation Example ===");

    // Create input tensors
    let a = Tensor::new(vec![2, 2]);
    let b = Tensor::new(vec![2, 2]);

    // Set initial values
    a.values[0].borrow_mut().data = Some(1.0);
    a.values[1].borrow_mut().data = Some(2.0);
    a.values[2].borrow_mut().data = Some(3.0);
    a.values[3].borrow_mut().data = Some(4.0);

    b.values[0].borrow_mut().data = Some(5.0);
    b.values[1].borrow_mut().data = Some(6.0);
    b.values[2].borrow_mut().data = Some(7.0);
    b.values[3].borrow_mut().data = Some(8.0);

    println!("Input tensor A:");
    for (i, val) in a.values.iter().enumerate() {
        println!("  A[{}] = {}", i, val.borrow().data.unwrap_or(0.0));
    }

    println!("Input tensor B:");
    for (i, val) in b.values.iter().enumerate() {
        println!("  B[{}] = {}", i, val.borrow().data.unwrap_or(0.0));
    }

    // Create computation graph: result = relu(A + B) * 2
    let c = a.add(&b); // A + B
    let d = c.relu(); // relu(A + B)
    let result = d.mean(); // mean(relu(A + B))

    println!("Computation: result = mean(relu(A + B))");

    // Zero gradients
    a.zero_grad();
    b.zero_grad();

    // Backward pass
    result.backward();

    // Print gradients
    println!("Gradients for A:");
    for (i, grad) in a.gradients().iter().enumerate() {
        println!("  ∂result/∂A[{}] = {}", i, grad);
    }

    println!("Gradients for B:");
    for (i, grad) in b.gradients().iter().enumerate() {
        println!("  ∂result/∂B[{}] = {}", i, grad);
    }

    println!("=== End Example ===");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_autodiff_basic() {
        // Create input tensors
        let x = Tensor::new(vec![2, 2]);
        let y = Tensor::new(vec![2, 2]);

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
        let mut x = Tensor::new(vec![1]);
        x.values[0].borrow_mut().data = Some(3.0); // Start at x = 3

        let learning_rate = 0.1;

        for step in 0..10 {
            // Zero gradients
            x.zero_grad();

            // Forward pass: f(x) = x^2 + 2x + 1
            let x_squared = x.mult(&x); // x^2
            let two_x = x.add(&x); // 2x
            let x_sq_plus_2x = x_squared.add(&two_x); // x^2 + 2x
            let one = Tensor::new(vec![1]);
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
