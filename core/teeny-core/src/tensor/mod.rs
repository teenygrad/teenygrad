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
}

/// Reference-counted pointer to a Value
pub type ValueRef = Rc<RefCell<Value>>;

/// Operations that can be performed
#[derive(Debug, Clone)]
pub enum Operation {
    Input,
    Add,
    Transpose,
    Multiply,
    MatrixMultiply,
    Convolution2D,
    ReLU,
    // Other operations...
}

/// A tensor in our computation graph
#[derive(Debug, Clone)]
pub struct Tensor {
    pub values: Vec<ValueRef>,
    pub shape: Vec<usize>,
}

impl Tensor {
    pub fn new(shape: Vec<usize>) -> Self {
        let size = shape.iter().product();
        let mut values = Vec::with_capacity(size);

        for _ in 0..size {
            values.push(Rc::new(RefCell::new(Value {
                id: rand::random::<f32>() as usize,
                data: None,
                operation: Operation::Input,
                dependencies: Vec::new(),
            })));
        }

        Tensor { values, shape }
    }

    pub fn add(&self, other: &Tensor) -> Tensor {
        assert_eq!(self.shape, other.shape, "Shape mismatch in addition");

        let mut result_values = Vec::with_capacity(self.values.len());

        for (a, b) in self.values.iter().zip(other.values.iter()) {
            result_values.push(Rc::new(RefCell::new(Value {
                id: rand::random::<f32>() as usize,
                data: None,
                operation: Operation::Add,
                dependencies: vec![a.clone(), b.clone()],
            })));
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

                result_values.push(Rc::new(RefCell::new(Value {
                    id: rand::random::<f32>() as usize,
                    data: None,
                    operation: Operation::MatrixMultiply,
                    dependencies,
                })));
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
            result_values.push(Rc::new(RefCell::new(Value {
                id: rand::random::<f32>() as usize,
                data: None,
                operation: Operation::ReLU,
                dependencies: vec![value.clone()],
            })));
        }

        Tensor {
            values: result_values,
            shape: self.shape.clone(),
        }
    }

    pub fn transpose(&self) -> Tensor {
        let mut result_values = Vec::with_capacity(self.values.len());

        for value in &self.values {
            result_values.push(Rc::new(RefCell::new(Value {
                id: rand::random::<f32>() as usize,
                data: None,
                operation: Operation::Transpose,
                dependencies: vec![value.clone()],
            })));
        }

        Tensor {
            values: result_values,
            shape: self.shape.clone(),
        }
    }
}
