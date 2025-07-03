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

use std::{cell::RefCell, ops::Mul, rc::Rc};

use crate::tensor::{Tensor, TensorData, Value, ValueRef, tensor_ops::TensorOp};

#[derive(Debug, Clone)]
pub struct MultOp;

impl TensorOp for MultOp {
    fn eval(&self, dependencies: &[ValueRef]) -> TensorData {
        assert_eq!(dependencies.len(), 2);

        dependencies.iter().for_each(|v| v.borrow_mut().eval());

        dependencies[0].borrow().data.as_ref().unwrap()
            * dependencies[1].borrow().data.as_ref().unwrap()
    }

    fn backward(&self, dependencies: &[ValueRef], grad: &TensorData) {
        assert_eq!(dependencies.len(), 2);
        let mut a = dependencies[0].borrow_mut();
        let mut b = dependencies[1].borrow_mut();

        let grad_a = grad * b.data.clone().unwrap();
        let grad_b = grad * a.data.clone().unwrap();

        a.accumulate_grad(&grad_a);
        b.accumulate_grad(&grad_b);
    }
}

impl Mul<Tensor> for Tensor {
    type Output = Tensor;

    fn mul(self, other: Tensor) -> Self::Output {
        let requires_grad = self.value.borrow().requires_grad || other.value.borrow().requires_grad;

        let value = Rc::new(RefCell::new(Value::new(
            rand::random::<f32>() as usize,
            None,
            Box::new(MultOp),
            vec![self.value.clone(), other.value.clone()],
            requires_grad,
        )));

        Tensor {
            value,
            shape: self.shape.clone(),
        }
    }
}

impl Mul<f32> for Tensor {
    type Output = Tensor;

    fn mul(self, other: f32) -> Self::Output {
        self.mul(Tensor::new(ndarray::Array::from_elem(vec![1], other), true))
    }
}

impl Mul<Tensor> for f32 {
    type Output = Tensor;

    fn mul(self, other: Tensor) -> Self::Output {
        other.mul(Tensor::new(ndarray::Array::from_elem(vec![1], self), true))
    }
}

impl Mul<&Tensor> for f32 {
    type Output = Tensor;

    fn mul(self, other: &Tensor) -> Self::Output {
        other.mul(Tensor::new(ndarray::Array::from_elem(vec![1], self), true))
    }
}

impl Mul<&Tensor> for Tensor {
    type Output = Tensor;

    fn mul(self, other: &Tensor) -> Self::Output {
        self.mul(other.clone())
    }
}

impl Mul<Tensor> for &Tensor {
    type Output = Tensor;

    fn mul(self, other: Tensor) -> Self::Output {
        self.clone().mul(other)
    }
}

impl Mul<&Tensor> for &Tensor {
    type Output = Tensor;

    fn mul(self, other: &Tensor) -> Self::Output {
        self.clone().mul(other.clone())
    }
}
