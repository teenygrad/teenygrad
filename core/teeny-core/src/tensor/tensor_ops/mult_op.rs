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

use ndarray::{ArrayBase, CowRepr, Dim};

use crate::tensor::{Tensor, TensorData, Value, ValueRef, tensor_ops::TensorOp};

#[derive(Debug, Clone)]
pub struct MultOp;

impl MultOp {
    fn is_1d(data: &TensorData) -> bool {
        data.shape().len() == 1
    }

    fn is_2d(data: &TensorData) -> bool {
        data.shape().len() == 2
    }

    fn to_2d(data: &TensorData) -> ArrayBase<CowRepr<f32>, Dim<[usize; 2]>> {
        data.to_shape((data.shape()[0], data.shape()[1])).unwrap()
    }
}

impl TensorOp for MultOp {
    fn eval(&self, dependencies: &[ValueRef]) -> TensorData {
        assert_eq!(dependencies.len(), 2);

        dependencies.iter().for_each(|v| v.borrow_mut().eval());

        let a = dependencies[0].borrow();
        let b = dependencies[1].borrow();

        let a_data = a.data.as_ref().unwrap();
        let b_data = b.data.as_ref().unwrap();

        if Self::is_2d(a_data) && Self::is_2d(b_data) {
            let a_2d = Self::to_2d(a_data);
            let b_2d = Self::to_2d(b_data);
            let result = a_2d.dot(&b_2d);
            result.into_dyn()
        } else if Self::is_2d(a_data) && Self::is_1d(b_data) {
            let a_2d = Self::to_2d(a_data);
            let b_1d = b_data[0];
            let result = a_2d * b_1d;
            result.to_owned().into_dyn()
        } else {
            panic!(
                "A and B must be 1D or 2D, got {:?}, {:?} and {:?}, {:?}, {:?}",
                a_data.shape().len(),
                a_data.shape(),
                b_data.shape().len(),
                b_data.shape(),
                b_data
            );
        }
    }

    fn backward(&self, dependencies: &[ValueRef], grad: &TensorData) {
        assert_eq!(dependencies.len(), 2);
        let mut a = dependencies[0].borrow_mut();
        let mut b = dependencies[1].borrow_mut();

        let a_data = a.data.as_ref().unwrap();
        let b_data = b.data.as_ref().unwrap();

        println!(
            "Incoming: {:?} \nGrad A data: {:?}\n Grad B data: {:?}",
            grad, a_data, b_data
        );

        // Check if this is matrix multiplication (2D tensors)
        if Self::is_2d(a_data) && Self::is_2d(b_data) {
            // For matrix multiplication A @ B, gradients are:
            // grad_a = grad @ B.T
            // grad_b = A.T @ grad
            let a_2d = Self::to_2d(a_data);
            let b_2d = Self::to_2d(b_data);
            let grad_2d = Self::to_2d(grad);

            let grad_a = grad_2d.dot(&b_2d.t());
            let grad_b = a_2d.t().dot(&grad_2d);

            a.accumulate_grad(&grad_a.into_dyn());
            b.accumulate_grad(&grad_b.into_dyn());
        } else {
            // Element-wise multiplication
            let grad_a = grad * b_data;
            let grad_b = grad * a_data;

            a.accumulate_grad(&grad_a);
            b.accumulate_grad(&grad_b);
        }
    }
}

impl Mul<Tensor> for Tensor {
    type Output = Tensor;

    fn mul(self, other: Tensor) -> Self::Output {
        let requires_grad = self.value.borrow().requires_grad || other.value.borrow().requires_grad;

        let value = Rc::new(RefCell::new(Value::new(
            None,
            Box::new(MultOp),
            vec![self.value.clone(), other.value.clone()],
            requires_grad,
        )));

        Tensor { value }
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

#[cfg(test)]
mod tests {
    use ndarray::array;

    use super::*;

    #[test]
    fn test_mult_backprop() {
        let a: Tensor = array![[1.0, 2.0], [3.0, 4.0]].into();
        let b: Tensor = array![[5.0, 6.0], [7.0, 8.0]].into();

        let c = &a * &b;
        c.value.borrow_mut().grad = Some(array![1.0].into_dyn());

        c.eval();
        c.backward();

        assert_eq!(c.value.borrow().data.as_ref().unwrap().shape(), vec![2, 2]);
        assert_eq!(
            c.value.borrow().data,
            Some(array![[5.0, 12.0], [21.0, 32.0]].into_dyn())
        );

        assert_eq!(a.grad(), Some(array![[5.0, 6.0], [7.0, 8.0]].into_dyn()));
        assert_eq!(b.grad(), Some(array![[1.0, 2.0], [3.0, 4.0]].into_dyn()));
    }
}
