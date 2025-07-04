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

use std::{
    cell::RefCell,
    ops::{Mul, Neg, Sub},
    rc::Rc,
};

use crate::tensor::{Tensor, TensorData, Value, ValueRef, tensor_ops::TensorOp};

#[derive(Debug, Clone)]
pub struct SubOp;

impl TensorOp for SubOp {
    fn eval(&self, dependencies: &[ValueRef]) -> TensorData {
        assert_eq!(dependencies.len(), 2);
        dependencies[0].borrow().data.as_ref().unwrap()
            - dependencies[1].borrow().data.as_ref().unwrap()
    }

    fn backward(&self, dependencies: &[ValueRef], grad: &TensorData) {
        assert_eq!(dependencies.len(), 2);
        let mut a = dependencies[0].borrow_mut();
        let mut b = dependencies[1].borrow_mut();

        a.accumulate_grad(grad);
        b.accumulate_grad(&-grad);
    }
}

impl Sub<Tensor> for Tensor {
    type Output = Tensor;

    fn sub(self, other: Tensor) -> Self::Output {
        let requires_grad = self.value.borrow().requires_grad || other.value.borrow().requires_grad;

        let value = Rc::new(RefCell::new(Value::new(
            None,
            Box::new(SubOp),
            vec![self.value.clone(), other.value.clone()],
            requires_grad,
        )));

        Tensor {
            value,
            shape: self.shape.clone(),
        }
    }
}

impl Sub<&Tensor> for Tensor {
    type Output = Tensor;

    fn sub(self, other: &Tensor) -> Self::Output {
        self.sub(other.clone())
    }
}

impl Sub<Tensor> for &Tensor {
    type Output = Tensor;

    fn sub(self, other: Tensor) -> Self::Output {
        self.clone().sub(other)
    }
}

impl Sub<&Tensor> for &Tensor {
    type Output = Tensor;

    fn sub(self, other: &Tensor) -> Self::Output {
        self.clone().sub(other.clone())
    }
}

impl Sub<&Tensor> for f32 {
    type Output = Tensor;

    fn sub(self, other: &Tensor) -> Self::Output {
        other
            .clone()
            .sub(Tensor::new(ndarray::Array::from_elem(vec![1], self), true))
    }
}

impl Sub<Tensor> for f32 {
    type Output = Tensor;

    fn sub(self, other: Tensor) -> Self::Output {
        other
            .clone()
            .sub(Tensor::new(ndarray::Array::from_elem(vec![1], self), true))
    }
}

impl Neg for Tensor {
    type Output = Tensor;

    fn neg(self) -> Self::Output {
        self.mul(Tensor::new(ndarray::Array::from_elem(vec![1], -1.0), true))
    }
}

#[cfg(test)]
mod tests {
    use ndarray::array;

    use crate::{nn::loss::Loss, tensor::Tensor};

    #[test]
    fn test_sub_backprop() {
        let a: Tensor = array![[1.0, 2.0], [3.0, 4.0]].into();
        let b: Tensor = array![[5.0, 6.0], [7.0, 8.0]].into();

        let c = &a - &b;
        let mut loss = Loss::new(c.clone());
        loss.backward();

        assert_eq!(
            c.value.borrow().data,
            Some(array![[-4.0, -4.0], [-4.0, -4.0]].into_dyn())
        );

        assert_eq!(a.grad(), Some(array![[1.0, 1.0], [1.0, 1.0]].into_dyn()));
        assert_eq!(
            b.grad(),
            Some(array![[-1.0, -1.0], [-1.0, -1.0]].into_dyn())
        );
    }
}
