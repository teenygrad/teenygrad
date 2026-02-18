/*
 * Copyright (c) 2026 Teenygrad.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

use std::{
    cell::RefCell,
    ops::{Mul, Neg, Sub},
    rc::Rc,
};

use ndarray::array;

use crate::tensorx::{Tensor, TensorData, Value, ValueRef, tensor_ops::TensorOp};

#[derive(Debug, Clone)]
pub struct SubOp;

impl TensorOp for SubOp {
    fn eval(&self, dependencies: &[ValueRef]) -> TensorData {
        assert_eq!(dependencies.len(), 2);
        dependencies[0].borrow_mut().eval();
        dependencies[1].borrow_mut().eval();

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

        Tensor { value }
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
        let neg_one: Tensor = array![-1.0].into();
        self.mul(neg_one)
    }
}

#[cfg(test)]
mod tests {
    use ndarray::array;

    use crate::{nn::loss::Loss, tensorx::Tensor};

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
