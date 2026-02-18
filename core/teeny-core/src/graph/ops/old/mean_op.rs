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

use std::{cell::RefCell, rc::Rc};

use ndarray::array;

use crate::tensorx::{Tensor, TensorData, Value, ValueRef, tensor_ops::TensorOp};

#[derive(Debug, Clone)]
pub struct MeanOp;

impl TensorOp for MeanOp {
    fn eval(&self, dependencies: &[ValueRef]) -> TensorData {
        assert_eq!(dependencies.len(), 1);
        let mut a = dependencies[0].borrow_mut();
        a.eval();
        let data = a.data.as_ref().unwrap();
        let mean = data.mean();
        // Create a 0-dimensional tensor (scalar) with the mean value
        array![mean.unwrap()].into_dyn()
    }

    fn backward(&self, dependencies: &[ValueRef], grad: &TensorData) {
        assert_eq!(dependencies.len(), 1);
        let mut a = dependencies[0].borrow_mut();

        if a.requires_grad {
            let data = a.data.as_ref().unwrap();
            let n = data.len() as f32;
            let grad_per_element = grad / n;

            a.accumulate_grad(&grad_per_element);
        }
    }
}

impl Tensor {
    pub fn mean(&self) -> Tensor {
        let requires_grad = self.value.borrow().requires_grad;

        let value = Rc::new(RefCell::new(Value::new(
            None,
            Box::new(MeanOp),
            vec![self.value.clone()],
            requires_grad,
        )));

        Tensor { value }
    }
}

pub fn mean(a: &Tensor) -> Tensor {
    a.mean()
}

#[cfg(test)]
mod tests {
    use ndarray::array;

    use super::*;
    use crate::{nn::loss::Loss, tensorx::Tensor};

    #[test]
    fn test_mean_backprop() {
        // Test with a simple 2x2 tensor
        let a: Tensor = array![[1.0, 2.0], [3.0, 4.0]].into();

        let mean_tensor = mean(&a);
        let mut loss = Loss::new(mean_tensor.clone());
        loss.backward();

        // The mean of [1, 2, 3, 4] should be 2.5
        assert_eq!(
            mean_tensor.value.borrow().data,
            Some(array![2.5].into_dyn())
        );

        // The gradient should be distributed equally to all elements
        // Since we're computing loss of the mean, the gradient is 1.0
        // Each element gets 1.0 / 4 = 0.25
        assert_eq!(
            a.grad(),
            Some(array![[0.25, 0.25], [0.25, 0.25]].into_dyn())
        );
    }

    #[test]
    fn test_mean_single_element() {
        // Test with a single element tensor
        let a: Tensor = array![5.0].into();

        let mean_tensor = mean(&a);
        mean_tensor.eval();

        // The mean of a single element should be the element itself
        assert_eq!(
            mean_tensor.value.borrow().data,
            Some(array![5.0].into_dyn())
        );
    }

    #[test]
    fn test_mean_large_tensor() {
        // Test with a larger tensor
        let a: Tensor = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]].into();

        let mean_tensor = mean(&a);
        mean_tensor.eval();

        // The mean of [1,2,3,4,5,6,7,8,9] should be 5.0
        assert_eq!(
            mean_tensor.value.borrow().data,
            Some(array![5.0].into_dyn())
        );
    }

    #[test]
    fn test_mean_with_zeros() {
        // Test with tensor containing zeros
        let a: Tensor = array![[0.0, 0.0], [1.0, 1.0]].into();

        let mean_tensor = mean(&a);
        mean_tensor.eval();

        // The mean of [0, 0, 1, 1] should be 0.5
        assert_eq!(
            mean_tensor.value.borrow().data,
            Some(array![0.5].into_dyn())
        );
    }

    #[test]
    fn test_mean_gradient_distribution() {
        // Test that gradients are distributed correctly
        let a: Tensor = array![[1.0, 2.0], [3.0, 4.0]].into();

        let mean_tensor = mean(&a);
        let mut loss = Loss::new(mean_tensor);
        loss.backward();

        // Each element should receive equal gradient
        let expected_grad = 1.0 / 4.0; // 1.0 (loss gradient) / 4 elements
        assert_eq!(
            a.grad(),
            Some(
                array![
                    [expected_grad, expected_grad],
                    [expected_grad, expected_grad]
                ]
                .into_dyn()
            )
        );
    }
}
