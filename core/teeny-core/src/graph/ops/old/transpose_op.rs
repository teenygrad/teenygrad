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

use crate::tensorx::{Tensor, TensorData, Value, ValueRef, tensor_ops::TensorOp};

#[derive(Debug, Clone)]
pub struct TransposeOp;

impl TensorOp for TransposeOp {
    fn eval(&self, dependencies: &[ValueRef]) -> TensorData {
        assert_eq!(dependencies.len(), 1);
        dependencies[0].borrow_mut().eval();

        dependencies[0]
            .borrow()
            .data
            .as_ref()
            .unwrap()
            .clone()
            .t()
            .to_owned()
    }

    fn backward(&self, dependencies: &[ValueRef], grad: &TensorData) {
        if !dependencies.is_empty() && dependencies[0].borrow().requires_grad {
            // Transpose the gradient back to match the original tensor shape
            let transposed_grad = grad.t().to_owned();
            dependencies[0]
                .borrow_mut()
                .accumulate_grad(&transposed_grad);
        }
    }
}

impl Tensor {
    pub fn t(&self) -> Tensor {
        let requires_grad = self.value.borrow().requires_grad;

        let value = Rc::new(RefCell::new(Value::new(
            None,
            Box::new(TransposeOp),
            vec![self.value.clone()],
            requires_grad,
        )));

        Tensor { value }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nn::loss::Loss;
    use ndarray::array;

    #[test]
    fn test_transpose_backprop() {
        let a: Tensor = array![[1.0, 2.0], [3.0, 4.0]].into();

        let b = a.t();
        let mut loss = Loss::new(b.clone());
        loss.backward();

        // Check that the transposed tensor has the correct shape and values
        assert_eq!(b.value.borrow().data.as_ref().unwrap().shape(), vec![2, 2]);
        assert_eq!(
            b.value.borrow().data,
            Some(array![[1.0, 3.0], [2.0, 4.0]].into_dyn())
        );

        // Check that the gradient is correctly transposed back
        // The gradient should be [[1.0, 1.0], [1.0, 1.0]] transposed back to [[1.0, 1.0], [1.0, 1.0]]
        assert_eq!(a.grad(), Some(array![[1.0, 1.0], [1.0, 1.0]].into_dyn()));
    }
}
