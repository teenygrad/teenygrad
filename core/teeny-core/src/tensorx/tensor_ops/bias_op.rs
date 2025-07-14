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

use std::{cell::RefCell, rc::Rc};

use crate::tensorx::{Tensor, TensorData, Value, ValueRef, tensor_ops::TensorOp};

#[derive(Debug, Clone)]
pub struct BiasOp;

impl TensorOp for BiasOp {
    fn eval(&self, dependencies: &[ValueRef]) -> TensorData {
        assert_eq!(dependencies.len(), 2);
        let mut tensor = dependencies[0].borrow_mut();
        let mut bias = dependencies[1].borrow_mut();

        tensor.eval();
        bias.eval();

        let tensor_data = tensor.data.as_ref().unwrap();
        let bias_data = bias.data.as_ref().unwrap();

        // Add bias to tensor (broadcasting bias across batch dimension)
        tensor_data + bias_data
    }

    fn backward(&self, dependencies: &[ValueRef], grad: &TensorData) {
        assert_eq!(dependencies.len(), 2);
        let mut tensor = dependencies[0].borrow_mut();
        let mut bias = dependencies[1].borrow_mut();

        // For tensor: gradient passes through unchanged
        tensor.accumulate_grad(grad);

        // For bias: sum gradients across batch dimension (dimension 0)
        if bias.requires_grad {
            let bias_grad = grad.sum_axis(ndarray::Axis(0));
            bias.accumulate_grad(&bias_grad);
        }
    }
}

pub fn bias(tensor: Tensor, bias: Tensor) -> Tensor {
    let requires_grad = tensor.value.borrow().requires_grad || bias.value.borrow().requires_grad;

    let value = Rc::new(RefCell::new(Value::new(
        None,
        Box::new(BiasOp),
        vec![tensor.value.clone(), bias.value.clone()],
        requires_grad,
    )));

    Tensor { value }
}

#[cfg(test)]
mod tests {
    use ndarray::array;

    use super::*;
    use crate::nn::loss::Loss;

    #[test]
    fn test_bias_forward() {
        let w: Tensor = array![[1.0, 2.0], [3.0, 4.0]].into();
        let b: Tensor = array![0.5, 1.0].into();

        let result = bias(w, b);
        result.eval();

        // Expected: [[1.0+0.5, 2.0+1.0], [3.0+0.5, 4.0+1.0]] = [[1.5, 3.0], [3.5, 5.0]]
        assert_eq!(
            result.value.borrow().data,
            Some(array![[1.5, 3.0], [3.5, 5.0]].into_dyn())
        );
    }

    #[test]
    fn test_bias_backprop() {
        let w: Tensor = array![[1.0, 2.0], [3.0, 4.0]].into();
        let b: Tensor = array![0.5, 1.0].into();

        let result = bias(w.clone(), b.clone());
        let mut loss = Loss::new(result);
        loss.backward();

        // For tensor: gradient should be 1.0 for each element
        assert_eq!(w.grad(), Some(array![[1.0, 1.0], [1.0, 1.0]].into_dyn()));

        // For bias: gradient should be sum across batch dimension
        // Sum of [1.0, 1.0] and [1.0, 1.0] = [2.0, 2.0]
        assert_eq!(b.grad(), Some(array![2.0, 2.0].into_dyn()));
    }

    #[test]
    fn test_bias_large_batch() {
        let w: Tensor = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]].into();
        let b: Tensor = array![0.5, 1.0].into();

        let result = bias(w.clone(), b.clone());
        let mut loss = Loss::new(result);
        loss.backward();

        // For bias: gradient should be sum across batch dimension (3 samples)
        // Sum of [1.0, 1.0], [1.0, 1.0], [1.0, 1.0] = [3.0, 3.0]
        assert_eq!(b.grad(), Some(array![3.0, 3.0].into_dyn()));
    }
}
