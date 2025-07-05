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

use crate::tensor::{TensorData, ValueRef, tensor_ops::TensorOp};

#[derive(Debug, Clone)]
pub struct LogOp;

impl TensorOp for LogOp {
    fn eval(&self, dependencies: &[ValueRef]) -> TensorData {
        assert_eq!(dependencies.len(), 1);

        dependencies[0].borrow_mut().eval();
        dependencies[0].borrow().data.as_ref().unwrap().ln()
    }

    fn backward(&self, dependencies: &[ValueRef], grad: &TensorData) {
        assert_eq!(dependencies.len(), 1);
        let mut a = dependencies[0].borrow_mut();

        let grad_a = grad / a.data.clone().unwrap();
        a.accumulate_grad(&grad_a);
    }
}

#[cfg(test)]
mod tests {
    use ndarray::array;

    use crate::{
        nn::loss::Loss,
        tensor::{Tensor, log},
    };

    #[test]
    fn test_log_backprop() {
        let a: Tensor = array![[1.0, 2.0], [3.0, 4.0]].into();
        let b: Tensor = array![[5.0, 6.0], [7.0, 8.0]].into();

        let c = log(&a * &b);
        let mut loss = Loss::new(c.clone());
        loss.backward();

        assert_eq!(c.value.borrow().data.as_ref().unwrap().shape(), vec![2, 2]);
        assert_eq!(
            format!("{:?}", c.value.borrow().data),
            "Some([[1.609438, 2.4849067],\n [3.0445225, 3.465736]], shape=[2, 2], strides=[2, 1], layout=Cc (0x5), dynamic ndim=2)"
        );

        assert_eq!(
            format!("{:?}", a.grad()),
            "Some([[1.0, 0.5],\n [0.33333334, 0.25]], shape=[2, 2], strides=[2, 1], layout=Cc (0x5), dynamic ndim=2)"
        );

        assert_eq!(
            format!("{:?}", b.grad()),
            "Some([[0.2, 0.16666667],\n [0.14285715, 0.125]], shape=[2, 2], strides=[2, 1], layout=Cc (0x5), dynamic ndim=2)"
        );
    }
}
