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

use std::sync::Arc;

use crate::{dtype::Dtype, graph::Node, tensor::shape::Shape};

#[derive(Debug, Clone)]
pub struct AddOp<S: Shape, N: Dtype> {
    lhs: Arc<Node<S, N>>,
    rhs: Arc<Node<S, N>>,
}

impl<S: Shape, N: Dtype> AddOp<S, N> {
    pub fn new(lhs: &Arc<Node<S, N>>, rhs: &Arc<Node<S, N>>) -> Self {
        Self {
            lhs: lhs.clone(),
            rhs: rhs.clone(),
        }
    }
}

// use std::{cell::RefCell, ops::Add, rc::Rc};

// use crate::tensorx::{Tensor, TensorData, Value, ValueRef, tensor_ops::TensorOp};

// #[derive(Debug, Clone)]
// pub struct AddOp;

// impl TensorOp for AddOp {
//     fn eval(&self, dependencies: &[ValueRef]) -> TensorData {
//         dependencies.iter().for_each(|v| v.borrow_mut().eval());

//         dependencies
//             .iter()
//             .map(|v| v.borrow().data.clone().unwrap())
//             .reduce(|a, b| a + b)
//             .unwrap()
//     }

//     fn backward(&self, dependencies: &[ValueRef], grad: &TensorData) {
//         assert_eq!(dependencies.len(), 2);
//         let mut a = dependencies[0].borrow_mut();
//         let mut b = dependencies[1].borrow_mut();

//         a.accumulate_grad(grad);
//         b.accumulate_grad(grad);
//     }
// }

// impl Add<Tensor> for Tensor {
//     type Output = Tensor;

//     fn add(self, other: Tensor) -> Self::Output {
//         let a = self.value.clone();
//         let b = other.value.clone();
//         let requires_grad = a.borrow().requires_grad || b.borrow().requires_grad;

//         let value = Rc::new(RefCell::new(Value::new(
//             None,
//             Box::new(AddOp),
//             vec![a, b],
//             requires_grad,
//         )));

//         Tensor { value }
//     }
// }

// impl Add<&Tensor> for Tensor {
//     type Output = Tensor;

//     fn add(self, other: &Tensor) -> Self::Output {
//         self.add(other.clone())
//     }
// }

// impl Add<Tensor> for &Tensor {
//     type Output = Tensor;

//     fn add(self, other: Tensor) -> Self::Output {
//         self.clone().add(other)
//     }
// }

// impl Add<&Tensor> for &Tensor {
//     type Output = Tensor;

//     fn add(self, other: &Tensor) -> Self::Output {
//         self.clone().add(other.clone())
//     }
// }

// #[cfg(test)]
// mod tests {
//     use ndarray::array;

//     use crate::nn::loss::Loss;

//     use super::*;

//     #[test]
//     fn test_add_backprop() {
//         let a: Tensor = array![[1.0, 2.0], [3.0, 4.0]].into();
//         let b: Tensor = array![[5.0, 6.0], [7.0, 8.0]].into();

//         let c = &a + &b;
//         let mut loss = Loss::new(c.clone());
//         loss.backward();

//         assert_eq!(c.value.borrow().data.as_ref().unwrap().shape(), vec![2, 2]);

//         assert_eq!(
//             c.value.borrow().data,
//             Some(array![[6.0, 8.0], [10.0, 12.0]].into_dyn())
//         );

//         assert_eq!(a.grad(), Some(array![[1.0, 1.0], [1.0, 1.0]].into_dyn()));
//         assert_eq!(b.grad(), Some(array![[1.0, 1.0], [1.0, 1.0]].into_dyn()));
//     }
// }
