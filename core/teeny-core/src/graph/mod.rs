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

pub mod node_ref;
pub mod ops;

use crate::dtype::Dtype;
use crate::error::Result;
use crate::graph::ops::OpShape;
use crate::graph::ops::dot::DotOp;
use crate::graph::ops::inverse::InverseOp;
use crate::graph::ops::ones::OnesOp;
use crate::graph::ops::pow::PowOp;
use crate::graph::ops::powi::Powi;
use crate::graph::ops::safetensor::SafeTensorOp;
use crate::graph::ops::sqrt::SqrtOp;
use crate::graph::ops::tensor::TensorOp;
use crate::graph::ops::transpose::TransposeOp;
use crate::tensor::shape::DynamicShape;

use crate::util::unique_id::UniqueId;

#[cfg(feature = "ndarray")]
use ndarray::IxDyn;
pub use node_ref::NodeRef;

use ops::add::AddOp;
use ops::arange::ArangeOp;
use ops::div::DivOp;
use ops::exp::ExpOp;
use ops::log::LogOp;
use ops::mean::MeanOp;
use ops::mult::MultOp;
use ops::neg::NegOp;
use ops::randn::RandnOp;
use ops::relu::ReluOp;
use ops::scalar::ScalarOp;
use ops::sigmoid::SigmoidOp;
use ops::sub::SubOp;
use ops::zeros::ZerosOp;

#[derive(Debug, Clone)]
pub enum NodeOp<'data, N: Dtype> {
    Scalar(ScalarOp<N>),
    Add(AddOp<'data, N>),
    Sub(SubOp<'data, N>),
    Mult(MultOp<'data, N>),
    Div(DivOp<'data, N>),
    Dot(DotOp<'data, N>),
    Neg(NegOp<'data, N>),
    Log(LogOp<'data, N>),
    Exp(ExpOp<'data, N>),
    Mean(MeanOp<'data, N>),
    Zeros(ZerosOp<N>),
    Arange(ArangeOp<N>),
    Randn(RandnOp<N>),
    Relu(ReluOp<N>),
    Sigmoid(SigmoidOp<N>),
    Transpose(TransposeOp<N>),
    Powi(Powi<N>),
    Sqrt(SqrtOp<N>),
    Ones(OnesOp<N>),
    Inverse(InverseOp<N>),
    Pow(PowOp<N>),
    Tensor(TensorOp<N>),
    SafeTensor(SafeTensorOp<'data, N>),
}

impl<'data, N: Dtype> NodeOp<'data, N> {
    pub fn shape(&self) -> Result<DynamicShape> {
        match self {
            NodeOp::Scalar(op) => op.shape(),
            NodeOp::Add(op) => op.shape(),
            NodeOp::Sub(op) => op.shape(),
            NodeOp::Mult(op) => op.shape(),
            NodeOp::Div(op) => op.shape(),
            NodeOp::Neg(op) => op.shape(),
            NodeOp::Log(op) => op.shape(),
            NodeOp::Exp(op) => op.shape(),
            NodeOp::Mean(op) => op.shape(),
            NodeOp::Zeros(op) => op.shape(),
            NodeOp::Arange(op) => op.shape(),
            NodeOp::Randn(op) => op.shape(),
            NodeOp::Relu(op) => op.shape(),
            NodeOp::Sigmoid(op) => op.shape(),
            NodeOp::Tensor(op) => op.shape(),
            NodeOp::Transpose(op) => op.shape(),
            NodeOp::Powi(op) => op.shape(),
            NodeOp::Sqrt(op) => op.shape(),
            NodeOp::Dot(op) => op.shape(),
            NodeOp::Ones(op) => op.shape(),
            NodeOp::Inverse(op) => op.shape(),
            NodeOp::Pow(op) => op.shape(),
            NodeOp::SafeTensor(op) => op.shape(),
        }
    }
}

#[cfg(feature = "training")]
#[derive(Debug, Clone)]
pub struct AutogradContext {
    pub requires_grad: bool,
    pub retain_grad: bool,
}

#[derive(Debug, Clone)]
pub struct Node<'data, N: Dtype> {
    pub id: UniqueId,
    pub op: NodeOp<'data, N>,

    #[cfg(feature = "training")]
    pub autograd_context: Option<AutogradContext>,
}

impl<'data, N: Dtype> Node<'data, N> {
    pub fn new(op: NodeOp<'data, N>, requires_grad: bool, retain_grad: bool) -> Self {
        Self {
            id: UniqueId::generate(),
            op,
            #[cfg(feature = "training")]
            autograd_context: Some(AutogradContext {
                requires_grad,
                retain_grad,
            }),
        }
    }

    pub fn shape(&self) -> Result<DynamicShape> {
        self.op.shape()
    }
}

pub fn zeros<N: Dtype>(shape: DynamicShape) -> NodeRef<'static, N> {
    ZerosOp::new(shape).into()
}

pub fn ones<N: Dtype>(shape: DynamicShape) -> NodeRef<'static, N> {
    ZerosOp::new(shape).into()
}

pub fn randn<N: Dtype>(shape: DynamicShape) -> NodeRef<'static, N> {
    RandnOp::new(shape).into()
}

pub fn inverse<'data, N: Dtype>(x: NodeRef<'data, N>) -> NodeRef<'data, N> {
    InverseOp::new(x).into()
}

pub fn exp<'data, N: Dtype>(x: NodeRef<'data, N>) -> NodeRef<'data, N> {
    ExpOp::new(x).into()
}

pub fn arange<N: Dtype>(start: N, end: N, step: N) -> NodeRef<'static, N> {
    ArangeOp::new(start, end, step).into()
}

pub fn pow<'data, N: Dtype>(x: NodeRef<'data, N>, y: NodeRef<'data, N>) -> NodeRef<'data, N> {
    PowOp::new(x, y).into()
}

#[cfg(feature = "ndarray")]
pub fn tensor<N: Dtype>(input: ndarray::Array<N, IxDyn>) -> NodeRef<'static, N> {
    TensorOp::new(input).into()
}

pub fn log<'data, N: Dtype>(x: NodeRef<'data, N>) -> NodeRef<'data, N> {
    LogOp::new(x.clone()).into()
}

pub fn transpose<'data, N: Dtype>(x: &NodeRef<'data, N>) -> NodeRef<'data, N> {
    TransposeOp::new(x.clone()).into()
}

pub fn scalar<N: Dtype>(x: N) -> NodeRef<'static, N> {
    ScalarOp::new(x).into()
}

pub fn relu<'data, N: Dtype>(x: NodeRef<'data, N>) -> NodeRef<'data, N> {
    ReluOp::new(x.clone()).into()
}

pub fn sigmoid<'data, N: Dtype>(x: NodeRef<'data, N>) -> NodeRef<'data, N> {
    SigmoidOp::new(x.clone()).into()
}
// use std::ops::Add;
// use std::sync::Arc;

// #[cfg(feature = "training")]
// use ndarray::IxDyn;

// use crate::device::Device;
// use crate::dtype;
// use crate::error::Result;
// use crate::tensor::graph::tensor_ops::TensorOp;
// use crate::tensor::{Tensor, shape};

// pub mod tensor_ops;

// #[cfg(feature = "training")]
// #[derive(Debug)]
// pub struct AutogradContext<D: Device, N: dtype::Dtype> {
//     pub activation: ndarray::Array<N, IxDyn>,
//     pub grad: Option<ndarray::Array<N, IxDyn>>,
//     pub requires_grad: bool,
//     pub retain_grad: bool,
//     _marker: std::marker::PhantomData<D>,
// }

// /// A tensor in our computation graph
// #[derive(Debug, Clone)]
// pub struct GraphTensor<D: Device, N: dtype::Dtype> {
//     pub id: String,
//     pub operation: Box<dyn TensorOp>,
//     pub dependencies: Vec<Arc<GraphTensor<D, N>>>,
//     #[cfg(feature = "training")]
//     pub autograd_context: Option<AutogradContext<D, N>>,
//     _marker: std::marker::PhantomData<D>,
// }

// impl<D: Device, N: dtype::Dtype> Tensor<D, N> for GraphTensor<D, N> {
//     fn zeros<S: shape::Shape>(shape: S) -> Result<Self> {
//         todo!()
//     }

//     fn randn<S: shape::Shape>(shape: S) -> Result<Self> {
//         todo!()
//     }

//     fn arange(start: N, end: N, step: N) -> Result<Self> {
//         todo!()
//     }
// }

// impl<D: Device, N: dtype::Dtype> Add<GraphTensor<D, N>> for GraphTensor<D, N> {
//     type Output = GraphTensor<D, N>;

//     fn add(self, other: GraphTensor<D, N>) -> Self::Output {
//         todo!()
//     }
// }

// impl GraphTensor {
//     /// Zero all gradients in the tensor
//     pub fn zero_grad(&self) {
//         self.value.borrow_mut().zero_grad();
//     }

//     pub fn eval(&self) -> TensorData {
//         self.value.borrow_mut().eval();
//         self.value.borrow().data.clone().unwrap()
//     }

//     /// Backward pass through the entire tensor
//     pub fn backward(&self) {
//         let value = self.value.borrow_mut();
//         value.backward();
//     }

//     /// Get gradients for all values in the tensor
//     pub fn grad(&self) -> Option<TensorData> {
//         self.value.borrow().grad.clone()
//     }

//     /// Update values using gradients (for optimization)
//     pub fn update(&mut self, learning_rate: f32) {
//         let grad = self.value.borrow().grad.as_ref().unwrap().clone();

//         if let Some(ref mut data) = self.value.borrow_mut().data {
//             *data = learning_rate * grad;
//         }
//     }
// }

// pub fn log(x: GraphTensor) -> GraphTensor {
//     let requires_grad = x.value.borrow().requires_grad;

//     let value = Rc::new(RefCell::new(Value::new(
//         None,
//         Box::new(LogOp),
//         vec![x.value.clone()],
//         requires_grad,
//     )));

//     GraphTensor { value }
// }

// #[cfg(test)]
// mod tests {
//     use ndarray::array;

//     use super::*;

//     #[test]
//     fn test_autodiff_basic() {
//         // Create input tensors
//         let x: GraphTensor = array![[2.0, 3.0], [4.0, 5.0]].into();
//         let y: GraphTensor = array![[1.0, 2.0], [3.0, 4.0]].into();

//         // Create computation graph: z = (x + y) * 2 + relu(x)
//         let z1 = x + y; // x + y
//         let z2 = z1.relu(); // relu(x + y)
//         let z3 = z2.mean(); // mean(relu(x + y))

//         // zero gradients
//         z3.zero_grad();

//         // Backward pass
//         z3.backward();
//     }

//     #[test]
//     fn test_autodiff_optimization() {
//         // Simple optimization example: minimize f(x) = x^2 + 2x + 1
//         let x_shape = vec![1];
//         let mut x = GraphTensor::new(ndarray::Array::zeros(x_shape), true);
//         x.value.borrow_mut().data = Some(ndarray::Array::from_elem(vec![1], 3.0)); // Start at x = 3

//         let learning_rate = 0.1;

//         for _ in 0..10 {
//             // Zero gradients
//             x.zero_grad();

//             // Forward pass: f(x) = x^2 + 2x + 1
//             let x_squared = &x * &x; // x^2
//             let two_x = 2.0 * &x; // 2x
//             let x_sq_plus_2x = x_squared + two_x; // x^2 + 2x
//             let one_shape = vec![1];
//             let one = GraphTensor::new(ndarray::Array::zeros(one_shape), true);
//             one.value.borrow_mut().data = Some(ndarray::Array::from_elem(vec![1], 1.0));
//             let loss = x_sq_plus_2x + one; // x^2 + 2x + 1

//             // Backward pass
//             loss.backward();

//             // Update parameters
//             x.update(learning_rate);
//         }

//         // After optimization, x should be close to -1 (the minimum of f(x) = x^2 + 2x + 1)
//         let final_x = x.value.borrow().data.as_ref().unwrap().clone();
//         assert!(final_x.iter().any(|&v| (v - (-1.0)).abs() < 0.1));
//     }
// }
