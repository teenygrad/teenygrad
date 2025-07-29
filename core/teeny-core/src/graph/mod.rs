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

use std::sync::Arc;

use crate::dtype::{Dtype, DtypeEnum, Value};
use crate::error::Result;
use crate::graph::ops::OpShape;
use crate::graph::ops::dot::DotOp;
use crate::graph::ops::inverse::InverseOp;
use crate::graph::ops::ones::OnesOp;
use crate::graph::ops::pow::PowOp;
use crate::graph::ops::powi::Powi;
use crate::graph::ops::safetensor::SafeTensorOp;
use crate::graph::ops::sqrt::SqrtOp;
use crate::graph::ops::tensor::TensorOpF32;
use crate::graph::ops::to_dtype::ToDtype;
use crate::graph::ops::transpose::TransposeOp;
use crate::graph::ops::unsqueeze::UnsqueezeOp;
use crate::safetensors::{SafeTensors, TensorView};
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
pub enum NodeOp<'data> {
    Scalar(ScalarOp),
    Add(AddOp<'data>),
    Sub(SubOp<'data>),
    Mult(MultOp<'data>),
    Div(DivOp<'data>),
    Dot(DotOp<'data>),
    Neg(NegOp<'data>),
    Log(LogOp<'data>),
    Exp(ExpOp<'data>),
    Mean(MeanOp<'data>),
    Zeros(ZerosOp),
    Arange(ArangeOp),
    Randn(RandnOp),
    Relu(ReluOp<'data>),
    Sigmoid(SigmoidOp<'data>),
    Transpose(TransposeOp<'data>),
    Powi(Powi<'data>),
    Sqrt(SqrtOp<'data>),
    Ones(OnesOp),
    Inverse(InverseOp<'data>),
    Pow(PowOp<'data>),
    Tensor(TensorOpF32),
    SafeTensor(SafeTensorOp<'data>),
    Unsqueeze(UnsqueezeOp<'data>),
    ToDtype(ToDtype<'data>),
}

impl<'data> NodeOp<'data> {
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
            NodeOp::Unsqueeze(op) => op.shape(),
            NodeOp::ToDtype(op) => op.shape(),
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
pub struct Node<'data> {
    pub id: UniqueId,
    pub op: NodeOp<'data>,

    #[cfg(feature = "training")]
    pub autograd_context: Option<AutogradContext>,
}

impl<'data> Node<'data> {
    pub fn new(op: NodeOp<'data>, requires_grad: bool, retain_grad: bool) -> Self {
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

pub fn zeros(shape: DynamicShape, dtype: DtypeEnum) -> NodeRef<'static> {
    ZerosOp::new(shape, dtype).into()
}

pub fn ones(shape: DynamicShape, dtype: DtypeEnum) -> NodeRef<'static> {
    OnesOp::new(shape, dtype).into()
}

pub fn randn(shape: DynamicShape, dtype: DtypeEnum) -> NodeRef<'static> {
    RandnOp::new(shape, dtype).into()
}

pub fn inverse<'data>(x: NodeRef<'data>) -> NodeRef<'data> {
    InverseOp::new(x).into()
}

pub fn exp<'data>(x: NodeRef<'data>) -> NodeRef<'data> {
    ExpOp::new(x).into()
}

pub fn arange<'data>(start: Value, end: Value, step: Value) -> NodeRef<'data> {
    ArangeOp::new(start, end, step).into()
}

pub fn unsqueeze<'data>(x: NodeRef<'data>, dim: usize) -> NodeRef<'data> {
    UnsqueezeOp::new(x, dim).into()
}

pub fn pow<'data>(x: NodeRef<'data>, y: NodeRef<'data>) -> NodeRef<'data> {
    PowOp::new(x, y).into()
}

#[cfg(feature = "ndarray")]
pub fn tensor_f32<'data>(input: ndarray::Array<f32, IxDyn>) -> NodeRef<'data> {
    TensorOpF32::new(input).into()
}

pub fn safetensor<'data>(input: TensorView<'data>) -> NodeRef<'data> {
    SafeTensorOp::new(input).into()
}

pub fn safetensor_with_name<'data, T: SafeTensors<'data>>(
    name: &str,
    safetensors: &'data T,
) -> Result<NodeRef<'data>> {
    let tensor = safetensors.tensor(name)?;
    Ok(SafeTensorOp::new(tensor).into())
}

pub fn log<'data>(x: NodeRef<'data>) -> NodeRef<'data> {
    LogOp::new(x.clone()).into()
}

pub fn transpose<'data>(x: &NodeRef<'data>) -> NodeRef<'data> {
    TransposeOp::new(x.clone()).into()
}

pub fn scalar(x: Value) -> NodeRef<'static> {
    ScalarOp::new(x).into()
}

pub fn relu<'data>(x: NodeRef<'data>) -> NodeRef<'data> {
    ReluOp::new(x).into()
}

pub fn sigmoid<'data>(x: NodeRef<'data>) -> NodeRef<'data> {
    SigmoidOp::new(x).into()
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
// pub struct AutogradContext<D: Device: dtype::Dtype> {
//     pub activation: ndarray::Array<N, IxDyn>,
//     pub grad: Option<ndarray::Array<N, IxDyn>>,
//     pub requires_grad: bool,
//     pub retain_grad: bool,
//     _marker: std::marker::PhantomData<D>,
// }

// /// A tensor in our computation graph
// #[derive(Debug, Clone)]
// pub struct GraphTensor<D: Device: dtype::Dtype> {
//     pub id: String,
//     pub operation: Box<dyn TensorOp>,
//     pub dependencies: Vec<Arc<GraphTensor<D>>>,
//     #[cfg(feature = "training")]
//     pub autograd_context: Option<AutogradContext<D>>,
//     _marker: std::marker::PhantomData<D>,
// }

// impl<D: Device: dtype::Dtype> Tensor<D> for GraphTensor<D> {
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

// impl<D: Device: dtype::Dtype> Add<GraphTensor<D>> for GraphTensor<D> {
//     type Output = GraphTensor<D>;

//     fn add(self, other: GraphTensor<D>) -> Self::Output {
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
