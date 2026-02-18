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

use crate::dtype::DtypeEnum;
use crate::error::Result;
use crate::graph::shape::DynamicShape;

pub mod add;
pub mod and;
pub mod arange;
pub mod cat;
pub mod cos;
pub mod cumsum;
pub mod diff;
pub mod div;
pub mod dot;
pub mod eq;
pub mod exp;
pub mod expand;
pub mod index;
pub mod inverse;
pub mod isneginf;
pub mod leq;
pub mod log;
pub mod mean;
pub mod mult;
pub mod neg;
pub mod neq;
pub mod ones;
pub mod or;
pub mod pad;
pub mod pow;
pub mod powi;
pub mod randn;
pub mod relu;
pub mod rsqrt;
pub mod safetensor;
pub mod scalar;
pub mod sigmoid;
pub mod sin;
pub mod slice;
pub mod sqrt;
pub mod sub;
pub mod tensor;
pub mod to_dtype;
pub mod transpose;
pub mod unsqueeze;
pub mod vmap;
pub mod r#where;
pub mod zeros;

pub trait Op {
    fn shape(&self) -> Result<DynamicShape>;
    fn dtype(&self) -> DtypeEnum;
}

// use std::fmt::Debug;

// use crate::tensorx::{TensorData, ValueRef};
// pub trait TensorOp: Debug {
//     fn is_param(&self) -> bool {
//         false
//     }

//     fn eval(&self, dependencies: &[ValueRef]) -> TensorData;

//     fn backward(
//         &self,
//         dependencies: &[ValueRef],
//         grad: &ndarray::ArrayBase<ndarray::OwnedRepr<f32>, ndarray::IxDyn>,
//     );
// }
