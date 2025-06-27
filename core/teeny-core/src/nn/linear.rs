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

use std::ops::Mul;

use crate::{
    nn::module::ForwardModule,
    tensor::{self, Add, Shape, Tensor},
    types::NumericType,
};

pub struct Linear<S: Shape, T: NumericType> {
    pub weight: Box<dyn Tensor<S, Element = T>>,
    pub bias: Option<Box<dyn Tensor<S, Element = T>>>,
}

impl<S: Shape, T: NumericType> Linear<S, T> {
    pub fn new(in_features: usize, out_features: usize, bias: bool) -> Self {
        Self {
            weight: tensor::zeros::<S, T>(&[in_features, out_features]),
            bias: if bias {
                Some(tensor::zeros::<S, T>(&[out_features]))
            } else {
                None
            },
        }
    }
}

impl<'a, S: Shape + 'static, T: NumericType + 'static, U, V> ForwardModule<U, V> for Linear<S, T>
where
    U: Tensor<S, Element = T>
        + Mul<&'a dyn Tensor<S, Element = T>, Output = Box<dyn Tensor<S, Element = T>>>
        + Add<&'a dyn Tensor<S, Element = T>, Output = Box<dyn Tensor<S, Element = T>>>,
    V: Tensor<S, Element = T>
        + Add<&'a dyn Tensor<S, Element = T>, Output = Box<dyn Tensor<S, Element = T>>>,
{
    fn forward(&self, input: U) -> V {
        let output: Box<dyn Tensor<S, Element = T>> = input * self.weight.as_ref();
        if let Some(bias) = self.bias {
            bias.as_ref() + output.as_ref()
        } else {
            output
        }
    }
}
