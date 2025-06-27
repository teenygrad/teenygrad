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

use std::ops::{Add, Mul};

use crate::{
    nn::module::ForwardModule,
    tensor::{self, Shape, Tensor},
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

impl<S: Shape, T: NumericType>
    ForwardModule<Box<dyn Tensor<S, Element = T>>, Box<dyn Tensor<S, Element = T>>>
    for Linear<S, T>
{
    fn forward(&self, input: Box<dyn Tensor<S, Element = T>>) -> Box<dyn Tensor<S, Element = T>> {
        // Perform matrix multiplication: input @ weight
        // For now, we'll use element-wise multiplication as a placeholder
        // In a real implementation, you'd need proper matrix multiplication
        let output = input * self.weight.clone();

        // Add bias if present
        if let Some(ref bias) = self.bias {
            output + bias.clone()
        } else {
            output
        }
    }
}
