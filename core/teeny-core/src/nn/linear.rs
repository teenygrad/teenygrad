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

use crate::{nn::module::Module, tensor::Tensor};

pub struct Linear {
    weight: Tensor,
    bias: Option<Tensor>,
}

impl Linear {
    pub fn new(input_dim: usize, output_dim: usize, use_bias: bool) -> Self {
        // Initialize weight tensor with proper shape
        let weight = Tensor::new(vec![output_dim, input_dim]);

        // Initialize bias if needed
        let bias = if use_bias {
            Some(Tensor::new(vec![output_dim]))
        } else {
            None
        };

        Linear { weight, bias }
    }
}

impl Module for Linear {
    fn forward(&self, input: &Tensor) -> Tensor {
        // Matrix multiplication
        let output = input.mult(&self.weight.transpose());

        // Add bias if present
        if let Some(bias) = &self.bias {
            output.add(bias)
        } else {
            output
        }
    }

    fn parameters(&self) -> Vec<Tensor> {
        let mut params = vec![self.weight.clone()];
        if let Some(bias) = &self.bias {
            params.push(bias.clone());
        }
        params
    }
}
