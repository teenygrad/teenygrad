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

// use crate::{nn::module::Module1, tensor::Tensor};

use crate::error::Result;
use crate::nn::module::NodeRefModule;
use crate::{dtype, graph::NodeRef, nn::Module};

pub struct Sequential<'data, N: dtype::Dtype> {
    layers: Vec<NodeRefModule<'data, N>>,
}

impl<'data, N: dtype::Dtype> Sequential<'data, N> {
    pub fn new(layers: Vec<NodeRefModule<'data, N>>) -> Self {
        Sequential { layers }
    }
}

impl<'data, N: dtype::Dtype> Module<'data, N, NodeRef<'data, N>, NodeRef<'data, N>>
    for Sequential<'data, N>
{
    fn forward(&self, input: NodeRef<'data, N>) -> Result<NodeRef<'data, N>> {
        let mut output = input.clone();

        for layer in &self.layers {
            output = layer.forward(output)?;
        }

        Ok(output)
    }

    fn parameters(&self) -> Vec<NodeRef<'data, N>> {
        self.layers
            .iter()
            .flat_map(|layer| layer.parameters())
            .collect()
    }
}

#[cfg(test)]
mod tests {

    // use super::*;
    // use crate::{
    //     graph::tensor,
    //     nn::{linear::*, relu::relu},
    //     sequential,
    //     tensor::shape::DynamicShape,
    // };

    #[test]
    #[cfg(feature = "ndarray")]
    fn test_sequential_backprop() {
        //use ndarray::Array1;

        // let linear1 = Linear::new(1, 3, true).unwrap();
        // let linear2 = Linear::new(3, 1, true).unwrap();

        // let model = sequential![linear1, ReLU::new(), linear2];

        // let input = tensor(Array1::from(vec![1.0f32]).into_dyn());
        // let output = model.forward(input).unwrap();

        // assert_eq!(output.shape().unwrap(), DynamicShape::new(&[1, 1]));
        todo!()
    }
}
