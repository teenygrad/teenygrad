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

use crate::error::{Error, Result};
use crate::nn::module::NodeRefModule;
use crate::{dtype, graph::NodeRef, nn::Module};

pub struct Sequential<N: dtype::Dtype> {
    layers: Vec<NodeRefModule<N, Error>>,
}

impl<N: dtype::Dtype> Sequential<N> {
    pub fn new(layers: Vec<NodeRefModule<N, Error>>) -> Self {
        Sequential { layers }
    }
}

impl<N: dtype::Dtype> Module<N, NodeRef<N>, NodeRef<N>> for Sequential<N> {
    type Err = Error;

    fn forward(&self, input: NodeRef<N>) -> Result<NodeRef<N>> {
        let mut output = input.clone();

        for layer in &self.layers {
            output = layer.forward(output)?;
        }

        Ok(output)
    }

    fn parameters(&self) -> Vec<NodeRef<N>> {
        self.layers
            .iter()
            .flat_map(|layer| layer.parameters())
            .collect()
    }
}

#[cfg(test)]
mod tests {

    use super::*;
    use crate::{
        graph::tensor,
        nn::{linear::*, relu::ReLU},
        sequential,
        tensor::shape::DynamicShape,
    };

    #[test]
    #[cfg(feature = "ndarray")]
    fn test_sequential_backprop() {
        use ndarray::Array1;

        let linear1 = Linear::new(1, 3, true).unwrap();
        let linear2 = Linear::new(3, 1, true).unwrap();

        let model = sequential![linear1, ReLU::new(), linear2];

        let input = tensor(Array1::from(vec![1.0f32]).into_dyn());
        let output = model.forward(input).unwrap();

        assert_eq!(output.shape().unwrap(), DynamicShape::new(&[1, 1]));
    }
}
