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

// use crate::{nn::module::Module1, tensor::Tensor};

use crate::error::Result;
use crate::nn::module::NodeRefModule;
use crate::{graph::NodeRef, nn::Module};

pub struct Sequential<'data> {
    layers: Vec<NodeRefModule<'data>>,
}

impl<'data> Sequential<'data> {
    pub fn new(layers: Vec<NodeRefModule<'data>>) -> Self {
        Sequential { layers }
    }
}

impl<'data> Module<'data, NodeRef<'data>, NodeRef<'data>> for Sequential<'data> {
    fn forward(&mut self, input: NodeRef<'data>) -> Result<NodeRef<'data>> {
        let mut output = input.clone();

        for layer in &mut self.layers {
            output = layer.forward(output)?;
        }

        Ok(output)
    }

    fn parameters(&self) -> Vec<NodeRef<'data>> {
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
