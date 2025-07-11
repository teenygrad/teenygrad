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

pub struct Sequential<'a> {
    layers: Vec<Box<dyn Module<&'a Tensor, Tensor>>>,
}

impl<'a> Sequential<'a> {
    pub fn new(layers: Vec<Box<dyn Module<&'a Tensor, Tensor>>>) -> Self {
        Sequential { layers }
    }
}

impl<'a> Module<&'a Tensor, Tensor> for Sequential<'a> {
    fn forward(&self, input: &Tensor) -> Tensor {
        let mut _output = input.clone();

        todo!("Sequential::forward");
        // for layer in &self.layers {
        //     output = layer.forward(&output);
        // }

        //output
    }

    fn parameters(&self) -> Vec<Tensor> {
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
        nn::{ReLU, linear::*},
        sequential,
    };
    use ndarray::array;

    #[test]
    fn test_sequential_backprop() {
        let linear1 = Linear::new(2, 3, true);
        let linear2 = Linear::new(3, 1, true);

        let _model = sequential![linear1, ReLU::new(), linear2];

        let _input: Tensor = array![[1.0, 2.0], [3.0, 4.0]].into();
        todo!("test_sequential_backprop");
        // let output = model.forward(&input);
        // let mut loss = Loss::new(output.clone());

        // loss.backward();
        // // Check that output has the expected shape (2, 1)
        // assert_eq!(
        //     output.value.borrow().data.as_ref().unwrap().shape(),
        //     vec![2, 1]
        // );
    }
}
