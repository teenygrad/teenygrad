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

use crate::{
    dtype::DtypeEnum,
    error::Result,
    graph::{self, NodeRef},
    nn::Module,
    safetensors::SafeTensors,
    shape,
};

#[derive(Debug, Clone)]

pub struct Linear<'data> {
    pub name: String,
    pub weight: NodeRef<'data>,
    pub bias: Option<NodeRef<'data>>,
}

impl<'data> Linear<'data> {
    pub fn new(
        name: &str,
        dtype: DtypeEnum,
        input_dim: usize,
        output_dim: usize,
        use_bias: bool,
    ) -> Result<Self> {
        let weight = graph::randn(shape![output_dim, input_dim], dtype);

        let bias = if use_bias {
            Some(graph::zeros(shape![output_dim], dtype))
        } else {
            None
        };

        Ok(Linear {
            name: name.to_owned(),
            weight,
            bias,
        })
    }

    pub fn from_pretrained<T: SafeTensors<'data>>(
        name: &str,
        use_bias: bool,
        safetensors: &'data T,
    ) -> Result<Self> {
        let weight_name = format!("{name}.weight");
        let weight = graph::safetensor(safetensors.tensor(&weight_name)?);
        let bias = if use_bias {
            let bias_name = format!("{name}.bias");
            Some(graph::safetensor(safetensors.tensor(&bias_name)?))
        } else {
            None
        };

        Ok(Linear {
            name: name.to_owned(),
            weight,
            bias,
        })
    }
}

impl<'data> Module<'data, NodeRef<'data>, NodeRef<'data>> for Linear<'data> {
    fn forward(&self, x: NodeRef<'data>) -> Result<NodeRef<'data>> {
        let a = x * &self.weight.t();
        let result = if let Some(bias) = &self.bias {
            a + bias
        } else {
            a
        };

        Ok(result)
    }

    fn parameters(&self) -> Vec<NodeRef<'data>> {
        let mut params = vec![self.weight.clone()];
        if let Some(bias) = &self.bias {
            params.push(bias.clone());
        }

        params
    }
}

// impl<T: num::Num> Module1<T, &Tensor, Tensor> for Linear {
//     fn forward(&self, input: &Tensor) -> Tensor {
//         // Matrix multiplication
//         let output = input * &self.weight.t();

//         // Add bias if present
//         if let Some(b) = &self.bias {
//             bias(output, b.clone())
//         } else {
//             output
//         }
//     }

//     fn parameters(&self) -> Vec<TensorRef<T>> {
//         let mut params = vec![self.weight.clone()];
//         if let Some(bias) = &self.bias {
//             params.push(bias.clone());
//         }
//         todo!()
//         // params
//     }
// }

// #[cfg(test)]
// mod tests {
//     use ndarray::array;

//     use super::*;
//     use crate::{nn::loss::Loss, tensor::Tensor};

//     #[test]
//     fn test_linear_backprop() {
//         let input: Tensor = array![[1.0, 2.0], [3.0, 4.0]].into();
//         let linear = Linear::new(2, 3, true);

//         let output = linear.forward(&input);
//         let mut loss = Loss::new(output.clone());
//         loss.backward();

//         assert_eq!(
//             output.value.borrow().data.as_ref().unwrap().shape(),
//             vec![2, 3]
//         );

//         // Check that gradients are computed for weight and bias
//         assert_eq!(
//             format!("{:?}", linear.weight.grad()),
//             "Some([[0.0, 0.0],\n [0.0, 0.0],\n [0.0, 0.0]], shape=[3, 2], strides=[2, 1], layout=Cc (0x5), dynamic ndim=2)"
//         );
//         assert_eq!(
//             format!("{:?}", linear.bias.as_ref().unwrap().grad()),
//             "Some([0.0, 0.0, 0.0], shape=[3], strides=[1], layout=CFcf (0xf), dynamic ndim=1)"
//         );
//     }

//     #[test]
//     fn test_linear_no_bias() {
//         let input: Tensor = array![[1.0, 2.0], [3.0, 4.0]].into();
//         let linear = Linear::new(2, 3, false);

//         let output = linear.forward(&input);
//         let mut loss = Loss::new(output.clone());
//         loss.backward();

//         assert_eq!(
//             output.value.borrow().data.as_ref().unwrap().shape(),
//             vec![2, 3]
//         );

//         // Check that only weight has gradients (no bias)
//         assert_eq!(
//             format!("{:?}", linear.weight.grad()),
//             "Some([[0.0, 0.0],\n [0.0, 0.0],\n [0.0, 0.0]], shape=[3, 2], strides=[2, 1], layout=Cc (0x5), dynamic ndim=2)"
//         );
//     }
// }
