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
    nn::module::Module,
    tensor::{Tensor, tensor_ops::bias_op::bias},
};

pub struct Linear {
    weight: Tensor,
    bias: Option<Tensor>,
}

impl Linear {
    pub fn new(input_dim: usize, output_dim: usize, use_bias: bool) -> Self {
        // Initialize weight tensor with proper shape
        let weight: Tensor =
            Tensor::new_param(ndarray::Array::zeros((output_dim, input_dim)).into_dyn());

        // Initialize bias if needed
        let bias = if use_bias {
            let bias_shape = vec![output_dim];
            Some(Tensor::new_param(
                ndarray::Array::zeros(bias_shape).into_dyn(),
            ))
        } else {
            None
        };

        Linear { weight, bias }
    }
}

impl Module<&Tensor, Tensor> for Linear {
    fn forward(&self, input: &Tensor) -> Tensor {
        // Matrix multiplication
        let output = input * &self.weight.t();

        // Add bias if present
        if let Some(b) = &self.bias {
            bias(output, b.clone())
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

#[cfg(test)]
mod tests {
    use ndarray::array;

    use super::*;
    use crate::{nn::loss::Loss, tensor::Tensor};

    #[test]
    fn test_linear_backprop() {
        let input: Tensor = array![[1.0, 2.0], [3.0, 4.0]].into();
        let linear = Linear::new(2, 3, true);

        let output = linear.forward(&input);
        let mut loss = Loss::new(output.clone());
        loss.backward();

        assert_eq!(
            output.value.borrow().data.as_ref().unwrap().shape(),
            vec![2, 3]
        );

        // Check that gradients are computed for weight and bias
        assert_eq!(
            format!("{:?}", linear.weight.grad()),
            "Some([[0.0, 0.0],\n [0.0, 0.0],\n [0.0, 0.0]], shape=[3, 2], strides=[2, 1], layout=Cc (0x5), dynamic ndim=2)"
        );
        assert_eq!(
            format!("{:?}", linear.bias.as_ref().unwrap().grad()),
            "Some([0.0, 0.0, 0.0], shape=[3], strides=[1], layout=CFcf (0xf), dynamic ndim=1)"
        );
    }

    #[test]
    fn test_linear_no_bias() {
        let input: Tensor = array![[1.0, 2.0], [3.0, 4.0]].into();
        let linear = Linear::new(2, 3, false);

        let output = linear.forward(&input);
        let mut loss = Loss::new(output.clone());
        loss.backward();

        assert_eq!(
            output.value.borrow().data.as_ref().unwrap().shape(),
            vec![2, 3]
        );

        // Check that only weight has gradients (no bias)
        assert_eq!(
            format!("{:?}", linear.weight.grad()),
            "Some([[0.0, 0.0],\n [0.0, 0.0],\n [0.0, 0.0]], shape=[3, 2], strides=[2, 1], layout=Cc (0x5), dynamic ndim=2)"
        );
    }
}
