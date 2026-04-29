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

pub mod activation;
pub mod batchnorm;
pub mod conv1d;
pub mod conv2d;
pub mod conv3d;
pub mod flatten;
pub mod groupnorm;
pub mod instancenorm;
pub mod layernorm;
pub mod linear;
pub mod pad;
pub mod pool;
pub mod rmsnorm;

pub trait Layer<I> {
    type Output;

    fn call(&self, input: I) -> Self::Output;
}

impl<F, I, O> Layer<I> for F
where
    F: Fn(I) -> O,
{
    type Output = O;

    fn call(&self, input: I) -> O {
        self(input)
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        dtype::{Dtype, EagerTensor, RankedTensor, Tensor},
        nn::{
            Layer,
            activation::{relu::Relu, softmax::Softmax},
            linear::Linear,
        },
        sequential,
    };

    /// Dummy tensor: satisfies `Tensor<D, RANK>` for any dtype and rank.
    /// Shape is all-zeros (dynamic/unknown). Used only to verify type composition.
    #[derive(Copy, Clone)]
    struct DummyTensor;

    impl<D: Dtype, const RANK: usize> RankedTensor<D, RANK> for DummyTensor {
        const SHAPE: [usize; RANK] = [0; RANK];
    }

    impl<D: Dtype, const RANK: usize> Tensor<D, RANK> for DummyTensor {}
    impl EagerTensor for DummyTensor {}

    /// A 2-layer MLP: Linear(784→128) → ReLU → Linear(128→10) → Softmax
    ///
    /// RANK=2 throughout so a batch dimension `[batch, features]` flows
    /// through every layer without changing rank.
    #[test]
    fn test_mlp_composition() {
        let _model = sequential![
            Linear::<f32, DummyTensor, DummyTensor, 2>::new(784, 128, true),
            Relu::<f32, DummyTensor, 2>::new(),
            Linear::<f32, DummyTensor, DummyTensor, 2>::new(128, 10, true),
            Softmax::<f32, DummyTensor, 2>::new(1)
        ];
    }
}
