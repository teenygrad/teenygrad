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

use core::marker::PhantomData;

use crate::{
    dtype::{Dtype, Tensor},
    nn::Node,
};

/// Fully-connected linear layer: `output = input × Wᵀ [+ b]`
///
/// Operates on the last dimension of an N-D tensor, passing all leading
/// dimensions through unchanged (e.g. a batch dimension).
///
/// Type parameters:
/// - `D`    — element dtype
/// - `IT`   — concrete input tensor type  (any rank, last dim = `in_features`)
/// - `OT`   — concrete output tensor type (same rank, last dim = `out_features`)
/// - `RANK` — tensor rank (1 = unbatched, 2 = batched, etc.)
pub struct Linear<D: Dtype, IT: Tensor<D, RANK>, OT: Tensor<D, RANK>, const RANK: usize> {
    pub in_features: usize,
    pub out_features: usize,
    pub has_bias: bool,
    _pd: PhantomData<(D, IT, OT)>,
}

impl<D: Dtype, IT: Tensor<D, RANK>, OT: Tensor<D, RANK>, const RANK: usize>
    Linear<D, IT, OT, RANK>
{
    pub fn new(in_features: usize, out_features: usize, has_bias: bool) -> Self {
        Self {
            in_features,
            out_features,
            has_bias,
            _pd: PhantomData,
        }
    }
}

impl<D: Dtype, IT: Tensor<D, RANK>, OT: Tensor<D, RANK>, const RANK: usize>
    Node<IT> for Linear<D, IT, OT, RANK>
{
    type Output = OT;

    fn call(&self, _input: IT) -> Self::Output {
        todo!()
    }
}
