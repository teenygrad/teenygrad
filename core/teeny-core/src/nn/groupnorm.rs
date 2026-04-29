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
    dtype::{Dtype, EagerTensor, Tensor},
    nn::Layer,
};

/// Group normalisation over `num_groups` groups of channels.
///
/// Input shape: `[N, C, *]` — normalises over `(C/num_groups, *)` per sample.
/// Output shape: same as input.
///
/// Parameters:
/// - `num_groups`   — number of groups to divide C into (must divide C evenly)
/// - `num_channels` — number of channels C
/// - `eps`          — numerical stability constant (default 1e-5)
/// - `affine`       — if true, learns per-channel γ and β (default true)
pub struct GroupNorm<D: Dtype, IT, OT, const RANK: usize> {
    pub num_groups: usize,
    pub num_channels: usize,
    pub eps: f64,
    pub affine: bool,
    _pd: PhantomData<(D, IT, OT)>,
}

impl<D: Dtype, IT, OT, const RANK: usize> GroupNorm<D, IT, OT, RANK> {
    pub fn new(num_groups: usize, num_channels: usize) -> Self {
        Self {
            num_groups,
            num_channels,
            eps: 1e-5,
            affine: true,
            _pd: PhantomData,
        }
    }

    pub fn with_eps(mut self, eps: f64) -> Self {
        self.eps = eps;
        self
    }

    pub fn with_affine(mut self, affine: bool) -> Self {
        self.affine = affine;
        self
    }
}

impl<D: Dtype, IT: Tensor<D, RANK> + EagerTensor, OT: Tensor<D, RANK>, const RANK: usize>
    Layer<IT> for GroupNorm<D, IT, OT, RANK>
{
    type Output = OT;

    fn call(&self, _input: IT) -> Self::Output {
        todo!()
    }
}
