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

/// Layer normalisation over the last `normalized_shape.len()` dimensions.
///
/// Input shape:  `[*, normalized_shape[0], ..., normalized_shape[n-1]]`
/// Output shape: same as input.
///
/// Parameters:
/// - `normalized_shape` — the shape of the axes to normalise over (e.g. `[C]`
///   for 2-D input `[N, C]`, or `[H, W]` for spatial normalisation)
/// - `eps`     — numerical stability constant (default 1e-5)
/// - `affine`  — if true, learns per-element γ and β (default true)
pub struct LayerNorm<D: Dtype, IT, OT, const RANK: usize> {
    pub normalized_shape: alloc::vec::Vec<usize>,
    pub eps: f64,
    pub affine: bool,
    _pd: PhantomData<(D, IT, OT)>,
}

impl<D: Dtype, IT, OT, const RANK: usize> LayerNorm<D, IT, OT, RANK> {
    pub fn new(normalized_shape: impl Into<alloc::vec::Vec<usize>>) -> Self {
        Self {
            normalized_shape: normalized_shape.into(),
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
    Layer<IT> for LayerNorm<D, IT, OT, RANK>
{
    type Output = OT;

    fn call(&self, _input: IT) -> Self::Output {
        todo!()
    }
}
