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

/// Flatten layer: collapses all spatial dimensions into a single feature vector.
///
/// Input shape:  `[N, C, H, W]` (rank 4)
/// Output shape: `[N, C * H * W]` (rank 2)
///
/// Type parameters:
/// - `D`  — element dtype
/// - `IT` — input tensor type  (rank 4: `[N, C, H, W]`)
/// - `OT` — output tensor type (rank 2: `[N, C*H*W]`)
///
/// Tensor bounds are on impls, not the struct, so `SymTensor` can have its
/// own `Layer` impl without a coherence conflict.
pub struct Flatten<D: Dtype, IT, OT> {
    _pd: PhantomData<(D, IT, OT)>,
}

impl<D: Dtype, IT, OT> Flatten<D, IT, OT> {
    pub fn new() -> Self {
        Self { _pd: PhantomData }
    }
}

impl<D: Dtype, IT, OT> Default for Flatten<D, IT, OT> {
    fn default() -> Self {
        Self::new()
    }
}

impl<D: Dtype, IT: Tensor<D, 4> + EagerTensor, OT: Tensor<D, 2>> Layer<IT>
    for Flatten<D, IT, OT>
{
    type Output = OT;

    fn call(&self, _input: IT) -> Self::Output {
        todo!()
    }
}
