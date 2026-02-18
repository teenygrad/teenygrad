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

use std::ops::Add;

#[cfg(feature = "ndarray")]
use ndarray::IxDyn;
use teeny_core::dtype;

#[derive(Debug)]
pub struct CudaTensor<T: dtype::Dtype> {
    _marker: std::marker::PhantomData<T>,
}

impl<N: dtype::Dtype> Add<CudaTensor<N>> for CudaTensor<N> {
    type Output = CudaTensor<N>;

    fn add(self, _other: CudaTensor<N>) -> Self::Output {
        todo!()
    }
}

#[cfg(feature = "ndarray")]
impl<T: dtype::Dtype> From<ndarray::Array<T, IxDyn>> for CudaTensor<T> {
    fn from(_data: ndarray::Array<T, IxDyn>) -> Self {
        unimplemented!()
    }
}
