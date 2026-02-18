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

use ndarray::{Array, IxDyn};

use crate::dtype;
pub use crate::error::Result;

pub type SafeTensorsError = safetensors::SafeTensorError;
pub type Dtype = safetensors::tensor::Dtype;

#[derive(Debug, PartialEq, Eq, Clone)]
pub struct TensorView<'data>(pub safetensors::tensor::TensorView<'data>);

impl<'data> TensorView<'data> {
    pub fn dtype(&self) -> dtype::DtypeEnum {
        match self.0.dtype() {
            safetensors::tensor::Dtype::F32 => dtype::DtypeEnum::F32,
            safetensors::tensor::Dtype::BF16 => dtype::DtypeEnum::Bf16,
            _ => todo!(),
        }
    }

    pub fn into_array<N: dtype::Dtype>(self) -> Result<Array<N, IxDyn>> {
        let data = N::from_bytes(self.0.data());
        let shape = self.0.shape();
        let array = Array::from_shape_vec((shape[0], shape[1]), data)?;
        Ok(array.into_dyn())
    }
}

pub trait SafeTensors<'data>: Sized + Send + Sync {
    fn tensors(&'data self) -> Vec<(String, TensorView<'data>)>;
    fn iter(&self) -> impl Iterator<Item = (&str, TensorView<'data>)>;
    fn tensor(&'data self, name: &str) -> Result<TensorView<'data>>;
    fn names(&'data self) -> Vec<&'data str>;
    fn len(&'data self) -> usize;
    fn is_empty(&'data self) -> bool;
}
