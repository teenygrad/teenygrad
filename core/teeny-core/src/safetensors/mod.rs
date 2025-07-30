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
