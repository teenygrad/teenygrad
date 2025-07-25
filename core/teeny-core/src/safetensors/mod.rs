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

pub use crate::error::Result;

pub type SafeTensorsError = safetensors::SafeTensorError;
pub type TensorView<'data> = safetensors::tensor::TensorView<'data>;
pub type Dtype = safetensors::tensor::Dtype;

pub trait SafeTensors<'data>: Sized + Send + Sync {
    fn tensors(&'data self) -> Vec<(String, TensorView<'data>)>;
    fn iter(&self) -> impl Iterator<Item = (&str, TensorView<'data>)>;
    fn tensor(&'data self, tensor_name: &'data str) -> Result<TensorView<'data>>;
    fn names(&'data self) -> Vec<&'data str>;
    fn len(&'data self) -> usize;
    fn is_empty(&'data self) -> bool;
}
