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

#[cfg(feature = "ndarray")]
use ndarray::IxDyn;

use teeny_core::dtype;

#[derive(Debug)]
pub struct CpuTensor<T: dtype::Dtype> {
    pub data: ndarray::Array<T, IxDyn>,
}

// impl<T: dtype::Dtype> CpuTensor<T> {
//     pub fn zeros<S: shape::Shape>(shape: S) -> Self {
//         Self {
//             data: ndarray::Array::zeros::<IxDyn>(shape.into()),
//         }
//     }
// }

impl<T: dtype::Dtype> From<ndarray::Array<T, IxDyn>> for CpuTensor<T> {
    fn from(data: ndarray::Array<T, IxDyn>) -> Self {
        Self { data }
    }
}
