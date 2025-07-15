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

use std::ops::Add;

use crate::{device::Device, dtype, tensor::ndarray::NdarrayTensor};

impl<D: Device, N: dtype::Dtype> Add<NdarrayTensor<D, N>> for NdarrayTensor<D, N> {
    type Output = Self;

    fn add(self, _other: Self) -> Self::Output {
        NdarrayTensor {
            data: self.data + _other.data,
            #[cfg(feature = "training")]
            autograd_context: None,
            _marker: std::marker::PhantomData,
        }
    }
}
