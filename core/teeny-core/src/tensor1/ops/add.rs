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
use std::sync::Arc;

use crate::tensor1::{DTensor, num};

use crate::tensor1::{Tensor, TensorRef};

impl<T: num::Num> Add<TensorRef<T>> for TensorRef<T> {
    type Output = TensorRef<T>;

    fn add(self, other: TensorRef<T>) -> Self::Output {
        TensorRef(Tensor::add(&*self.0, &*other.0))
    }
}

impl<T: num::Num> Add<&TensorRef<T>> for TensorRef<T> {
    type Output = TensorRef<T>;

    fn add(self, other: &TensorRef<T>) -> Self::Output {
        TensorRef(Tensor::add(&*self.0, &*other.0))
    }
}

impl<T: num::Num> Add<TensorRef<T>> for &TensorRef<T> {
    type Output = TensorRef<T>;

    fn add(self, other: TensorRef<T>) -> Self::Output {
        TensorRef(Tensor::add(&*self.0, &*other.0))
    }
}

impl<T: num::Num> Add<&TensorRef<T>> for &TensorRef<T> {
    type Output = TensorRef<T>;

    fn add(self, other: &TensorRef<T>) -> Self::Output {
        TensorRef(Tensor::add(&*self.0, &*other.0))
    }
}

// Add support for Arc<DTensor<T>> in TensorRef operations
impl<T: num::Num> Add<Arc<DTensor<T>>> for TensorRef<T> {
    type Output = TensorRef<T>;

    fn add(self, other: Arc<DTensor<T>>) -> Self::Output {
        TensorRef(Tensor::add(&*self.0, &*other))
    }
}

impl<T: num::Num> Add<&Arc<DTensor<T>>> for TensorRef<T> {
    type Output = TensorRef<T>;

    fn add(self, other: &Arc<DTensor<T>>) -> Self::Output {
        TensorRef(Tensor::add(&*self.0, &**other))
    }
}
