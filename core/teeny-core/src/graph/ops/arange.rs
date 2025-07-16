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

use std::marker::PhantomData;

use crate::{dtype::Dtype, tensor::shape::Shape};

#[derive(Debug, Clone)]
pub struct ArangeOp<S: Shape, N: Dtype> {
    pub start: N,
    pub end: N,
    pub step: N,
    _marker: PhantomData<S>,
}

impl<S: Shape, N: Dtype> ArangeOp<S, N> {
    pub fn new(start: N, end: N, step: N) -> Self {
        Self {
            start,
            end,
            step,
            _marker: PhantomData,
        }
    }
}
