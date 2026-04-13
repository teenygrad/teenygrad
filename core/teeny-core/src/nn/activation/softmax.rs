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
    dtype::{Float, Tensor},
    nn::Node,
};

pub struct Softmax<D: Float, T: Tensor<D, RANK>, const RANK: usize> {
    pub dim: usize,
    _pd: PhantomData<(D, T)>,
}

impl<D: Float, T: Tensor<D, RANK>, const RANK: usize> Softmax<D, T, RANK> {
    pub fn new(dim: usize) -> Self {
        Self {
            dim,
            _pd: PhantomData,
        }
    }
}

impl<D: Float, T: Tensor<D, RANK>, const RANK: usize> Node<T> for Softmax<D, T, RANK> {
    type Output = T;

    fn call(&self, _input: T) -> Self::Output {
        todo!()
    }
}
