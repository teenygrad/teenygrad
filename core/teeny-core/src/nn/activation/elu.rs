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
    dtype::{EagerTensor, Float, Tensor},
    nn::Layer,
};

pub struct Elu<D: Float, T, const RANK: usize> {
    pub alpha: f64,
    _pd: PhantomData<(D, T)>,
}

impl<D: Float, T, const RANK: usize> Elu<D, T, RANK> {
    pub fn new(alpha: f64) -> Self {
        Self { alpha, _pd: PhantomData }
    }
}

impl<D: Float, T: Tensor<D, RANK> + EagerTensor, const RANK: usize> Layer<T> for Elu<D, T, RANK> {
    type Output = T;
    fn call(&self, _input: T) -> Self::Output {
        todo!()
    }
}

pub struct Selu<D: Float, T, const RANK: usize> {
    _pd: PhantomData<(D, T)>,
}

impl<D: Float, T, const RANK: usize> Selu<D, T, RANK> {
    pub fn new() -> Self {
        Self { _pd: PhantomData }
    }
}

impl<D: Float, T, const RANK: usize> Default for Selu<D, T, RANK> {
    fn default() -> Self {
        Self::new()
    }
}

impl<D: Float, T: Tensor<D, RANK> + EagerTensor, const RANK: usize> Layer<T> for Selu<D, T, RANK> {
    type Output = T;
    fn call(&self, _input: T) -> Self::Output {
        todo!()
    }
}

pub struct Celu<D: Float, T, const RANK: usize> {
    pub alpha: f64,
    _pd: PhantomData<(D, T)>,
}

impl<D: Float, T, const RANK: usize> Celu<D, T, RANK> {
    pub fn new(alpha: f64) -> Self {
        Self { alpha, _pd: PhantomData }
    }
}

impl<D: Float, T: Tensor<D, RANK> + EagerTensor, const RANK: usize> Layer<T> for Celu<D, T, RANK> {
    type Output = T;
    fn call(&self, _input: T) -> Self::Output {
        todo!()
    }
}
