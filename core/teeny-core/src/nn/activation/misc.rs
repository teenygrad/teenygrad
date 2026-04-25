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

pub struct LeakyRelu<D: Float, T, const RANK: usize> {
    pub negative_slope: f64,
    _pd: PhantomData<(D, T)>,
}

impl<D: Float, T, const RANK: usize> LeakyRelu<D, T, RANK> {
    pub fn new(negative_slope: f64) -> Self {
        Self { negative_slope, _pd: PhantomData }
    }
}

impl<D: Float, T: Tensor<D, RANK> + EagerTensor, const RANK: usize> Layer<T>
    for LeakyRelu<D, T, RANK>
{
    type Output = T;
    fn call(&self, _input: T) -> Self::Output {
        todo!()
    }
}

pub struct Threshold<D: Float, T, const RANK: usize> {
    pub threshold: f64,
    pub value: f64,
    _pd: PhantomData<(D, T)>,
}

impl<D: Float, T, const RANK: usize> Threshold<D, T, RANK> {
    pub fn new(threshold: f64, value: f64) -> Self {
        Self { threshold, value, _pd: PhantomData }
    }
}

impl<D: Float, T: Tensor<D, RANK> + EagerTensor, const RANK: usize> Layer<T>
    for Threshold<D, T, RANK>
{
    type Output = T;
    fn call(&self, _input: T) -> Self::Output {
        todo!()
    }
}

pub struct Softsign<D: Float, T, const RANK: usize> {
    _pd: PhantomData<(D, T)>,
}

impl<D: Float, T, const RANK: usize> Softsign<D, T, RANK> {
    pub fn new() -> Self {
        Self { _pd: PhantomData }
    }
}

impl<D: Float, T, const RANK: usize> Default for Softsign<D, T, RANK> {
    fn default() -> Self {
        Self::new()
    }
}

impl<D: Float, T: Tensor<D, RANK> + EagerTensor, const RANK: usize> Layer<T>
    for Softsign<D, T, RANK>
{
    type Output = T;
    fn call(&self, _input: T) -> Self::Output {
        todo!()
    }
}

pub struct Softshrink<D: Float, T, const RANK: usize> {
    pub lambda: f64,
    _pd: PhantomData<(D, T)>,
}

impl<D: Float, T, const RANK: usize> Softshrink<D, T, RANK> {
    pub fn new(lambda: f64) -> Self {
        Self { lambda, _pd: PhantomData }
    }
}

impl<D: Float, T: Tensor<D, RANK> + EagerTensor, const RANK: usize> Layer<T>
    for Softshrink<D, T, RANK>
{
    type Output = T;
    fn call(&self, _input: T) -> Self::Output {
        todo!()
    }
}

pub struct Softplus<D: Float, T, const RANK: usize> {
    pub beta: f64,
    pub threshold: f64,
    _pd: PhantomData<(D, T)>,
}

impl<D: Float, T, const RANK: usize> Softplus<D, T, RANK> {
    pub fn new(beta: f64, threshold: f64) -> Self {
        Self { beta, threshold, _pd: PhantomData }
    }
}

impl<D: Float, T: Tensor<D, RANK> + EagerTensor, const RANK: usize> Layer<T>
    for Softplus<D, T, RANK>
{
    type Output = T;
    fn call(&self, _input: T) -> Self::Output {
        todo!()
    }
}
