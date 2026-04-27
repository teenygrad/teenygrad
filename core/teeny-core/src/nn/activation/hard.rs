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

pub struct Hardtanh<D: Float, T, const RANK: usize> {
    pub min_val: f64,
    pub max_val: f64,
    _pd: PhantomData<(D, T)>,
}

impl<D: Float, T, const RANK: usize> Hardtanh<D, T, RANK> {
    pub fn new(min_val: f64, max_val: f64) -> Self {
        Self { min_val, max_val, _pd: PhantomData }
    }
}

impl<D: Float, T: Tensor<D, RANK> + EagerTensor, const RANK: usize> Layer<T>
    for Hardtanh<D, T, RANK>
{
    type Output = T;
    fn call(&self, _input: T) -> Self::Output {
        todo!()
    }
}

pub struct Relu6<D: Float, T, const RANK: usize> {
    _pd: PhantomData<(D, T)>,
}

impl<D: Float, T, const RANK: usize> Relu6<D, T, RANK> {
    pub fn new() -> Self {
        Self { _pd: PhantomData }
    }
}

impl<D: Float, T, const RANK: usize> Default for Relu6<D, T, RANK> {
    fn default() -> Self {
        Self::new()
    }
}

impl<D: Float, T: Tensor<D, RANK> + EagerTensor, const RANK: usize> Layer<T>
    for Relu6<D, T, RANK>
{
    type Output = T;
    fn call(&self, _input: T) -> Self::Output {
        todo!()
    }
}

pub struct Hardsigmoid<D: Float, T, const RANK: usize> {
    _pd: PhantomData<(D, T)>,
}

impl<D: Float, T, const RANK: usize> Hardsigmoid<D, T, RANK> {
    pub fn new() -> Self {
        Self { _pd: PhantomData }
    }
}

impl<D: Float, T, const RANK: usize> Default for Hardsigmoid<D, T, RANK> {
    fn default() -> Self {
        Self::new()
    }
}

impl<D: Float, T: Tensor<D, RANK> + EagerTensor, const RANK: usize> Layer<T>
    for Hardsigmoid<D, T, RANK>
{
    type Output = T;
    fn call(&self, _input: T) -> Self::Output {
        todo!()
    }
}

pub struct Hardswish<D: Float, T, const RANK: usize> {
    _pd: PhantomData<(D, T)>,
}

impl<D: Float, T, const RANK: usize> Hardswish<D, T, RANK> {
    pub fn new() -> Self {
        Self { _pd: PhantomData }
    }
}

impl<D: Float, T, const RANK: usize> Default for Hardswish<D, T, RANK> {
    fn default() -> Self {
        Self::new()
    }
}

impl<D: Float, T: Tensor<D, RANK> + EagerTensor, const RANK: usize> Layer<T>
    for Hardswish<D, T, RANK>
{
    type Output = T;
    fn call(&self, _input: T) -> Self::Output {
        todo!()
    }
}

pub struct Hardshrink<D: Float, T, const RANK: usize> {
    pub lambda: f64,
    _pd: PhantomData<(D, T)>,
}

impl<D: Float, T, const RANK: usize> Hardshrink<D, T, RANK> {
    pub fn new(lambda: f64) -> Self {
        Self { lambda, _pd: PhantomData }
    }
}

impl<D: Float, T: Tensor<D, RANK> + EagerTensor, const RANK: usize> Layer<T>
    for Hardshrink<D, T, RANK>
{
    type Output = T;
    fn call(&self, _input: T) -> Self::Output {
        todo!()
    }
}
