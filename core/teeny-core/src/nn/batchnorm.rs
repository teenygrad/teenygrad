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
    dtype::{Dtype, EagerTensor, Tensor},
    nn::Layer,
};

// ─── BatchNorm1d ─────────────────────────────────────────────────────────────

/// Batch normalisation over a 2-D `[N, C]` or 3-D `[N, C, L]` input.
///
/// During training the per-batch mean and variance are used; the running
/// statistics are updated with exponential moving average.  During inference
/// the frozen running statistics are used instead.
///
/// Parameters:
/// - `num_features` — number of channels C
/// - `eps`          — numerical stability constant added to the variance
/// - `momentum`     — weight of the current batch in the running-stats EMA
/// - `affine`       — if true, learns per-channel γ (weight) and β (bias)
/// - `track_running_stats` — if true, maintains running_mean / running_var
pub struct BatchNorm1d<D: Dtype, IT, OT, const RANK: usize> {
    /// Number of channels/features.
    pub num_features: usize,
    /// Numerical stability constant added to the variance.
    pub eps: f64,
    /// Weight of the current batch in the running-stats exponential moving average.
    pub momentum: f64,
    /// Whether to learn per-channel scale (gamma) and shift (beta) parameters.
    pub affine: bool,
    /// Whether to maintain running mean/variance across batches.
    pub track_running_stats: bool,
    _pd: PhantomData<(D, IT, OT)>,
}

impl<D: Dtype, IT, OT, const RANK: usize> BatchNorm1d<D, IT, OT, RANK> {
    /// Creates a new layer with default `eps`/`momentum`/`affine`/`track_running_stats`.
    pub fn new(num_features: usize) -> Self {
        Self {
            num_features,
            eps: 1e-5,
            momentum: 0.1,
            affine: true,
            track_running_stats: true,
            _pd: PhantomData,
        }
    }

    /// Sets the numerical stability constant.
    pub fn with_eps(mut self, eps: f64) -> Self {
        self.eps = eps;
        self
    }

    /// Sets the running-stats EMA momentum.
    pub fn with_momentum(mut self, momentum: f64) -> Self {
        self.momentum = momentum;
        self
    }

    /// Sets whether to learn per-channel scale/shift parameters.
    pub fn with_affine(mut self, affine: bool) -> Self {
        self.affine = affine;
        self
    }

    /// Sets whether to maintain running mean/variance across batches.
    pub fn with_track_running_stats(mut self, track: bool) -> Self {
        self.track_running_stats = track;
        self
    }
}

impl<D: Dtype, IT: Tensor<D, RANK> + EagerTensor, OT: Tensor<D, RANK>, const RANK: usize> Layer<IT>
    for BatchNorm1d<D, IT, OT, RANK>
{
    type Output = OT;

    fn call(&self, _input: IT) -> Self::Output {
        todo!()
    }
}

// ─── BatchNorm2d ─────────────────────────────────────────────────────────────

/// Batch normalisation over a 4-D `[N, C, H, W]` input.
///
/// Applies the same statistics as `BatchNorm1d` but treats each (C) channel
/// independently across the (N, H, W) axes.
pub struct BatchNorm2d<D: Dtype, IT, OT, const RANK: usize> {
    /// Number of channels/features.
    pub num_features: usize,
    /// Numerical stability constant added to the variance.
    pub eps: f64,
    /// Weight of the current batch in the running-stats exponential moving average.
    pub momentum: f64,
    /// Whether to learn per-channel scale (gamma) and shift (beta) parameters.
    pub affine: bool,
    /// Whether to maintain running mean/variance across batches.
    pub track_running_stats: bool,
    _pd: PhantomData<(D, IT, OT)>,
}

impl<D: Dtype, IT, OT, const RANK: usize> BatchNorm2d<D, IT, OT, RANK> {
    /// Creates a new layer with default `eps`/`momentum`/`affine`/`track_running_stats`.
    pub fn new(num_features: usize) -> Self {
        Self {
            num_features,
            eps: 1e-5,
            momentum: 0.1,
            affine: true,
            track_running_stats: true,
            _pd: PhantomData,
        }
    }

    /// Sets the numerical stability constant.
    pub fn with_eps(mut self, eps: f64) -> Self {
        self.eps = eps;
        self
    }

    /// Sets the running-stats EMA momentum.
    pub fn with_momentum(mut self, momentum: f64) -> Self {
        self.momentum = momentum;
        self
    }

    /// Sets whether to learn per-channel scale/shift parameters.
    pub fn with_affine(mut self, affine: bool) -> Self {
        self.affine = affine;
        self
    }

    /// Sets whether to maintain running mean/variance across batches.
    pub fn with_track_running_stats(mut self, track: bool) -> Self {
        self.track_running_stats = track;
        self
    }
}

impl<D: Dtype, IT: Tensor<D, RANK> + EagerTensor, OT: Tensor<D, RANK>, const RANK: usize> Layer<IT>
    for BatchNorm2d<D, IT, OT, RANK>
{
    type Output = OT;

    fn call(&self, _input: IT) -> Self::Output {
        todo!()
    }
}

// ─── BatchNorm3d ─────────────────────────────────────────────────────────────

/// Batch normalisation over a 5-D `[N, C, D, H, W]` input.
///
/// Applies the same statistics as `BatchNorm1d` but treats each (C) channel
/// independently across the (N, D, H, W) axes.
pub struct BatchNorm3d<D: Dtype, IT, OT, const RANK: usize> {
    /// Number of channels/features.
    pub num_features: usize,
    /// Numerical stability constant added to the variance.
    pub eps: f64,
    /// Weight of the current batch in the running-stats exponential moving average.
    pub momentum: f64,
    /// Whether to learn per-channel scale (gamma) and shift (beta) parameters.
    pub affine: bool,
    /// Whether to maintain running mean/variance across batches.
    pub track_running_stats: bool,
    _pd: PhantomData<(D, IT, OT)>,
}

impl<D: Dtype, IT, OT, const RANK: usize> BatchNorm3d<D, IT, OT, RANK> {
    /// Creates a new layer with default `eps`/`momentum`/`affine`/`track_running_stats`.
    pub fn new(num_features: usize) -> Self {
        Self {
            num_features,
            eps: 1e-5,
            momentum: 0.1,
            affine: true,
            track_running_stats: true,
            _pd: PhantomData,
        }
    }

    /// Sets the numerical stability constant.
    pub fn with_eps(mut self, eps: f64) -> Self {
        self.eps = eps;
        self
    }

    /// Sets the running-stats EMA momentum.
    pub fn with_momentum(mut self, momentum: f64) -> Self {
        self.momentum = momentum;
        self
    }

    /// Sets whether to learn per-channel scale/shift parameters.
    pub fn with_affine(mut self, affine: bool) -> Self {
        self.affine = affine;
        self
    }

    /// Sets whether to maintain running mean/variance across batches.
    pub fn with_track_running_stats(mut self, track: bool) -> Self {
        self.track_running_stats = track;
        self
    }
}

impl<D: Dtype, IT: Tensor<D, RANK> + EagerTensor, OT: Tensor<D, RANK>, const RANK: usize> Layer<IT>
    for BatchNorm3d<D, IT, OT, RANK>
{
    type Output = OT;

    fn call(&self, _input: IT) -> Self::Output {
        todo!()
    }
}
