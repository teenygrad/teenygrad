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

// ─── InstanceNorm1d ──────────────────────────────────────────────────────────

/// Instance normalisation over a 3-D `[N, C, L]` input.
///
/// Normalises each (N, C) pair independently over the L spatial dimension.
/// Output shape equals input shape.
///
/// Parameters:
/// - `num_features`        — number of channels C
/// - `eps`                 — numerical stability constant (default 1e-5)
/// - `momentum`            — EMA weight for running stats (default 0.1)
/// - `affine`              — if true, learns per-channel γ and β (default false)
/// - `track_running_stats` — maintain running mean/var (default false)
pub struct InstanceNorm1d<D: Dtype, IT, OT, const RANK: usize> {
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

impl<D: Dtype, IT, OT, const RANK: usize> InstanceNorm1d<D, IT, OT, RANK> {
    /// Creates a new layer with default `eps`/`momentum`, affine/running-stats off.
    pub fn new(num_features: usize) -> Self {
        Self {
            num_features,
            eps: 1e-5,
            momentum: 0.1,
            affine: false,
            track_running_stats: false,
            _pd: PhantomData,
        }
    }

    /// Sets the numerical stability constant.
    pub fn with_eps(mut self, eps: f64) -> Self {
        self.eps = eps;
        self
    }
    /// Sets the running-stats EMA momentum.
    pub fn with_momentum(mut self, m: f64) -> Self {
        self.momentum = m;
        self
    }
    /// Sets whether to learn per-channel scale/shift parameters.
    pub fn with_affine(mut self, a: bool) -> Self {
        self.affine = a;
        self
    }
    /// Sets whether to maintain running mean/variance across batches.
    pub fn with_track_running_stats(mut self, t: bool) -> Self {
        self.track_running_stats = t;
        self
    }
}

impl<D: Dtype, IT: Tensor<D, RANK> + EagerTensor, OT: Tensor<D, RANK>, const RANK: usize> Layer<IT>
    for InstanceNorm1d<D, IT, OT, RANK>
{
    type Output = OT;
    fn call(&self, _input: IT) -> Self::Output {
        todo!()
    }
}

// ─── InstanceNorm2d ──────────────────────────────────────────────────────────

/// Instance normalisation over a 4-D `[N, C, H, W]` input.
pub struct InstanceNorm2d<D: Dtype, IT, OT, const RANK: usize> {
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

impl<D: Dtype, IT, OT, const RANK: usize> InstanceNorm2d<D, IT, OT, RANK> {
    /// Creates a new layer with default `eps`/`momentum`, affine/running-stats off.
    pub fn new(num_features: usize) -> Self {
        Self {
            num_features,
            eps: 1e-5,
            momentum: 0.1,
            affine: false,
            track_running_stats: false,
            _pd: PhantomData,
        }
    }

    /// Sets the numerical stability constant.
    pub fn with_eps(mut self, eps: f64) -> Self {
        self.eps = eps;
        self
    }
    /// Sets the running-stats EMA momentum.
    pub fn with_momentum(mut self, m: f64) -> Self {
        self.momentum = m;
        self
    }
    /// Sets whether to learn per-channel scale/shift parameters.
    pub fn with_affine(mut self, a: bool) -> Self {
        self.affine = a;
        self
    }
    /// Sets whether to maintain running mean/variance across batches.
    pub fn with_track_running_stats(mut self, t: bool) -> Self {
        self.track_running_stats = t;
        self
    }
}

impl<D: Dtype, IT: Tensor<D, RANK> + EagerTensor, OT: Tensor<D, RANK>, const RANK: usize> Layer<IT>
    for InstanceNorm2d<D, IT, OT, RANK>
{
    type Output = OT;
    fn call(&self, _input: IT) -> Self::Output {
        todo!()
    }
}

// ─── InstanceNorm3d ──────────────────────────────────────────────────────────

/// Instance normalisation over a 5-D `[N, C, D, H, W]` input.
pub struct InstanceNorm3d<D: Dtype, IT, OT, const RANK: usize> {
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

impl<D: Dtype, IT, OT, const RANK: usize> InstanceNorm3d<D, IT, OT, RANK> {
    /// Creates a new layer with default `eps`/`momentum`, affine/running-stats off.
    pub fn new(num_features: usize) -> Self {
        Self {
            num_features,
            eps: 1e-5,
            momentum: 0.1,
            affine: false,
            track_running_stats: false,
            _pd: PhantomData,
        }
    }

    /// Sets the numerical stability constant.
    pub fn with_eps(mut self, eps: f64) -> Self {
        self.eps = eps;
        self
    }
    /// Sets the running-stats EMA momentum.
    pub fn with_momentum(mut self, m: f64) -> Self {
        self.momentum = m;
        self
    }
    /// Sets whether to learn per-channel scale/shift parameters.
    pub fn with_affine(mut self, a: bool) -> Self {
        self.affine = a;
        self
    }
    /// Sets whether to maintain running mean/variance across batches.
    pub fn with_track_running_stats(mut self, t: bool) -> Self {
        self.track_running_stats = t;
        self
    }
}

impl<D: Dtype, IT: Tensor<D, RANK> + EagerTensor, OT: Tensor<D, RANK>, const RANK: usize> Layer<IT>
    for InstanceNorm3d<D, IT, OT, RANK>
{
    type Output = OT;
    fn call(&self, _input: IT) -> Self::Output {
        todo!()
    }
}
