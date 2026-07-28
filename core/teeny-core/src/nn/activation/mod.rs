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

/// ELU and SELU.
pub mod elu;
/// GELU (exact and tanh-approximate).
pub mod gelu;
/// "Hard" (piecewise-linear) activation approximations: hardtanh, hardsigmoid, hardswish.
pub mod hard;
/// Other activations that don't fit the other modules.
pub mod misc;
/// ReLU and leaky ReLU.
pub mod relu;
/// Sigmoid and related activations.
pub mod sigmoid;
/// Softmax and log-softmax.
pub mod softmax;
/// Tanh and related activations.
pub mod tanh;
