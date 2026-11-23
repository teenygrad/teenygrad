/*
 * Copyright (c) 2026 teenygrad (https://teenygrad.org).
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

//! Kernels shared between models, as graph-level custom ops that run on every teeny
//! backend.
//!
//! A model's own kernels live in its module (for example
//! [`models::functiongemma::kernels`](crate::models::functiongemma::kernels)); a kernel
//! moves here once it is not specific to one architecture.

pub mod attention;
pub mod last_row;
pub mod linear;
pub mod rope;
