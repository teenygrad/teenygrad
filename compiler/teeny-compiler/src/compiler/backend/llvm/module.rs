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

use teeny_core::dtype;

/// A module compiled by the LLVM/MLIR backend.
#[derive(Debug, Clone, Default)]
pub struct MlirModule<N: dtype::Dtype> {
    _marker: std::marker::PhantomData<N>,
}

impl<N: dtype::Dtype> MlirModule<N> {
    /// Creates a new, empty MLIR module.
    pub fn new() -> Self {
        Self {
            _marker: std::marker::PhantomData,
        }
    }
}
