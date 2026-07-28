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

//! Host-side float byte serialization.
//!
//! Lives in a submodule so `teeny-triton`'s build script — which embeds
//! `dtype/mod.rs` into the no_core kernel DSL — never sees `alloc` / `Vec`.
//! The build script strips `pub mod …;` lines, so this file is host-only.

use super::Float;

/// Host-side little-endian byte serialization for float dtypes.
///
/// Bound RuntimeOps / host upload paths on `FloatBytes`; keep kernels on
/// [`Float`](super::Float) only.
pub trait FloatBytes: Float {
    /// Serialize this scalar to its little-endian byte representation.
    ///
    /// The byte length equals `Num::BITS / 8`. Used to upload host-side
    /// constant/parameter data into a device buffer of element type `D`.
    fn to_le_bytes(self) -> alloc::vec::Vec<u8>;
}

impl FloatBytes for f32 {
    fn to_le_bytes(self) -> alloc::vec::Vec<u8> {
        f32::to_le_bytes(self).to_vec()
    }
}

impl FloatBytes for f64 {
    fn to_le_bytes(self) -> alloc::vec::Vec<u8> {
        f64::to_le_bytes(self).to_vec()
    }
}
