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

//! ONNX model format support for [teenygrad](https://teenygrad.org) — parses `.onnx` protobuf
//! files into a `teeny_core::graph::Graph` via [`Onnx`].

#![warn(missing_docs)]

/// Error types.
pub mod errors;

mod onnx;
pub use onnx::Onnx;
