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

//! Quantization tooling for [teenygrad](https://teenygrad.org): reads `.safetensors` model
//! checkpoints and writes quantized `.safetensors` checkpoints following the
//! [compressed-tensors](https://github.com/vllm-project/compressed-tensors) convention (see
//! [`mod@format`]). Initially validated against Ultralytics YOLO models.
//!
//! Weight-only post-training quantization (INT8/INT4/FP8, [`quant`]) needs only the checkpoint
//! itself and is what this crate supports today. Static activation quantization (calibrating
//! from a forward pass over sample inputs) is tracked separately as `teenygrad-303.10` since it
//! depends on running the model's ONNX export through `teeny-onnx`/`teeny-compiler`.

#![warn(missing_docs)]

pub mod cli;
pub mod error;
pub mod format;
pub mod quant;
pub mod read;
pub mod validate;
pub mod write;

pub use error::{Error, Result};
