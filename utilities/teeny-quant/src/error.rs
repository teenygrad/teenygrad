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

//! Error types for `teeny-quant`.

/// `teeny-quant`'s result alias.
pub type Result<T> = std::result::Result<T, Error>;

/// Errors produced by `teeny-quant`.
#[derive(thiserror::Error, Debug)]
pub enum Error {
    /// An I/O operation (reading/writing a `.safetensors` file) failed.
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    /// Reading or writing the underlying `.safetensors` container failed.
    #[error("safetensors error: {0}")]
    SafeTensors(#[from] safetensors::SafeTensorError),

    /// Reading the source checkpoint (via `teeny-data`) failed.
    #[error("failed to read source checkpoint: {0}")]
    Read(#[from] anyhow::Error),

    /// Serializing/deserializing quantization metadata as JSON failed.
    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),

    /// A tensor's source dtype isn't one this crate knows how to quantize/upcast.
    #[error("tensor '{tensor}' has unsupported source dtype {dtype:?}")]
    UnsupportedDtype {
        /// The tensor's name.
        tensor: String,
        /// The tensor's on-disk dtype.
        dtype: safetensors::Dtype,
    },

    /// A requested tensor doesn't exist in the source checkpoint.
    #[error("tensor '{0}' not found")]
    TensorNotFound(String),

    /// A quantization axis was out of range for the tensor's rank.
    #[error("axis {axis} is out of range for tensor '{tensor}' with {rank} dimensions")]
    InvalidAxis {
        /// The tensor's name.
        tensor: String,
        /// The requested axis.
        axis: usize,
        /// The tensor's rank.
        rank: usize,
    },

    /// A group size of `0` was requested for group-wise quantization.
    #[error("group size must be non-zero (tensor '{0}')")]
    InvalidGroupSize(String),

    /// Two tensors that should describe the same underlying weight (e.g. a weight and its
    /// dequantized reconstruction) have different shapes.
    #[error("shape mismatch for tensor '{name}': {a:?} vs {b:?}")]
    ShapeMismatch {
        /// The tensor's name.
        name: String,
        /// The first shape.
        a: Vec<usize>,
        /// The second shape.
        b: Vec<usize>,
    },
}
