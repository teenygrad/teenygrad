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

//! Writing `.safetensors` output. `teeny-data::safetensors` only supports mmap'd reading, so
//! this writes directly via the `safetensors` crate rather than extending that (leaf) crate --
//! see `teenygrad-303.2`.

use std::collections::HashMap;
use std::path::Path;

use safetensors::Dtype;
use safetensors::tensor::TensorView;

use crate::error::Result;

/// One tensor to be written: its dtype, shape, and raw little-endian bytes (already in whatever
/// on-disk representation `dtype` implies -- e.g. packed `U8` nibbles for INT4).
#[derive(Debug, Clone)]
pub struct OutputTensor {
    /// The tensor's on-disk dtype.
    pub dtype: Dtype,
    /// The tensor's logical shape (for packed formats like INT4 this is the *packed* shape --
    /// see `quantization_config` metadata for the true logical shape).
    pub shape: Vec<usize>,
    /// Raw little-endian element bytes, `shape.iter().product() * dtype.bitsize() / 8` long.
    pub data: Vec<u8>,
}

fn build_views(tensors: &HashMap<String, OutputTensor>) -> Result<HashMap<String, TensorView<'_>>> {
    let mut views = HashMap::with_capacity(tensors.len());
    for (name, t) in tensors {
        views.insert(
            name.clone(),
            TensorView::new(t.dtype, t.shape.clone(), &t.data)?,
        );
    }
    Ok(views)
}

/// Serializes `tensors` plus `metadata` (embedded as the file's string-keyed `__metadata__`
/// header -- e.g. the `quantization_config` JSON blob, see [`crate::format`]) to an in-memory
/// `.safetensors` byte buffer.
pub fn serialize_safetensors(
    tensors: &HashMap<String, OutputTensor>,
    metadata: HashMap<String, String>,
) -> Result<Vec<u8>> {
    let views = build_views(tensors)?;
    Ok(safetensors::serialize(views, Some(metadata))?)
}

/// Serializes `tensors`/`metadata` (see [`serialize_safetensors`]) directly to a `.safetensors`
/// file at `path`.
pub fn write_safetensors(
    path: &Path,
    tensors: &HashMap<String, OutputTensor>,
    metadata: HashMap<String, String>,
) -> Result<()> {
    let bytes = serialize_safetensors(tensors, metadata)?;
    std::fs::write(path, bytes)?;
    Ok(())
}
