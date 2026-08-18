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

//! Reading source `.safetensors` checkpoints. Opening/mmapping is delegated to
//! [`teeny_data::safetensors::SafeTensors`]; this module only adds the "upcast whatever float
//! dtype is on disk to `f32` for quantization math" step.

use std::collections::HashMap;
use std::io::Read as _;
use std::path::Path;

use half::{bf16, f16};
use safetensors::Dtype;
use safetensors::tensor::TensorView;

use crate::error::{Error, Result};

/// Reads `view`'s raw bytes as `f32`, upcasting from `F32`/`F16`/`BF16` as needed. Integer or
/// boolean source tensors (already-quantized inputs, masks, etc.) aren't supported -- callers
/// should pass those through unquantized rather than routing them through this function.
pub fn read_f32(view: &TensorView<'_>, name: &str) -> Result<Vec<f32>> {
    match view.dtype() {
        Dtype::F32 => Ok(view
            .data()
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes(b.try_into().expect("chunks_exact(4)")))
            .collect()),
        Dtype::F16 => Ok(view
            .data()
            .chunks_exact(2)
            .map(|b| f16::from_le_bytes(b.try_into().expect("chunks_exact(2)")).to_f32())
            .collect()),
        Dtype::BF16 => Ok(view
            .data()
            .chunks_exact(2)
            .map(|b| bf16::from_le_bytes(b.try_into().expect("chunks_exact(2)")).to_f32())
            .collect()),
        other => Err(Error::UnsupportedDtype {
            tensor: name.to_string(),
            dtype: other,
        }),
    }
}

/// Whether `dtype` is one [`read_f32`] can upcast from.
pub fn is_quantizable_float(dtype: Dtype) -> bool {
    matches!(dtype, Dtype::F32 | Dtype::F16 | Dtype::BF16)
}

/// Reads a `.safetensors` file's `__metadata__` string map (e.g. the embedded
/// `quantization_config` -- see [`crate::format`]) directly from `path`.
///
/// `safetensors::SafeTensors` (the type `teeny_data::safetensors::SafeTensors::tensors` hands
/// back) doesn't expose this map publicly -- only the crate-internal `Metadata` header type
/// does, via `SafeTensors::read_metadata`. That function insists the buffer's total length
/// exactly match the header-declared data length, but doesn't care about the data *bytes*
/// themselves for metadata extraction -- so this reads the real header off disk but only
/// zero-pads out to the real file length for the (unread) tensor payload, avoiding an I/O read
/// of the potentially multi-GB data section just to reach a handful of metadata strings.
pub fn read_metadata(path: &Path) -> Result<HashMap<String, String>> {
    let file_len = std::fs::metadata(path)?.len() as usize;
    let mut file = std::fs::File::open(path)?;

    let mut len_bytes = [0u8; 8];
    file.read_exact(&mut len_bytes)?;
    let header_len = u64::from_le_bytes(len_bytes) as usize;

    let mut buffer = vec![0u8; file_len];
    buffer[..8].copy_from_slice(&len_bytes);
    file.read_exact(&mut buffer[8..8 + header_len])?;

    let (_, metadata) = safetensors::SafeTensors::read_metadata(&buffer)?;
    Ok(metadata.metadata().clone().unwrap_or_default())
}
