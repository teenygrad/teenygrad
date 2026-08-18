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

//! Memory-mapped `.safetensors` file loading.

use std::path::Path;

use crate::error::Result;
use memmap2::{Mmap, MmapOptions};
use safetensors;
use std::fs::File;

/// A memory-mapped `.safetensors` file, ready to be deserialized without loading its full
/// contents into memory up front.
pub struct SafeTensors {
    mmap: Mmap,
}

impl SafeTensors {
    /// Opens and memory-maps the `.safetensors` file at `path`.
    pub fn from_pretrained(path: &Path) -> Result<Self> {
        let file = File::open(path)?;
        let mmap = unsafe { MmapOptions::new().map(&file)? };
        Ok(Self { mmap })
    }

    /// Deserializes the mapped file's tensor headers (tensor data is read lazily from the
    /// memory-mapped region on access).
    pub fn tensors(&self) -> Result<safetensors::SafeTensors<'_>> {
        Ok(safetensors::SafeTensors::deserialize(&self.mmap)?)
    }
}
