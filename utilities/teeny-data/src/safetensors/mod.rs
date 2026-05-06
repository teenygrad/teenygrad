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

use std::path::Path;

use crate::error::Result;
use memmap2::{Mmap, MmapOptions};
use safetensors;
use std::fs::File;

pub struct SafeTensors {
    mmap: Mmap,
}

impl SafeTensors {
    pub fn from_pretrained(path: &Path) -> Result<Self> {
        let file = File::open(path)?;
        let mmap = unsafe { MmapOptions::new().map(&file)? };
        Ok(Self { mmap })
    }

    pub fn tensors(&self) -> Result<safetensors::SafeTensors<'_>> {
        Ok(safetensors::SafeTensors::deserialize(&self.mmap)?)
    }
}
