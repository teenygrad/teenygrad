/*
 * Copyright (c) 2025 Teenygrad. All rights reserved.
 *
 * This program is free software: you can redistribute it and/or modify it
 * under the terms of the GNU General Public License as published by the
 * Free Software Foundation, either version 3 of the License, or (at your
 * option) any later version.
 *
 * This program is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program. If not, see <https://www.gnu.org/licenses/>.
 */

use std::{fs::File, path::Path};

use memmap2::{Mmap, MmapOptions};

pub use teeny_core::safetensors::{Dtype, SafeTensors, SafeTensorsError, TensorView};

use crate::error::{Error, Result};

pub struct SafeTensorsMmaps {
    pub mmaps: Vec<Mmap>,
}

impl SafeTensorsMmaps {
    pub fn from_pretrained(folder: &Path) -> Result<Self> {
        use std::fs;

        let mut mmaps = Vec::new();

        for entry in fs::read_dir(folder)? {
            let entry = entry?;
            let path = entry.path();
            if let Some(ext) = path.extension() {
                if ext == "safetensors" {
                    let mmap = Self::mmap_file(&path)?;
                    mmaps.push(mmap);
                }
            }
        }

        Ok(Self { mmaps })
    }

    fn mmap_file(path: &Path) -> Result<Mmap> {
        let file = File::open(path).unwrap();
        let mmap = unsafe { MmapOptions::new().map(&file).unwrap() };
        Ok(mmap)
    }
}

pub struct FileSafeTensors<'data> {
    tensors: Vec<safetensors::SafeTensors<'data>>,
}

impl<'data> FileSafeTensors<'data> {
    pub fn from_pretrained(mmaps: &'data SafeTensorsMmaps) -> Result<Self> {
        let tensors = mmaps
            .mmaps
            .iter()
            .map(|mmap| safetensors::SafeTensors::deserialize(mmap).unwrap())
            .collect();

        Ok(FileSafeTensors { tensors })
    }
}

impl<'data> SafeTensors<'data> for FileSafeTensors<'data> {
    fn tensors(&'data self) -> Vec<(String, TensorView<'data>)> {
        self.tensors
            .iter()
            .flat_map(|t| t.tensors())
            .collect::<Vec<_>>()
    }

    fn iter(&self) -> impl Iterator<Item = (&str, TensorView<'data>)> {
        self.tensors.iter().flat_map(|t| t.iter())
    }

    fn tensor(&'data self, tensor_name: &str) -> Result<TensorView<'data>> {
        self.tensors
            .iter()
            .find_map(|t| t.tensor(tensor_name).ok())
            .ok_or(
                Error::SafeTensorsError(SafeTensorsError::TensorNotFound(tensor_name.to_string()))
                    .into(),
            )
    }

    fn names(&self) -> Vec<&'_ str> {
        self.tensors.iter().flat_map(|t| t.names()).collect()
    }

    fn len(&self) -> usize {
        self.tensors.iter().map(|t| t.len()).sum()
    }

    fn is_empty(&self) -> bool {
        self.tensors.iter().all(|t| t.is_empty())
    }
}
