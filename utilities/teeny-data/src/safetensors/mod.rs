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

// use std::{fs::File, path::Path};

// use memmap2::{Mmap, MmapOptions};

// pub use teeny_core::safetensors::{Dtype, SafeTensors, SafeTensorsError, TensorView};

// use crate::error::{Error, Result};

// pub struct SafeTensorsMmaps {
//     pub mmaps: Vec<Mmap>,
// }

// impl SafeTensorsMmaps {
//     pub fn from_pretrained(folder: &Path) -> Result<Self> {
//         use std::fs;

//         let mut mmaps = Vec::new();

//         for entry in fs::read_dir(folder)? {
//             let entry = entry?;
//             let path = entry.path();
//             if let Some(ext) = path.extension()
//                 && ext == "safetensors"
//             {
//                 let mmap = Self::mmap_file(&path)?;
//                 mmaps.push(mmap);
//             }
//         }

//         Ok(Self { mmaps })
//     }

//     fn mmap_file(path: &Path) -> Result<Mmap> {
//         let file = File::open(path).unwrap();
//         let mmap = unsafe { MmapOptions::new().map(&file).unwrap() };
//         Ok(mmap)
//     }
// }

// pub struct FileSafeTensors<'data> {
//     tensors: Vec<safetensors::SafeTensors<'data>>,
// }

// impl<'data> FileSafeTensors<'data> {
//     pub fn from_pretrained(mmaps: &'data SafeTensorsMmaps) -> Result<Self> {
//         let tensors = mmaps
//             .mmaps
//             .iter()
//             .map(|mmap| safetensors::SafeTensors::deserialize(mmap).unwrap())
//             .collect();

//         Ok(FileSafeTensors { tensors })
//     }
// }

// impl<'data> SafeTensors<'data> for FileSafeTensors<'data> {
//     fn tensors(&'data self) -> Vec<(String, TensorView<'data>)> {
//         self.tensors
//             .iter()
//             .flat_map(|t| t.tensors())
//             .map(|(name, view)| (name, TensorView(view)))
//             .collect::<Vec<_>>()
//     }

//     fn iter(&self) -> impl Iterator<Item = (&str, TensorView<'data>)> {
//         self.tensors
//             .iter()
//             .flat_map(|t| t.iter())
//             .map(|(name, view)| (name, TensorView(view)))
//     }

//     fn tensor(&'data self, tensor_name: &str) -> Result<TensorView<'data>> {
//         self.tensors
//             .iter()
//             .find_map(|t| t.tensor(tensor_name).ok())
//             .map(TensorView)
//             .ok_or(
//                 Error::SafeTensorsError(SafeTensorsError::TensorNotFound(tensor_name.to_string()))
//                     .into(),
//             )
//     }

//     fn names(&self) -> Vec<&'_ str> {
//         self.tensors.iter().flat_map(|t| t.names()).collect()
//     }

//     fn len(&self) -> usize {
//         self.tensors.iter().map(|t| t.len()).sum()
//     }

//     fn is_empty(&self) -> bool {
//         self.tensors.iter().all(|t| t.is_empty())
//     }
// }
