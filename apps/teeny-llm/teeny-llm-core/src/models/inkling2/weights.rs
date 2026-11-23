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

//! Reading an Inkling-2 checkpoint.
//!
//! **Lazy, by shard, by tensor.** The Gemma port reads its whole `model.safetensors`
//! into a `HashMap<String, Vec<f32>>`, which is fine at 270M parameters and
//! impossible here: Inkling-2-Mini is 20.2 GB over two shards, and as f32 it would be
//! roughly 160 GB. This maps each shard and converts one tensor at a time, so only
//! the tensor being loaded is ever in memory.
//!
//! **Quantised tensors stay quantised.** The experts are NVFP4 (`U8` nibbles with
//! `F8_E4M3` scales) and a few layers are MXFP8. Those bytes are handed to the kernel
//! as they are, one byte per f32 element, and decoded in
//! [`super::kernels::expert_gather_nvfp4`]. Only the BF16 tensors are widened.
//!
//! See [`super::runner::checkpoint_name`] for the six slots whose graph names differ
//! from the checkpoint's.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use memmap2::Mmap;
use safetensors::{Dtype, SafeTensors};

/// A checkpoint on disk: the shard index, and each shard mapped on demand.
pub struct Checkpoint {
    dir: PathBuf,
    /// Tensor name to the shard holding it.
    index: HashMap<String, String>,
    /// Shards mapped so far, by file name.
    shards: HashMap<String, Mmap>,
}

/// How a tensor's bytes reach the device.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Encoding {
    /// Widened to f32: the BF16 and F32 tensors.
    Float,
    /// One byte per f32 element, value `0 ..= 255`: the packed NVFP4 nibbles and the
    /// E4M3 scales, which the kernel decodes.
    Bytes,
}

impl Checkpoint {
    /// Opens a checkpoint directory, reading `model.safetensors.index.json`.
    ///
    /// A single-file checkpoint is also accepted: every tensor then maps to that file.
    pub fn open(dir: &Path) -> anyhow::Result<Self> {
        let idx_path = dir.join("model.safetensors.index.json");
        let index = if idx_path.exists() {
            let text = std::fs::read_to_string(&idx_path)
                .map_err(|e| anyhow::anyhow!("reading {}: {e}", idx_path.display()))?;
            let v: serde_json::Value = serde_json::from_str(&text)
                .map_err(|e| anyhow::anyhow!("parsing {}: {e}", idx_path.display()))?;
            let map = v
                .get("weight_map")
                .and_then(|m| m.as_object())
                .ok_or_else(|| anyhow::anyhow!("{} has no weight_map", idx_path.display()))?;
            map.iter()
                .map(|(k, v)| {
                    let shard = v
                        .as_str()
                        .ok_or_else(|| anyhow::anyhow!("weight_map[{k}] is not a string"))?;
                    Ok((k.clone(), shard.to_string()))
                })
                .collect::<anyhow::Result<HashMap<_, _>>>()?
        } else {
            let single = dir.join("model.safetensors");
            anyhow::ensure!(
                single.exists(),
                "no model.safetensors.index.json or model.safetensors in {}",
                dir.display()
            );
            let bytes = std::fs::read(&single)
                .map_err(|e| anyhow::anyhow!("reading {}: {e}", single.display()))?;
            let st = SafeTensors::deserialize(&bytes)
                .map_err(|e| anyhow::anyhow!("parsing {}: {e}", single.display()))?;
            st.names()
                .into_iter()
                .map(|n| (n.to_string(), "model.safetensors".to_string()))
                .collect()
        };
        Ok(Self {
            dir: dir.to_path_buf(),
            index,
            shards: HashMap::new(),
        })
    }

    /// Tensor names in the checkpoint.
    pub fn names(&self) -> impl Iterator<Item = &str> {
        self.index.keys().map(String::as_str)
    }

    /// Whether the checkpoint has `name`.
    pub fn has(&self, name: &str) -> bool {
        self.index.contains_key(name)
    }

    fn shard(&mut self, name: &str) -> anyhow::Result<&Mmap> {
        let file = self
            .index
            .get(name)
            .ok_or_else(|| anyhow::anyhow!("no tensor {name} in the checkpoint"))?
            .clone();
        if !self.shards.contains_key(&file) {
            let path = self.dir.join(&file);
            let f = std::fs::File::open(&path)
                .map_err(|e| anyhow::anyhow!("opening {}: {e}", path.display()))?;
            // Safety: the checkpoint is read-only for the life of the process. A file
            // changed underneath would be a torn read, as for any mapped file.
            let map = unsafe { Mmap::map(&f) }
                .map_err(|e| anyhow::anyhow!("mapping {}: {e}", path.display()))?;
            self.shards.insert(file.clone(), map);
        }
        Ok(&self.shards[&file])
    }

    /// Reads one tensor, widening floats and passing quantised bytes through.
    ///
    /// Returns the values and the shape the checkpoint records, so a caller can check
    /// it against what the graph expects rather than trusting the name alone.
    pub fn tensor(&mut self, name: &str) -> anyhow::Result<(Vec<f32>, Vec<usize>, Encoding)> {
        let map = self.shard(name)?;
        let st = SafeTensors::deserialize(&map[..])
            .map_err(|e| anyhow::anyhow!("parsing the shard holding {name}: {e}"))?;
        let view = st
            .tensor(name)
            .map_err(|e| anyhow::anyhow!("reading tensor {name}: {e}"))?;
        let shape = view.shape().to_vec();
        let raw = view.data();
        let (values, enc) = match view.dtype() {
            Dtype::F32 => (
                raw.as_chunks::<4>()
                    .0
                    .iter()
                    .map(|c| f32::from_le_bytes(*c))
                    .collect(),
                Encoding::Float,
            ),
            Dtype::BF16 => (
                raw.as_chunks::<2>()
                    .0
                    .iter()
                    .map(|c| half::bf16::from_le_bytes(*c).to_f32())
                    .collect(),
                Encoding::Float,
            ),
            Dtype::F16 => (
                raw.as_chunks::<2>()
                    .0
                    .iter()
                    .map(|c| half::f16::from_le_bytes(*c).to_f32())
                    .collect(),
                Encoding::Float,
            ),
            // Quantised payloads: the packed nibbles and the E4M3 / MXFP8 scales. The
            // kernel decodes them, so each byte travels as its own value.
            Dtype::U8 | Dtype::F8_E4M3 | Dtype::F8_E5M2 => {
                (raw.iter().map(|b| *b as f32).collect(), Encoding::Bytes)
            }
            Dtype::I64 => (
                raw.as_chunks::<8>()
                    .0
                    .iter()
                    .map(|c| i64::from_le_bytes(*c) as f32)
                    .collect(),
                Encoding::Float,
            ),
            other => anyhow::bail!("tensor {name}: unsupported dtype {other:?}"),
        };
        Ok((values, shape, enc))
    }
}
