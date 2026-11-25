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
    /// Whether this checkpoint has already been through `inkling2_convert`, so the
    /// tensors are the f32 the engine loads and need no packing, folding or
    /// expansion.
    converted: bool,
}

/// Packs bytes four to an f32 element, which is how the quantised kernels read them.
///
/// Parameter buffers take the graph's dtype, so packed weights travel inside f32
/// elements and the kernel bitcasts one back to an i32 to mask a byte out of it. The
/// bytes are laid out little-endian, so byte `i` of the input is byte `i % 4` of word
/// `i / 4`, which is what [`super::kernels::expert_gather_nvfp4`] indexes.
///
/// A trailing partial word is zero-filled. That is safe because the kernel masks
/// lanes past `K`, so those bytes are never read.
pub fn pack_bytes(bytes: &[f32]) -> Vec<f32> {
    bytes
        .chunks(4)
        .map(|c| {
            let mut w = [0u8; 4];
            for (i, v) in c.iter().enumerate() {
                w[i] = *v as u8;
            }
            f32::from_le_bytes(w)
        })
        .collect()
}

/// Packs each row of `row_len` bytes into its own whole words, four bytes each.
///
/// Rows are padded rather than run together: the kernels address a row as
/// `row * words_per_row`, so a row whose byte count is not a multiple of four must
/// still start on a word boundary. With Inkling-2-Mini's widths nothing actually
/// pads (a 2304-wide row gives 288 weight words and 36 scale words exactly), but a
/// narrow row would, and the tests use narrow rows.
pub fn pack_rows(bytes: &[f32], row_len: usize) -> Vec<f32> {
    assert!(row_len > 0, "pack_rows: row_len must be positive");
    assert!(
        bytes.len().is_multiple_of(row_len),
        "pack_rows: {} bytes is not a whole number of {row_len}-byte rows",
        bytes.len()
    );
    bytes.chunks(row_len).flat_map(pack_bytes).collect()
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
        let converted = index.contains_key("__converted__");
        Ok(Self {
            dir: dir.to_path_buf(),
            index,
            shards: HashMap::new(),
            converted,
        })
    }

    /// Tensor names in the checkpoint.
    pub fn names(&self) -> impl Iterator<Item = &str> {
        self.index.keys().map(String::as_str)
    }

    /// The encoding `tensor` would return for `name`, without reading its data.
    pub fn encoding_of(&mut self, name: &str) -> anyhow::Result<Encoding> {
        Ok(self.tensor(name)?.2)
    }

    /// Whether this checkpoint is already in the engine's load-time form.
    pub fn is_converted(&self) -> bool {
        self.converted
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

impl Checkpoint {
    /// Reads a tensor the way a graph parameter wants it.
    ///
    /// Floats are widened as [`Checkpoint::tensor`] does. Quantised tensors are
    /// **packed four bytes to an f32 element**, per row, where the row is the
    /// tensor's last dimension: `w13_weight` is `[experts, 2 * inter, hidden / 2]`
    /// bytes, so each `hidden / 2` row becomes its own whole words, which is how
    /// [`super::kernels::expert_gather_nvfp4`] addresses them.
    ///
    /// The returned length is what `param_shapes` declares, so a mismatch here is a
    /// mismatch the loader can report rather than one `load_param_f32` discovers.
    pub fn param(&mut self, name: &str) -> anyhow::Result<Vec<f32>> {
        let (values, shape, enc) = self.tensor(name)?;
        Ok(match enc {
            Encoding::Float => values,
            Encoding::Bytes => {
                let row = *shape.last().ok_or_else(|| {
                    anyhow::anyhow!("quantised tensor {name} has no shape to pack by")
                })?;
                pack_rows(&values, row)
            }
        })
    }

    /// NVFP4's two scale levels folded into one f32 per 16-weight block.
    ///
    /// A weight is `e2m1(nibble) * e4m3(scale) * (amax / denominator)`. The last two
    /// factors are constant per block and per expert, so multiplying them here leaves
    /// the kernel a single load where it used to decode an E4M3 byte per *weight* --
    /// a shift, a mask, three field splits, a bit construction, a reinterpret and two
    /// selects, sixteen times over for a scale sixteen weights share.
    ///
    /// Costs 4 bytes a block instead of 1, which is 0.25 bytes a weight.
    pub fn folded_scale(&mut self, prefix: &str) -> anyhow::Result<Vec<f32>> {
        let (bytes, shape, _) = self.tensor(&format!("{prefix}_scale"))?;
        // A converted checkpoint stores the product already.
        if self.converted {
            return Ok(bytes);
        }
        let s2 = self.expert_scale2(prefix)?;
        anyhow::ensure!(
            !s2.is_empty() && bytes.len() % s2.len() == 0,
            "{prefix}_scale has {} entries, which {} experts do not divide",
            bytes.len(),
            s2.len()
        );
        anyhow::ensure!(
            shape.first() == Some(&s2.len()),
            "{prefix}_scale is {shape:?} but there are {} experts",
            s2.len()
        );
        let per_expert = bytes.len() / s2.len();
        Ok(bytes
            .iter()
            .enumerate()
            .map(|(i, b)| e4m3_to_f32(*b as u8) * s2[i / per_expert])
            .collect())
    }

    /// NVFP4's per-expert scale, which the checkpoint does not store as one tensor.
    ///
    /// It keeps `<prefix>_amax` per expert and a single
    /// `<prefix>_outer_scale_denominator` for the layer; the scale the kernel wants
    /// is the quotient. The MXFP8 layers have neither, and
    /// [`super::kernels::expert_gather_mxfp8`] takes no per-expert scale at all.
    pub fn expert_scale2(&mut self, prefix: &str) -> anyhow::Result<Vec<f32>> {
        let (amax, _, _) = self.tensor(&format!("{prefix}_amax"))?;
        let (denom, _, _) = self.tensor(&format!("{prefix}_outer_scale_denominator"))?;
        let d = *denom
            .first()
            .ok_or_else(|| anyhow::anyhow!("{prefix}_outer_scale_denominator is empty"))?;
        anyhow::ensure!(
            d != 0.0 && d.is_finite(),
            "{prefix}_outer_scale_denominator is {d}, which cannot divide"
        );
        Ok(amax.iter().map(|a| a / d).collect())
    }
}

/// One F8_E4M3 byte as an f32: sign, four exponent bits biased by 7, three mantissa
/// bits, with a zero exponent meaning a subnormal `m * 2^-9`.
pub fn e4m3_to_f32(b: u8) -> f32 {
    let sign = if b >> 7 == 1 { -1.0f32 } else { 1.0 };
    let e = (b >> 3) & 0xF;
    let m = (b & 7) as f32;
    let mag = if e == 0 {
        m * 2.0f32.powi(-9)
    } else {
        (1.0 + m / 8.0) * 2.0f32.powi(e as i32 - 7)
    };
    sign * mag
}
