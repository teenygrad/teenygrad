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

//! Filling a compiled layer's parameters from a checkpoint.
//!
//! A graph names its weight slots after the checkpoint where it can
//! ([`super::runner::checkpoint_name`] maps the rest), so loading is: ask the
//! compiled model for its slots, resolve each name, read it, and hand it over.
//!
//! Two slots are not a plain read. `mlp.experts.scale2` and
//! `mlp.experts_down.scale2` are NVFP4's per-expert scale, which the checkpoint
//! keeps as an `_amax` per expert over a denominator shared by the layer, so
//! [`super::weights::Checkpoint::expert_scale2`] computes them.
//!
//! [`resolve`] does the naming half without a device, so a plan can be checked
//! against a real checkpoint before anything is compiled. That is worth having
//! separately: a missing tensor found at resolve time names the slot, while one
//! found at load time arrives as a failed read somewhere inside a step.

use super::runner::checkpoint_name;
use super::kernels::expert_gemv_nvfp4::{ROWS, interleave_row_bytes, interleave_rows};
use super::weights::{Checkpoint, Encoding, pack_rows};
use crate::kernels::linear::pack_bf16_halves;

/// What a slot needs from the checkpoint.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Source {
    /// Read this tensor and pass it through [`Checkpoint::param`].
    Tensor(String),
    /// Compute the per-expert scale from `<prefix>_amax` and the layer's
    /// denominator.
    ExpertScale(String),
    /// NVFP4's block scale and per-expert scale, multiplied together at load time so
    /// the kernel loads one f32 instead of decoding an E4M3 byte per weight.
    FoldedScale(String),
    /// Read this tensor and transpose it from `[in, out]` to the `[out, in]` a linear
    /// wants, given its row length in the checkpoint.
    ///
    /// Only the relative-position projection needs this. It is an `nn.Parameter` used
    /// as `states @ proj` in the reference, not an `nn.Linear`, so it is stored
    /// `[d_rel, rel_extent]`. The element count is the same either way, so reading it
    /// untransposed loads cleanly and computes nonsense; the name is `.proj` rather
    /// than `.weight` because it is not a linear's weight.
    Transposed(String, usize),
    /// Read this packed NVFP4 weight and interleave it by blocks of
    /// [`ROWS`] rows, for the integer projection.
    InterleavedWeight(String),
    /// Fold this NVFP4 scale as [`Source::FoldedScale`] does, halve it for the
    /// integer projection's E2M1-times-two weights, and interleave it by rows.
    InterleavedScale(String),
    /// Read what the inner source reads, then pack it as bf16, two values to a word,
    /// for [`crate::kernels::linear::Linear4Bf16Op`]. The `.weight_bf16` slots.
    PackedBf16(Box<Source>),
}

/// Where each slot of a layer comes from.
pub fn resolve(slots: &[String], layer: usize) -> Vec<(String, Source)> {
    slots
        .iter()
        .map(|slot| (slot.clone(), resolve_slot(slot, layer)))
        .collect()
}

/// Where one slot of a layer comes from.
fn resolve_slot(slot: &str, layer: usize) -> Source {
    // A packed bf16 weight reads whatever the plain `.weight` slot would.
    if let Some(base) = slot.strip_suffix("_bf16").filter(|b| b.ends_with(".weight")) {
        return Source::PackedBf16(Box::new(resolve_slot(base, layer)));
    }
    match slot {
                // The NVFP4 gathers ask for a folded scale by name; the MXFP8 ones
                // keep their own E8M0 bytes under the plain `scale`.
                "mlp.experts.weight_rows" | "mlp.experts_down.weight_rows" => {
                    Source::InterleavedWeight(checkpoint_name(slot, layer))
                }
                "mlp.experts.folded_scale_rows" | "mlp.experts_down.folded_scale_rows" => {
                    let name = checkpoint_name(slot, layer);
                    Source::InterleavedScale(
                        name.strip_suffix("_scale").unwrap_or(&name).to_string(),
                    )
                }
                "mlp.experts.folded_scale" | "mlp.experts_down.folded_scale" => {
                    let name = checkpoint_name(slot, layer);
                    Source::FoldedScale(name.strip_suffix("_scale").unwrap_or(&name).to_string())
                }
                "mlp.experts.scale2" | "mlp.experts_down.scale2" => {
                    // checkpoint_name maps these to the amax tensor; the prefix is
                    // that name without the suffix.
                    let amax = checkpoint_name(slot, layer);
                    Source::ExpertScale(amax.strip_suffix("_amax").unwrap_or(&amax).to_string())
                }
                "attn.rel_logits_proj.weight" => {
                    Source::Transposed(checkpoint_name(slot, layer), 0)
                }
                other => Source::Tensor(checkpoint_name(other, layer)),
    }
}

/// Checks every slot of a layer exists in the checkpoint, without reading them.
///
/// Returns the slots whose tensors are missing, as `(slot, what it looked for)`.
pub fn missing(ck: &Checkpoint, slots: &[String], layer: usize) -> Vec<(String, String)> {
    resolve(slots, layer)
        .into_iter()
        .filter_map(|(slot, src)| missing_tensor(ck, &src).map(|name| (slot, name)))
        .collect()
}

/// The checkpoint tensor `src` needs and the checkpoint lacks, if any.
fn missing_tensor(ck: &Checkpoint, src: &Source) -> Option<String> {
    match src {
        Source::Tensor(name) | Source::Transposed(name, _) | Source::InterleavedWeight(name) => {
            (!ck.has(name)).then(|| name.clone())
        }
        Source::FoldedScale(prefix) | Source::InterleavedScale(prefix) => {
            let scale = format!("{prefix}_scale");
            (!ck.has(&scale)).then_some(scale)
        }
        Source::ExpertScale(prefix) => {
            let amax = format!("{prefix}_amax");
            let denom = format!("{prefix}_outer_scale_denominator");
            if !ck.has(&amax) {
                Some(amax)
            } else if !ck.has(&denom) {
                Some(denom)
            } else {
                None
            }
        }
        Source::PackedBf16(inner) => missing_tensor(ck, inner),
    }
}

/// Reads the values for every slot of a layer.
///
/// Separate from handing them to a device so the sizes can be checked, and so a
/// caller that only wants to know whether a layer loads does not need one.
///
/// `shapes` are the slots' declared shapes, in the same order. The interleaved NVFP4
/// slots take their expert and row counts from them, since a converted checkpoint
/// stores its packed tensors flat.
pub fn read(
    ck: &mut Checkpoint,
    slots: &[String],
    shapes: &[Vec<usize>],
    layer: usize,
) -> anyhow::Result<Vec<(String, Vec<f32>)>> {
    anyhow::ensure!(
        shapes.len() == slots.len(),
        "{} slots but {} shapes",
        slots.len(),
        shapes.len()
    );
    resolve(slots, layer)
        .into_iter()
        .zip(shapes)
        .map(|((slot, src), declared)| {
            let values = read_source(ck, &slot, &src, declared)?;
            Ok((slot, values))
        })
        .collect()
}

/// Reads the values one slot's source describes. `declared` is the slot's shape.
fn read_source(
    ck: &mut Checkpoint,
    slot: &str,
    src: &Source,
    declared: &[usize],
) -> anyhow::Result<Vec<f32>> {
    Ok(match src {
        Source::PackedBf16(inner) => {
            anyhow::ensure!(
                declared.len() == 2,
                "slot {slot} is declared {declared:?}, expected [rows, words]"
            );
            let values = read_source(ck, slot, inner, declared)?;
            pack_bf16_halves(&values, declared[0], declared[1] * 2)
                .map_err(|e| anyhow::anyhow!("slot {slot}: {e}"))?
        }
        Source::Tensor(name) => ck
            .param(name)
            .map_err(|e| anyhow::anyhow!("slot {slot} reads {name}: {e}"))?,
        Source::ExpertScale(prefix) => ck
            .expert_scale2(prefix)
            .map_err(|e| anyhow::anyhow!("slot {slot} scales from {prefix}: {e}"))?,
        Source::FoldedScale(prefix) => ck
            .folded_scale(prefix)
            .map_err(|e| anyhow::anyhow!("slot {slot} folds {prefix}: {e}"))?,
        Source::InterleavedWeight(name) => {
            let (values, shape, enc) = ck
                .tensor(name)
                .map_err(|e| anyhow::anyhow!("slot {slot} reads {name}: {e}"))?;
            let (experts, rows) = expert_rows(slot, declared)?;
            let words = match enc {
                Encoding::Bytes => {
                    let row = *shape.last().ok_or_else(|| {
                        anyhow::anyhow!("slot {slot}: {name} has no shape to pack by")
                    })?;
                    pack_rows(&values, row)
                }
                // A converted checkpoint stores the packed words, flat.
                Encoding::Float => values,
            };
            // Four bytes, eight weights, to a word.
            let row_bytes = words.len() / (experts * rows) * 4;
            interleave_row_bytes(&words, experts, rows, row_bytes, ROWS)
        }
        Source::InterleavedScale(prefix) => {
            let (experts, rows) = expert_rows(slot, declared)?;
            let folded: Vec<f32> = ck
                .folded_scale(prefix)
                .map_err(|e| anyhow::anyhow!("slot {slot} folds {prefix}: {e}"))?
                .into_iter()
                .map(|s| s * 0.5)
                .collect();
            let width = folded.len() / (experts * rows);
            interleave_rows(&folded, experts, rows, width, ROWS)
        }
        Source::Transposed(name, _) => {
            let (flat, shape, _) = ck
                .tensor(name)
                .map_err(|e| anyhow::anyhow!("slot {slot} reads {name}: {e}"))?;
            anyhow::ensure!(
                shape.len() == 2,
                "slot {slot}: {name} is {shape:?}, expected a matrix to transpose"
            );
            let (rows, cols) = (shape[0], shape[1]);
            let mut out = vec![0.0f32; flat.len()];
            for r in 0..rows {
                for c in 0..cols {
                    out[c * rows + r] = flat[r * cols + c];
                }
            }
            out
        }
    })
}


/// The expert and row counts of a slot declared `[experts, rows, _]`.
fn expert_rows(slot: &str, declared: &[usize]) -> anyhow::Result<(usize, usize)> {
    anyhow::ensure!(
        declared.len() == 3 && declared[0] > 0 && declared[1] % ROWS == 0,
        "slot {slot} is declared {declared:?}, expected [experts, rows, _] with rows a \
         multiple of {ROWS}"
    );
    Ok((declared[0], declared[1]))
}
