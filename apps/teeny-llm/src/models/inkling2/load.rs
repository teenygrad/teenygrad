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

use super::kernels::expert_gemv_nvfp4::{
    QBLOCK, ROWS, byte_scale_rows, interleave_row_bytes, interleave_rows,
};
use super::runner::checkpoint_name;
use super::weights::{Checkpoint, Encoding, pack_rows};
use crate::kernels::linear::{int8_name, interleave_lanes, pack_bf16_halves, quantize_lanes_i8};
use crate::kernels::linear_ime::quantize_ime_i8;

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
    /// NVFP4's E4M3 block scale bytes interleaved by rows, then each expert's outer
    /// scale halved ([`byte_scale_rows`]): from the raw bytes of an original
    /// checkpoint, or stored as `<prefix>_scale.e4m3rows` in a converted one
    /// (`inkling2_convert --add-compact`).
    ByteScale(String),
    /// Read what the inner source reads, then pack it as bf16, two values to a word,
    /// for [`crate::kernels::linear::Linear4Bf16Op`]. The `.weight_bf16` slots.
    PackedBf16(Box<Source>),
    /// [`Source::PackedBf16`], then interleaved by blocks of output features for
    /// [`crate::kernels::linear::LinearLanesBf16Op`]. The `.weight_lanes` slots.
    LanesBf16(Box<Source>),
    /// Read what the inner source reads, then quantise it to int8 per output row
    /// for [`crate::kernels::linear::LinearLanesI8Op`]. The `.weight_i8` slots.
    LanesI8(Box<Source>),
    /// Read what the inner source reads, then quantise it to int8 per output row in
    /// the A100 matrix engine's tiles, for [`crate::kernels::linear_ime::LinearImeOp`].
    /// The `.weight_ime` slots; stored as `<weight>.i8ime` when converted.
    ImeI8(Box<Source>),
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
    if let Some(base) = slot
        .strip_suffix("_bf16")
        .filter(|b| b.ends_with(".weight"))
    {
        return Source::PackedBf16(Box::new(resolve_slot(base, layer)));
    }
    if let Some(stem) = slot.strip_suffix(".weight_ime") {
        return Source::ImeI8(Box::new(resolve_slot(&format!("{stem}.weight"), layer)));
    }
    if let Some(stem) = slot.strip_suffix(".weight_i8") {
        return Source::LanesI8(Box::new(resolve_slot(&format!("{stem}.weight"), layer)));
    }
    if let Some(stem) = slot.strip_suffix(".weight_lanes") {
        return Source::LanesBf16(Box::new(resolve_slot(&format!("{stem}.weight"), layer)));
    }
    match slot {
        // The NVFP4 gathers ask for a folded scale by name; the MXFP8 ones
        // keep their own E8M0 bytes under the plain `scale`.
        // The MXFP8 projection's E8M0 scales are bytes per row, interleaved like
        // its weights.
        "mlp.experts.weight_rows"
        | "mlp.experts_down.weight_rows"
        | "mlp.experts.scale_rows"
        | "mlp.experts_down.scale_rows" => Source::InterleavedWeight(checkpoint_name(slot, layer)),
        "mlp.experts.byte_scale_rows" | "mlp.experts_down.byte_scale_rows" => {
            let name = checkpoint_name(slot, layer);
            Source::ByteScale(name.strip_suffix("_scale").unwrap_or(&name).to_string())
        }
        "mlp.experts.folded_scale_rows" | "mlp.experts_down.folded_scale_rows" => {
            let name = checkpoint_name(slot, layer);
            Source::InterleavedScale(name.strip_suffix("_scale").unwrap_or(&name).to_string())
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
        "attn.rel_logits_proj.weight" => Source::Transposed(checkpoint_name(slot, layer), 0),
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
        // A converted checkpoint folded its scale bytes away; only the compact
        // tensor carries them.
        Source::ByteScale(prefix) if ck.is_converted() => {
            let stored = byte_scale_name(prefix);
            (!ck.has(&stored)).then_some(stored)
        }
        Source::FoldedScale(prefix)
        | Source::InterleavedScale(prefix)
        | Source::ByteScale(prefix) => {
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
        Source::PackedBf16(inner)
        | Source::LanesBf16(inner)
        | Source::LanesI8(inner)
        | Source::ImeI8(inner) => missing_tensor(ck, inner),
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

/// The name a converted checkpoint stores a linear's A100 matrix-engine int8 buffer
/// under (`inkling2_convert --add-int8`).
pub fn ime_name(name: &str) -> String {
    format!("{name}.i8ime")
}

/// The name a converted checkpoint stores a weight already packed as bf16 pairs
/// under (`inkling2_convert --add-compact`).
pub fn bf16_name(name: &str) -> String {
    format!("{name}.bf16")
}

/// The name a converted checkpoint stores an NVFP4 projection's byte scales under.
pub fn byte_scale_name(prefix: &str) -> String {
    format!("{prefix}_scale.e4m3rows")
}

/// The stored tensor a slot's values are, unchanged, if there is one: then they can
/// go from the checkpoint's mapping to the device without a copy in between.
///
/// Mirrors the branches of [`read_source`] that return a stored tensor as it is,
/// which on a converted checkpoint is nearly all of them by bytes.
fn stored_as_is(ck: &Checkpoint, src: &Source) -> Option<String> {
    let name = match src {
        Source::Tensor(name) => name.clone(),
        Source::InterleavedWeight(name) if ck.is_converted() => name.clone(),
        Source::InterleavedScale(prefix) if ck.is_converted() => format!("{prefix}_scale"),
        Source::ByteScale(prefix) => byte_scale_name(prefix),
        Source::PackedBf16(inner) => match inner.as_ref() {
            Source::Tensor(name) => bf16_name(name),
            _ => return None,
        },
        Source::LanesI8(inner) => match inner.as_ref() {
            Source::Tensor(name) => int8_name(name),
            _ => return None,
        },
        Source::ImeI8(inner) => match inner.as_ref() {
            Source::Tensor(name) => ime_name(name),
            _ => return None,
        },
        _ => return None,
    };
    ck.has(&name).then_some(name)
}

/// [`read`] for loading, through a shared checkpoint with every shard mapped
/// ([`Checkpoint::map_all`]): a slot whose values are a stored f32 tensor as it is
/// comes back borrowed from the mapping, and only the rest are computed, through
/// `private`, which is opened on first need.
///
/// Borrowing skips both the decode into a fresh vector and that vector's memory,
/// which for the converted Mini is almost every byte loaded.
pub fn read_shared<'a>(
    ck: &'a Checkpoint,
    private: &mut Option<Checkpoint>,
    slots: &[String],
    shapes: &[Vec<usize>],
    layer: usize,
) -> anyhow::Result<Vec<(String, std::borrow::Cow<'a, [f32]>)>> {
    use std::borrow::Cow;
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
            let want: usize = declared.iter().product();
            if let Some(slice) = stored_as_is(ck, &src).and_then(|n| ck.f32_slice(&n)) {
                if slice.len() == want {
                    return Ok((slot, Cow::Borrowed(slice)));
                }
            }
            if private.is_none() {
                *private = Some(Checkpoint::open(ck.dir())?);
            }
            let own = private.as_mut().expect("opened above");
            Ok((
                slot.clone(),
                Cow::Owned(read_source(own, &slot, &src, declared)?),
            ))
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
            if let Source::Tensor(name) = inner.as_ref() {
                let stored = bf16_name(name);
                if ck.has(&stored) {
                    return ck
                        .param(&stored)
                        .map_err(|e| anyhow::anyhow!("slot {slot} reads {stored}: {e}"));
                }
            }
            let values = read_source(ck, slot, inner, declared)?;
            pack_bf16_halves(&values, declared[0], declared[1] * 2)
                .map_err(|e| anyhow::anyhow!("slot {slot}: {e}"))?
        }
        Source::ImeI8(inner) => {
            anyhow::ensure!(
                declared.len() == 2 && declared[1] >= 1,
                "slot {slot} is declared {declared:?}, expected [rows, words + 1]"
            );
            let (n, k) = (declared[0], (declared[1] - 1) * 4);
            if let Source::Tensor(name) = inner.as_ref() {
                let stored = ime_name(name);
                if ck.has(&stored) {
                    return ck
                        .param(&stored)
                        .map_err(|e| anyhow::anyhow!("slot {slot} reads {stored}: {e}"));
                }
            }
            let values = read_source(ck, slot, inner, &[n, k])?;
            quantize_ime_i8(&values, n, k).map_err(|e| anyhow::anyhow!("slot {slot}: {e}"))?
        }
        Source::LanesI8(inner) => {
            anyhow::ensure!(
                declared.len() == 2 && declared[1] >= 1,
                "slot {slot} is declared {declared:?}, expected [rows, words + 1]"
            );
            let (n, k) = (declared[0], (declared[1] - 1) * 4);
            // A converted checkpoint may carry the buffer already (inkling2_convert
            // --add-int8), which turns seconds of quantising into a read.
            if let Source::Tensor(name) = inner.as_ref() {
                let stored = int8_name(name);
                if ck.has(&stored) {
                    let buf = ck
                        .param(&stored)
                        .map_err(|e| anyhow::anyhow!("slot {slot} reads {stored}: {e}"))?;
                    anyhow::ensure!(
                        buf.len() == n * (k / 4 + 1),
                        "slot {slot}: {stored} holds {} words, expected {}",
                        buf.len(),
                        n * (k / 4 + 1)
                    );
                    return Ok(buf);
                }
            }
            let values = read_source(ck, slot, inner, &[n, k])?;
            quantize_lanes_i8(&values, n, k).map_err(|e| anyhow::anyhow!("slot {slot}: {e}"))?
        }
        Source::LanesBf16(inner) => {
            anyhow::ensure!(
                declared.len() == 2,
                "slot {slot} is declared {declared:?}, expected [rows, words]"
            );
            let values = read_source(ck, slot, inner, declared)?;
            let packed = pack_bf16_halves(&values, declared[0], declared[1] * 2)
                .map_err(|e| anyhow::anyhow!("slot {slot}: {e}"))?;
            interleave_lanes(&packed, declared[0], declared[1])
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
        // A converted checkpoint stores these already interleaved (format 2).
        Source::InterleavedWeight(name) if ck.is_converted() => ck
            .param(name)
            .map_err(|e| anyhow::anyhow!("slot {slot} reads {name}: {e}"))?,
        Source::InterleavedScale(prefix) if ck.is_converted() => ck
            .folded_scale(prefix)
            .map_err(|e| anyhow::anyhow!("slot {slot} reads {prefix}: {e}"))?,
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
        Source::ByteScale(prefix) => {
            let stored = byte_scale_name(prefix);
            if ck.has(&stored) {
                return ck
                    .param(&stored)
                    .map_err(|e| anyhow::anyhow!("slot {slot} reads {stored}: {e}"));
            }
            anyhow::ensure!(
                !ck.is_converted(),
                "slot {slot}: this converted checkpoint has no {stored}; run \
                 inkling2_convert --model <original> --add-compact <converted> once"
            );
            let (bytes, shape, enc) = ck
                .tensor(&format!("{prefix}_scale"))
                .map_err(|e| anyhow::anyhow!("slot {slot} reads {prefix}_scale: {e}"))?;
            anyhow::ensure!(
                enc == Encoding::Bytes && shape.len() == 3,
                "slot {slot}: {prefix}_scale is {shape:?}, expected [experts, rows, blocks] of E4M3 bytes"
            );
            let scale2 = ck
                .expert_scale2(prefix)
                .map_err(|e| anyhow::anyhow!("slot {slot} scales from {prefix}: {e}"))?;
            byte_scale_rows(&bytes, &scale2, shape[0], shape[1], shape[2] * QBLOCK)
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
