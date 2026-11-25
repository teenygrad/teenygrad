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
use super::weights::Checkpoint;

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
}

/// Where each slot of a layer comes from.
pub fn resolve(slots: &[String], layer: usize) -> Vec<(String, Source)> {
    slots
        .iter()
        .map(|slot| {
            let src = match slot.as_str() {
                // The NVFP4 gathers ask for a folded scale by name; the MXFP8 ones
                // keep their own E8M0 bytes under the plain `scale`.
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
            };
            (slot.clone(), src)
        })
        .collect()
}

/// Checks every slot of a layer exists in the checkpoint, without reading them.
///
/// Returns the slots whose tensors are missing, as `(slot, what it looked for)`.
pub fn missing(ck: &Checkpoint, slots: &[String], layer: usize) -> Vec<(String, String)> {
    resolve(slots, layer)
        .into_iter()
        .filter_map(|(slot, src)| match src {
            Source::Tensor(name) | Source::Transposed(name, _) => {
                (!ck.has(&name)).then_some((slot, name))
            }
            Source::FoldedScale(prefix) => {
                let scale = format!("{prefix}_scale");
                (!ck.has(&scale)).then_some((slot, scale))
            }
            Source::ExpertScale(prefix) => {
                let amax = format!("{prefix}_amax");
                let denom = format!("{prefix}_outer_scale_denominator");
                if !ck.has(&amax) {
                    Some((slot, amax))
                } else if !ck.has(&denom) {
                    Some((slot, denom))
                } else {
                    None
                }
            }
        })
        .collect()
}

/// Reads the values for every slot of a layer.
///
/// Separate from handing them to a device so the sizes can be checked, and so a
/// caller that only wants to know whether a layer loads does not need one.
pub fn read(
    ck: &mut Checkpoint,
    slots: &[String],
    layer: usize,
) -> anyhow::Result<Vec<(String, Vec<f32>)>> {
    resolve(slots, layer)
        .into_iter()
        .map(|(slot, src)| {
            let values = match &src {
                Source::Tensor(name) => ck
                    .param(name)
                    .map_err(|e| anyhow::anyhow!("slot {slot} reads {name}: {e}"))?,
                Source::ExpertScale(prefix) => ck
                    .expert_scale2(prefix)
                    .map_err(|e| anyhow::anyhow!("slot {slot} scales from {prefix}: {e}"))?,
                Source::FoldedScale(prefix) => ck
                    .folded_scale(prefix)
                    .map_err(|e| anyhow::anyhow!("slot {slot} folds {prefix}: {e}"))?,
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
            };
            Ok((slot, values))
        })
        .collect()
}
