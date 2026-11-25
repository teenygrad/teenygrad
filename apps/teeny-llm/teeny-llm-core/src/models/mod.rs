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

//! Model implementations, one module per model family, each behind its own cargo
//! feature, and [`Model`], the models this build can run.

use std::fmt;
use std::path::{Path, PathBuf};
use std::str::FromStr;

use crate::{Error, Result};

#[cfg(feature = "functiongemma3")]
pub mod functiongemma;
mod load;

pub use load::{LoadOptions, LoadedModel, Progress};

/// A model this build of teeny-llm-core can run: one variant per model family (and
/// version) whose feature is enabled, so a binary only offers what it was built with.
///
/// Command lines take it as `--model <tag>`, Ollama style: `family:size`
/// (`functiongemma3:270m`), or the family alone for its default size (`functiongemma3`).
/// [`Model::tags`] lists the choices, [`Model::spec`] describes a model without loading
/// it, and [`Model::locate`] finds its weights. A family that comes in one size, like
/// FunctionGemma 3, is a plain variant; one that comes in several carries a size enum of
/// its own (`Qwen3(Qwen3Size)`), so variant names stay short and only real sizes exist.
///
/// With no model feature enabled the enum has no variants.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum Model {
    /// FunctionGemma: Gemma 3 270M tuned for function calling (`models::functiongemma`).
    #[cfg(feature = "functiongemma3")]
    FunctionGemma3,
}

/// What a [`Model`] is, known without loading it. The exact parameter count and the
/// hyperparameters come from the weights themselves; see [`parameter_count`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub struct ModelSpec {
    /// The canonical tag, `family:size` (`functiongemma3:270m`).
    pub tag: &'static str,
    /// The family's tag, before the `:` (`functiongemma3`).
    pub family: &'static str,
    /// The size's tag, after the `:` (`270m`).
    pub size: &'static str,
    /// A human-readable name (`FunctionGemma 270M`).
    pub display_name: &'static str,
    /// The architecture and its version (`Gemma 3`).
    pub architecture: &'static str,
    /// The nominal parameter count the size is named for, as on the model card.
    pub nominal_parameters: u64,
    /// The longest context the model was trained for, in tokens.
    pub context_length: u32,
    /// The Hugging Face repository the weights come from.
    pub hf_repo: &'static str,
    /// The licence the weights are released under.
    pub license: &'static str,
}

impl Model {
    /// Every model compiled into this build, each in every size.
    pub const ALL: &'static [Model] = &[
        #[cfg(feature = "functiongemma3")]
        Model::FunctionGemma3,
    ];

    /// What the model is.
    pub fn spec(self) -> ModelSpec {
        match self {
            #[cfg(feature = "functiongemma3")]
            Model::FunctionGemma3 => ModelSpec {
                tag: "functiongemma3:270m",
                family: "functiongemma3",
                size: "270m",
                display_name: "FunctionGemma 270M",
                architecture: "Gemma 3",
                nominal_parameters: 270_000_000,
                context_length: 32_768,
                hf_repo: "google/functiongemma-270m-it",
                license: "Gemma",
            },
        }
    }

    /// Whether this is the size `--model <family>` picks when no size is given.
    pub fn is_default_size(self) -> bool {
        match self {
            #[cfg(feature = "functiongemma3")]
            Model::FunctionGemma3 => true,
        }
    }

    /// The canonical tag, `family:size`, which [`Display`](fmt::Display) prints and
    /// `/v1/models` lists.
    pub fn tag(self) -> &'static str {
        self.spec().tag
    }

    /// The Hugging Face repository the weights come from.
    pub fn hf_repo(self) -> &'static str {
        self.spec().hf_repo
    }

    /// The files a weights folder for this model must hold.
    pub fn required_files(self) -> &'static [&'static str] {
        match self {
            #[cfg(feature = "functiongemma3")]
            Model::FunctionGemma3 => &[
                "config.json",
                "model.safetensors",
                "tokenizer.json",
                "chat_template.jinja",
                "special_tokens_map.json",
            ],
        }
    }

    /// Every tag `--model` accepts in this build: each model's `family:size`, and each
    /// family alone (meaning its default size).
    pub fn tags() -> Vec<&'static str> {
        let mut tags = Vec::new();
        for m in Self::ALL {
            if m.is_default_size() {
                tags.push(m.spec().family);
            }
            tags.push(m.tag());
        }
        tags
    }

    /// The exact number of parameters in this model's weights folder `dir`; see
    /// [`parameter_count`].
    pub fn parameter_count(self, dir: &Path) -> Result<u64> {
        parameter_count(dir)
    }

    /// The folder holding this model's weights: `dir` when given (a `--model-dir`),
    /// otherwise the snapshot of [`Model::hf_repo`] in the local Hugging Face cache
    /// (`$HF_HUB_CACHE`, else `$HF_HOME/hub`, else `~/.cache/huggingface/hub`). Nothing is
    /// downloaded. Either way the folder must hold [`Model::required_files`].
    pub fn locate(self, dir: Option<&Path>) -> Result<PathBuf> {
        let folder = match dir {
            Some(d) => d.to_path_buf(),
            None => self.cached_snapshot()?,
        };
        let missing: Vec<&str> = self
            .required_files()
            .iter()
            .copied()
            .filter(|f| !folder.join(f).is_file())
            .collect();
        if !missing.is_empty() {
            return Err(Error::Weights(format!(
                "{} is not a {} weights folder: it lacks {}",
                folder.display(),
                self,
                missing.join(", ")
            )));
        }
        Ok(folder)
    }

    /// The cached snapshot of [`Model::hf_repo`]: the one `refs/main` names, or the only
    /// one there is.
    fn cached_snapshot(self) -> Result<PathBuf> {
        let hub = hf_hub_cache().ok_or_else(|| {
            Error::Weights(format!(
                "no --model-dir given for {self} and no Hugging Face cache to look in \
                 (set HF_HUB_CACHE, HF_HOME or HOME)"
            ))
        })?;
        let repo = hub.join(format!("models--{}", self.hf_repo().replace('/', "--")));
        let not_cached = || {
            Error::Weights(format!(
                "no --model-dir given for {self}, and {} is not in the Hugging Face cache \
                 at {}; download it (`hf download {}`) or pass --model-dir",
                self.hf_repo(),
                hub.display(),
                self.hf_repo()
            ))
        };
        let snapshots = repo.join("snapshots");
        if let Ok(commit) = std::fs::read_to_string(repo.join("refs").join("main")) {
            let pinned = snapshots.join(commit.trim());
            if pinned.is_dir() {
                return Ok(pinned);
            }
        }
        let mut found: Vec<PathBuf> = std::fs::read_dir(&snapshots)
            .map_err(|_| not_cached())?
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| p.is_dir())
            .collect();
        match found.len() {
            0 => Err(not_cached()),
            1 => Ok(found.remove(0)),
            n => Err(Error::Weights(format!(
                "{} has {n} cached snapshots under {} and no refs/main to choose between \
                 them; pass --model-dir",
                self.hf_repo(),
                snapshots.display()
            ))),
        }
    }
}

/// The Hugging Face hub cache folder, as the `huggingface_hub` library finds it.
fn hf_hub_cache() -> Option<PathBuf> {
    let var = |k: &str| {
        std::env::var_os(k)
            .filter(|v| !v.is_empty())
            .map(PathBuf::from)
    };
    var("HF_HUB_CACHE")
        .or_else(|| var("HF_HOME").map(|h| h.join("hub")))
        .or_else(|| var("HOME").map(|h| h.join(".cache").join("huggingface").join("hub")))
}

impl fmt::Display for Model {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.tag())
    }
}

impl FromStr for Model {
    type Err = Error;

    /// Parses a tag: `family:size`, or `family` for its default size. Case-insensitive.
    fn from_str(s: &str) -> Result<Self> {
        let tag = s.trim().to_ascii_lowercase();
        let (family, size) = match tag.split_once(':') {
            Some((f, z)) => (f, Some(z)),
            None => (tag.as_str(), None),
        };
        Self::ALL
            .iter()
            .copied()
            .find(|m| {
                let spec = m.spec();
                spec.family == family
                    && match size {
                        Some(z) => spec.size == z,
                        None => m.is_default_size(),
                    }
            })
            .ok_or_else(|| Error::ModelNotFound {
                requested: s.to_owned(),
                available: Self::tags().into_iter().map(str::to_owned).collect(),
            })
    }
}

/// The exact number of parameters in the safetensors files of `dir`, summed over every
/// tensor's shape. Only each file's header is read, not the weights. Sharded checkpoints
/// (`model-00001-of-00002.safetensors`, ...) are counted whole.
pub fn parameter_count(dir: &Path) -> Result<u64> {
    let weights_error = |what: String| Error::Weights(format!("{}: {what}", dir.display()));
    let mut files: Vec<PathBuf> = std::fs::read_dir(dir)
        .map_err(|e| weights_error(format!("cannot list the folder ({e})")))?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|x| x == "safetensors"))
        .collect();
    files.sort();
    if files.is_empty() {
        return Err(weights_error("no .safetensors files".into()));
    }
    let mut total = 0u64;
    for file in &files {
        total = total
            .checked_add(safetensors_parameters(file).map_err(|e| {
                weights_error(format!(
                    "{}: {e}",
                    file.file_name().unwrap_or_default().display()
                ))
            })?)
            .ok_or_else(|| weights_error("parameter count overflows u64".into()))?;
    }
    Ok(total)
}

/// Largest safetensors header accepted, as the format's reference reader limits it.
const MAX_SAFETENSORS_HEADER: u64 = 100 * 1024 * 1024;

/// The parameters in one safetensors file, from its header: an 8-byte little-endian length,
/// then that much JSON mapping tensor names to `{dtype, shape, data_offsets}`.
fn safetensors_parameters(file: &Path) -> std::result::Result<u64, String> {
    use std::io::Read;
    let mut f = std::fs::File::open(file).map_err(|e| format!("cannot open ({e})"))?;
    let mut len = [0u8; 8];
    f.read_exact(&mut len)
        .map_err(|e| format!("too short for a safetensors header ({e})"))?;
    let len = u64::from_le_bytes(len);
    if len > MAX_SAFETENSORS_HEADER {
        return Err(format!(
            "header length {len} is above the {MAX_SAFETENSORS_HEADER}-byte limit"
        ));
    }
    let mut header = vec![0u8; len as usize];
    f.read_exact(&mut header)
        .map_err(|e| format!("header truncated ({e})"))?;
    let header: serde_json::Value =
        serde_json::from_slice(&header).map_err(|e| format!("header is not JSON ({e})"))?;
    let tensors = header
        .as_object()
        .ok_or_else(|| "header is not a JSON object".to_owned())?;
    let mut total = 0u64;
    for (name, t) in tensors {
        if name == "__metadata__" {
            continue;
        }
        let shape = t["shape"]
            .as_array()
            .ok_or_else(|| format!("tensor `{name}` has no shape"))?;
        let mut n = 1u64;
        for d in shape {
            let d = d
                .as_u64()
                .ok_or_else(|| format!("tensor `{name}` has a non-integer dimension {d}"))?;
            n = n
                .checked_mul(d)
                .ok_or_else(|| format!("tensor `{name}` overflows u64"))?;
        }
        total = total
            .checked_add(n)
            .ok_or_else(|| "parameter count overflows u64".to_owned())?;
    }
    Ok(total)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(feature = "functiongemma3")]
    #[test]
    fn test_model_tags_round_trip() {
        for m in Model::ALL {
            let spec = m.spec();
            assert_eq!(spec.tag, format!("{}:{}", spec.family, spec.size));
            assert_eq!(m.tag().parse::<Model>().unwrap(), *m);
            assert_eq!(m.to_string(), m.tag());
        }
        assert_eq!(Model::tags(), ["functiongemma3", "functiongemma3:270m"]);
    }

    #[cfg(feature = "functiongemma3")]
    #[test]
    fn test_family_alone_picks_the_default_size() {
        for tag in [
            "functiongemma3",
            "FunctionGemma3:270M",
            " functiongemma3:270m ",
        ] {
            assert_eq!(
                tag.parse::<Model>().unwrap(),
                Model::FunctionGemma3,
                "{tag}"
            );
        }
        assert!("functiongemma3:1b".parse::<Model>().is_err());
    }

    #[test]
    fn test_parameter_count_sums_every_tensor_in_every_shard() {
        let dir = std::env::temp_dir().join(format!("teeny-llm-params-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let write = |name: &str, header: &str| {
            let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
            bytes.extend_from_slice(header.as_bytes());
            std::fs::write(dir.join(name), bytes).unwrap();
        };
        write(
            "model-00001-of-00002.safetensors",
            r#"{"__metadata__":{"format":"pt"},"a":{"dtype":"F32","shape":[3,4],"data_offsets":[0,48]}}"#,
        );
        write(
            "model-00002-of-00002.safetensors",
            r#"{"b":{"dtype":"BF16","shape":[5],"data_offsets":[0,10]},"c":{"dtype":"F32","shape":[],"data_offsets":[10,14]}}"#,
        );
        std::fs::write(dir.join("config.json"), "{}").unwrap();
        assert_eq!(parameter_count(&dir).unwrap(), 12 + 5 + 1);
        write("broken.safetensors", "not json");
        let err = parameter_count(&dir).unwrap_err().to_string();
        assert!(
            err.contains("broken.safetensors") && err.contains("not JSON"),
            "{err}"
        );
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn test_unknown_model_lists_the_available_ones() {
        let err = "gpt-4o".parse::<Model>().unwrap_err();
        assert!(
            matches!(&err, Error::ModelNotFound { requested, available }
                if requested == "gpt-4o" && *available == Model::tags()),
            "{err:?}"
        );
    }

    #[cfg(feature = "functiongemma3")]
    #[test]
    fn test_locate_checks_the_weights_folder() {
        let dir = std::env::temp_dir().join(format!("teeny-llm-locate-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let err = Model::FunctionGemma3
            .locate(Some(&dir))
            .unwrap_err()
            .to_string();
        assert!(err.contains("model.safetensors"), "{err}");
        for f in Model::FunctionGemma3.required_files() {
            std::fs::write(dir.join(f), b"").unwrap();
        }
        assert_eq!(Model::FunctionGemma3.locate(Some(&dir)).unwrap(), dir);
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[cfg(feature = "functiongemma3")]
    #[test]
    fn test_locate_finds_the_cached_snapshot_refs_main_names() {
        let hub = std::env::temp_dir().join(format!("teeny-llm-hub-{}", std::process::id()));
        let repo = hub.join("models--google--functiongemma-270m-it");
        let snap = repo.join("snapshots").join("abc123");
        std::fs::create_dir_all(&snap).unwrap();
        std::fs::create_dir_all(repo.join("snapshots").join("old")).unwrap();
        std::fs::create_dir_all(repo.join("refs")).unwrap();
        std::fs::write(repo.join("refs").join("main"), "abc123\n").unwrap();
        for f in Model::FunctionGemma3.required_files() {
            std::fs::write(snap.join(f), b"").unwrap();
        }
        // SAFETY: no other test in this binary reads or writes HF_HUB_CACHE.
        unsafe { std::env::set_var("HF_HUB_CACHE", &hub) };
        let found = Model::FunctionGemma3.locate(None);
        unsafe { std::env::remove_var("HF_HUB_CACHE") };
        assert_eq!(found.unwrap(), snap);
        std::fs::remove_dir_all(&hub).unwrap();
    }
}
