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

//! Runs Gemma step graphs: loads a Hugging Face checkpoint, compiles a step graph per
//! token count, and drives prompt processing and greedy decoding over a [`KvCache`].

use std::collections::HashMap;
use std::path::Path;

use teeny_core::graph::Graph;
use teeny_core::model::LoweringMode;
use teeny_kernels::graph::TritonLowering;
use teeny_kvcache::{HostKvCache, KvCache, KvLayout, PrefixCache};
use teeny_test::model::{RuntimeModel, runtime};

use super::model::{StepGraph, build_step};
use super::{GemmaConfig, PastLen};
use crate::kernels::rope::rope_tables;

/// A checkpoint's tensors as f32, by `state_dict` key.
pub struct Weights(HashMap<String, Vec<f32>>);

impl Weights {
    /// Reads every tensor of a `model.safetensors` (BF16, F16 or F32) as f32.
    pub fn load(path: &Path) -> anyhow::Result<Self> {
        use safetensors::{Dtype, SafeTensors};
        let bytes = std::fs::read(path)
            .map_err(|e| anyhow::anyhow!("reading weights {}: {e}", path.display()))?;
        let st = SafeTensors::deserialize(&bytes)
            .map_err(|e| anyhow::anyhow!("parsing safetensors {}: {e}", path.display()))?;
        let mut map = HashMap::new();
        for (name, view) in st.tensors() {
            let data = view.data();
            let v: Vec<f32> = match view.dtype() {
                Dtype::BF16 => data
                    .as_chunks::<2>()
                    .0
                    .iter()
                    .map(|b| half::bf16::from_le_bytes(*b).to_f32())
                    .collect(),
                Dtype::F16 => data
                    .as_chunks::<2>()
                    .0
                    .iter()
                    .map(|b| half::f16::from_le_bytes(*b).to_f32())
                    .collect(),
                Dtype::F32 => data
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .map(|b| f32::from_le_bytes(*b))
                    .collect(),
                other => anyhow::bail!("tensor {name}: unsupported dtype {other:?}"),
            };
            map.insert(name, v);
        }
        Ok(Self(map))
    }

    /// The tensor for graph slot `slot`: its `state_dict` key, except the tied
    /// `lm_head.weight`, which is `model.embed_tokens.weight`.
    pub fn for_slot(&self, slot: &str) -> anyhow::Result<&[f32]> {
        let key = match slot {
            "lm_head.weight" if !self.0.contains_key(slot) => "model.embed_tokens.weight",
            k => k,
        };
        self.0
            .get(key)
            .map(Vec::as_slice)
            .ok_or_else(|| anyhow::anyhow!("graph slot {slot} has no checkpoint tensor {key}"))
    }
}

/// Whether to print every node's time and each step's total to stderr
/// (`TEENY_GEMMA_TRACE=1`).
fn trace() -> bool {
    static TRACE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *TRACE.get_or_init(|| std::env::var_os("TEENY_GEMMA_TRACE").is_some_and(|v| v != "0"))
}

/// A step graph for a fixed token count, compiled and loaded.
pub struct CompiledStep<'d> {
    model: RuntimeModel<'d>,
    out_idx: Vec<usize>,
    /// Tokens per step.
    pub tokens: usize,
}

impl<'d> CompiledStep<'d> {
    /// Records, compiles and loads the step graph for `tokens` tokens.
    pub fn new(
        device: &runtime::Device<'d>,
        cfg: &GemmaConfig,
        weights: &Weights,
        tokens: usize,
        cache: usize,
        past: &PastLen,
    ) -> anyhow::Result<Self> {
        let (mut model, out_idx) = Self::compile(device, cfg, tokens, cache, past, false)?;
        model.set_trace(trace());
        let slots: Vec<(String, usize, usize)> = model.param_info_named().collect();
        let mut used = std::collections::HashSet::new();
        for (slot, node, idx) in &slots {
            model.load_param_f32(*node, *idx, weights.for_slot(slot)?)?;
            used.insert(slot.as_str());
        }
        let unused: Vec<&String> = weights
            .0
            .keys()
            .filter(|k| !used.contains(k.as_str()))
            .collect();
        anyhow::ensure!(
            unused.is_empty(),
            "checkpoint tensors not loaded into the graph: {unused:?}"
        );
        Ok(Self {
            model,
            out_idx,
            tokens,
        })
    }

    /// Compiles the step graph for `tokens` tokens into the kernel cache without loading
    /// any weights: what an ahead-of-time build runs (`cargo teeny package`), so a device
    /// without `teenyc` finds every kernel cached. The target and cache folder are the
    /// runtime's (`TEENYC_CAPABILITY`, `TEENYC_CACHE_DIR`); `force` recompiles kernels that
    /// are already cached.
    pub fn precompile(
        device: &runtime::Device<'d>,
        cfg: &GemmaConfig,
        tokens: usize,
        cache: usize,
        force: bool,
    ) -> anyhow::Result<()> {
        Self::compile(device, cfg, tokens, cache, &PastLen::default(), force).map(drop)
    }

    /// Records and compiles the step graph, returning the model (weights not yet loaded)
    /// and the DAG indices of its outputs.
    fn compile(
        device: &runtime::Device<'d>,
        cfg: &GemmaConfig,
        tokens: usize,
        cache: usize,
        past: &PastLen,
        force: bool,
    ) -> anyhow::Result<(RuntimeModel<'d>, Vec<usize>)> {
        let StepGraph { graph, outputs } = build_step::<f32>(cfg, tokens, cache, past);
        let graph: &Graph = &graph.borrow();
        let lowering = TritonLowering::new();
        let (_, graph_to_dag) = lowering.lower_with_mapping(graph, LoweringMode::Inference)?;
        let target = runtime::default_target(device)?;
        let model = RuntimeModel::compile(
            device,
            graph,
            &lowering,
            LoweringMode::Inference,
            &target,
            tokens,
            force,
        )?;
        let out_idx = outputs
            .iter()
            .map(|(_, t)| graph_to_dag[t.node_id])
            .collect();
        Ok((model, out_idx))
    }

    /// Runs the step on `inputs` (in [`super::model::step_inputs`] order), returning
    /// `logits`, then `k_l`, `v_l` per layer.
    pub fn run(
        &mut self,
        device: &runtime::Device<'d>,
        inputs: &[&[f32]],
    ) -> anyhow::Result<Vec<Vec<f32>>> {
        if !trace() {
            return self.model.run(device, inputs, &self.out_idx);
        }
        let started = std::time::Instant::now();
        let out = self.model.run(device, inputs, &self.out_idx);
        eprintln!(
            "[step {} tokens] total {:.2?}",
            self.tokens,
            started.elapsed()
        );
        out
    }
}

/// Gemma generation state: the config and the KV cache, with the tokens it holds.
pub struct Session {
    /// The model config.
    pub cfg: GemmaConfig,
    /// The KV cache and the tokens whose keys and values it holds.
    pub cache: PrefixCache,
}

impl Session {
    /// A session over a [`HostKvCache`] with room for `capacity` tokens.
    pub fn new(cfg: GemmaConfig, capacity: usize) -> Self {
        let cache = HostKvCache::new(Self::layout(&cfg, capacity));
        Self::with_cache(cfg, Box::new(cache))
    }

    /// A session over `cache`, which must have [`Session::layout`]'s shape.
    pub fn with_cache(cfg: GemmaConfig, cache: Box<dyn KvCache>) -> Self {
        Self {
            cfg,
            cache: PrefixCache::new(cache),
        }
    }

    /// The cache shape a `cfg` model with room for `capacity` tokens needs.
    pub fn layout(cfg: &GemmaConfig, capacity: usize) -> KvLayout {
        KvLayout {
            layers: cfg.num_hidden_layers,
            row: cfg.num_key_value_heads * cfg.head_dim,
            capacity,
        }
    }

    /// The past length every compiled step reads; the cache keeps it up to date.
    pub fn past(&self) -> PastLen {
        self.cache.past_len()
    }

    /// Brings the cache up to `prompt` and returns its last token's logits. Cached
    /// tokens that still match `prompt` are kept; the rest are dropped, and the new
    /// tokens run through `chunk` (a step compiled for several tokens) while a whole
    /// chunk remains, then through `one` (a one-token step). Re-rendering a whole
    /// conversation each turn therefore only processes what changed.
    pub fn feed<'d>(
        &mut self,
        device: &runtime::Device<'d>,
        chunk: &mut CompiledStep<'d>,
        one: &mut CompiledStep<'d>,
        prompt: &[u32],
    ) -> anyhow::Result<Vec<f32>> {
        anyhow::ensure!(!prompt.is_empty(), "empty prompt");
        // At least one token must run, to produce the logits.
        let keep = self.cache.reuse(prompt, 1);

        let mut rest = &prompt[keep..];
        let mut logits = Vec::new();
        while !rest.is_empty() {
            let n = if rest.len() >= chunk.tokens {
                chunk.tokens
            } else {
                1
            };
            let step = if n == chunk.tokens {
                &mut *chunk
            } else {
                &mut *one
            };
            logits = self.step(device, step, &rest[..n])?;
            rest = &rest[n..];
        }
        Ok(logits)
    }

    /// Runs `step` on `ids` (its token count) at the cache's current length, appends
    /// the step's keys and values, and returns the last token's logits.
    pub fn step<'d>(
        &mut self,
        device: &runtime::Device<'d>,
        step: &mut CompiledStep<'d>,
        ids: &[u32],
    ) -> anyhow::Result<Vec<f32>> {
        anyhow::ensure!(
            ids.len() == step.tokens,
            "step compiled for {} tokens, given {}",
            step.tokens,
            ids.len()
        );
        let p0 = self.cache.len();
        let positions: Vec<usize> = (p0..p0 + ids.len()).collect();
        let hd = self.cfg.head_dim;
        let (cl, sl) = rope_tables(self.cfg.rope_local_base_freq, hd, &positions);
        let (cg, sg) = rope_tables(self.cfg.rope_theta, hd, &positions);
        let idf: Vec<f32> = ids.iter().map(|&i| i as f32).collect();

        let mut feed: Vec<&[f32]> = vec![&idf, &cl, &sl, &cg, &sg];
        let cache = self.cache.cache();
        for l in 0..self.cfg.num_hidden_layers {
            feed.push(cache.keys(l));
            feed.push(cache.values(l));
        }
        // The attention ops read `p0` from the cache's past length while the step runs.
        let mut out = step.run(device, &feed)?;
        let logits = std::mem::take(&mut out[0]);
        let k: Vec<&[f32]> = (0..self.cfg.num_hidden_layers)
            .map(|l| out[1 + 2 * l].as_slice())
            .collect();
        let v: Vec<&[f32]> = (0..self.cfg.num_hidden_layers)
            .map(|l| out[2 + 2 * l].as_slice())
            .collect();
        self.cache.append(ids, &k, &v)?;
        Ok(logits)
    }
}

/// The index of the largest logit (greedy decoding).
pub fn argmax(logits: &[f32]) -> u32 {
    let mut best = 0;
    for (i, &v) in logits.iter().enumerate() {
        if v > logits[best] {
            best = i;
        }
    }
    best as u32
}
