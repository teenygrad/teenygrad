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

//! Runs Inkling-2 a token at a time: compiles each graph once, loads its parameters
//! once, then walks [`super::session::actions`] carrying the state between layers.
//!
//! **One graph per layer, 24 in all, not one per layer execution.** A step is 48
//! executions because the core runs three times, but a second compiled graph for a core
//! layer means a second copy of its 682 MB of expert weights: 8.2 GB that does not fit
//! beside everything else on a 31 GB board. The three passes differ only in their
//! inputs, `msg_scale` among them, so one graph serves them all.
//!
//! The block-state count a layer sees is fixed, which is what lets the graph be shared:
//! each later pass restores the hidden and block states the core was entered with, so a
//! core layer sees the same number of committed blocks on every pass.
//!
//! Prompt tokens go through the same one-token graphs as generated ones. Batching the
//! prompt would need a second set of graphs compiled for its length, and so a second
//! copy of every weight; at a few seconds a token that is the wrong trade here.

#![cfg(any(feature = "cuda", feature = "qemu", feature = "riscv"))]

use std::collections::HashMap;

use anyhow::Context;
use teeny_core::graph::{DtypeRepr, Graph, Op, SymTensor};
use teeny_core::model::LoweringMode;
use teeny_kernels::graph::TritonLowering;
use teeny_test::model::{Feed, RuntimeModel, runtime};

use super::config::Inkling2TextConfig;
use super::kernels::attention::PastLen;
use super::model::{LayerInputs, build_embed, build_head, build_layer, build_message_block};
use super::runner::{State, argmax, checkpoint_name, plan};
use super::session::{Action, actions, message_states};
use super::weights::Checkpoint;

/// A compiled graph with its outputs, ready to run.
struct Compiled<'d> {
    model: RuntimeModel<'d>,
    outs: Vec<usize>,
}

impl<'d> Compiled<'d> {
    /// Lowers `graph`, compiles it, and loads every parameter slot from `ck`.
    ///
    /// `layer` is what [`checkpoint_name`] resolves slot names against; graphs that are
    /// not a layer (the embedding, the head, the messages) pass 0, since their slots
    /// name top-level tensors.
    fn new(
        device: &runtime::Device<'d>,
        graph: &Graph,
        outputs: &[&SymTensor],
        rows: usize,
        ck: &mut Checkpoint,
        layer: usize,
    ) -> anyhow::Result<Self> {
        let lowering = TritonLowering::new();
        let (_, graph_to_dag) = lowering.lower_with_mapping(graph, LoweringMode::Inference)?;
        let target = runtime::default_target(device)?;
        let mut model = RuntimeModel::compile(
            device,
            graph,
            &lowering,
            LoweringMode::Inference,
            &target,
            rows,
            false,
        )?;
        let slots: Vec<(String, usize, usize)> = model.param_info_named().collect();
        let names: Vec<String> = slots.iter().map(|(n, _, _)| n.clone()).collect();
        let missing = super::load::missing(ck, &names, layer);
        anyhow::ensure!(
            missing.is_empty(),
            "layer {layer} has no checkpoint tensor for {missing:?} \
             (resolved through checkpoint_name, e.g. {:?})",
            missing
                .first()
                .map(|(slot, _)| checkpoint_name(slot, layer)),
        );
        let shapes: Vec<Vec<usize>> = slots
            .iter()
            .map(|(name, node, idx)| {
                model
                    .param_shape(*node, *idx)
                    .map(<[usize]>::to_vec)
                    .ok_or_else(|| anyhow::anyhow!("{name} has no declared shape"))
            })
            .collect::<anyhow::Result<_>>()?;
        let values = super::load::read(ck, &names, &shapes, layer)?;
        let by_name: HashMap<&str, &[f32]> = values
            .iter()
            .map(|(n, v)| (n.as_str(), v.as_slice()))
            .collect();
        for (name, node, idx) in &slots {
            model
                .load_param_f32(*node, *idx, by_name[name.as_str()])
                .with_context(|| format!("loading {name} of layer {layer}"))?;
        }
        Ok(Self {
            model,
            outs: outputs.iter().map(|t| graph_to_dag[t.node_id]).collect(),
        })
    }

    fn run(
        &mut self,
        device: &runtime::Device<'d>,
        feeds: &[&[f32]],
    ) -> anyhow::Result<Vec<Vec<f32>>> {
        self.model.run(device, feeds, &self.outs)
    }

    /// [`Self::run`] with each input either copied or read in place.
    fn run_feeds(
        &mut self,
        device: &runtime::Device<'d>,
        feeds: &[Feed<'_>],
    ) -> anyhow::Result<Vec<Vec<f32>>> {
        self.model.run_feeds(device, feeds, &self.outs)
    }
}

/// Builds a graph whose inputs are created in feed order.
struct Builder {
    graph: std::rc::Rc<std::cell::RefCell<Graph>>,
    first: SymTensor,
}

impl Builder {
    fn new(shape: &[usize]) -> Self {
        let (first, graph) =
            SymTensor::input(DtypeRepr::F32, shape.iter().map(|&d| Some(d)).collect());
        Self { graph, first }
    }

    fn input(&self, shape: &[usize]) -> SymTensor {
        let shape: Vec<Option<usize>> = shape.iter().map(|&d| Some(d)).collect();
        let node_id =
            self.graph
                .borrow_mut()
                .add_node(Op::Input, vec![], DtypeRepr::F32, shape.clone());
        SymTensor {
            node_id,
            graph: self.graph.clone(),
            dtype: DtypeRepr::F32,
            shape,
        }
    }
}

/// How many committed block states each layer sees, by walking the action sequence.
///
/// This has to agree across a core layer's three passes or the graph could not be
/// shared; the walk asserts that rather than assuming it.
pub fn block_counts(cfg: &Inkling2TextConfig) -> anyhow::Result<Vec<usize>> {
    let mut seen: Vec<Option<usize>> = vec![None; cfg.num_hidden_layers];
    let mut blocks = 0usize;
    let mut saved = 0usize;
    for action in actions(cfg) {
        match action {
            Action::SaveCore => saved = blocks,
            Action::RestoreCore => blocks = saved,
            Action::Messages(_) => {}
            Action::PushBlock(_) => blocks += 1,
            Action::Run(step) => match seen[step.layer] {
                None => seen[step.layer] = Some(blocks),
                Some(n) => anyhow::ensure!(
                    n == blocks,
                    "layer {} sees {n} block states on one pass and {blocks} on another, \
                     so one compiled graph cannot serve both",
                    step.layer
                ),
            },
        }
    }
    seen.into_iter()
        .enumerate()
        .map(|(l, n)| n.ok_or_else(|| anyhow::anyhow!("layer {l} never runs")))
        .collect()
}

/// A loaded model, ready to generate.
pub struct Engine<'d> {
    cfg: Inkling2TextConfig,
    layers: Vec<Compiled<'d>>,
    messages: Compiled<'d>,
    embed: Compiled<'d>,
    head: Compiled<'d>,
    past: PastLen,
    state: State,
    /// Tokens whose keys and values are in the cache.
    pub tokens: Vec<u32>,
}

impl<'d> Engine<'d> {
    /// Compiles every graph and loads every weight. This is the slow part: the
    /// parameters are around 22 GB for Inkling-2-Mini.
    pub fn load(
        device: &runtime::Device<'d>,
        cfg: Inkling2TextConfig,
        mup_width_multiplier: f64,
        ck: &mut Checkpoint,
        capacity: usize,
    ) -> anyhow::Result<Self> {
        let hid = cfg.hidden_size;
        let kv = cfg.num_key_value_heads * cfg.head_dim;
        let kw = cfg.sconv_kernel_size - 1;
        let blocks = block_counts(&cfg)?;
        let past = PastLen::default();
        let core = cfg.loop_config.core_start..cfg.loop_config.core_end;

        let embed = {
            let b = Builder::new(&[1]);
            let out = build_embed::<f32>(&cfg, &b.first);
            let g: &Graph = &b.graph.borrow();
            Compiled::new(device, g, &[&out], 1, ck, 0)?
        };

        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        for layer in 0..cfg.num_hidden_layers {
            let n_states = blocks[layer];
            let b = Builder::new(&[1, hid]);
            let boundary = layer % cfg.n_layers_per_attn_res_block == 0;
            let mlp_states = n_states + usize::from(boundary);
            let block_states = b.input(&[n_states, hid]);
            let mlp_block_states = b.input(&[mlp_states, hid]);
            let k_hist = b.input(&[kw, kv]);
            let v_hist = b.input(&[kw, kv]);
            let attn_hist = b.input(&[kw, hid]);
            let mlp_hist = b.input(&[kw, hid]);
            let k_cache = b.input(&[capacity, kv]);
            let v_cache = b.input(&[capacity, kv]);
            let msgs: Option<Vec<SymTensor>> = core
                .contains(&layer)
                .then(|| (0..4).map(|_| b.input(&[1, hid])).collect());
            let scale = msgs.as_ref().map(|_| b.input(&[1, hid]));
            let msg_refs: Option<[&SymTensor; 4]> =
                msgs.as_ref().map(|m| [&m[0], &m[1], &m[2], &m[3]]);
            let inputs = LayerInputs {
                x: &b.first,
                block_states: &block_states,
                n_states,
                mlp_block_states: &mlp_block_states,
                mlp_n_states: mlp_states,
                k_hist: &k_hist,
                v_hist: &v_hist,
                attn_hist: &attn_hist,
                mlp_hist: &mlp_hist,
                k_cache: &k_cache,
                v_cache: &v_cache,
                messages: msg_refs,
                msg_scale: scale.as_ref(),
                // One graph serves the whole sequence, so the bias has to be as wide
                // as the cache can ever be rather than as wide as it is today.
                bias_width: Some(capacity),
            };
            let out = build_layer::<f32>(&cfg, layer, &inputs, past.clone());
            let g: &Graph = &b.graph.borrow();
            let outs = [
                &out.out,
                &out.k,
                &out.v,
                &out.conv_in[0],
                &out.conv_in[1],
                &out.conv_in[2],
                &out.conv_in[3],
            ];
            layers.push(Compiled::new(device, g, &outs, 1, ck, layer)?);
        }

        let messages = {
            let n = message_states(&cfg).max(1);
            let b = Builder::new(&[n, hid]);
            let partial = b.input(&[1, hid]);
            let built = build_message_block::<f32>(&cfg, &b.first, &partial, n);
            let flat: Vec<&SymTensor> = built.iter().flatten().collect();
            let g: &Graph = &b.graph.borrow();
            Compiled::new(device, g, &flat, n, ck, 0)?
        };

        // The head sees the block states the last layer left behind: every boundary
        // the token crossed, which is one per `n_layers_per_attn_res_block` over the
        // whole stack plus the first.
        // Every boundary the token crossed is one block state. The core's later
        // passes restart from its entry state, so each boundary contributes once
        // however many times its layer runs.
        let per = cfg.n_layers_per_attn_res_block;
        let head_states = (0..cfg.num_hidden_layers).filter(|l| l % per == 0).count();
        let head = {
            let b = Builder::new(&[head_states.max(1), hid]);
            let partial = b.input(&[1, hid]);
            let out =
                build_head::<f32>(&cfg, &partial, &b.first, head_states, mup_width_multiplier);
            let g: &Graph = &b.graph.borrow();
            Compiled::new(device, g, &[&out], 1, ck, 0)?
        };

        let state = State::new(&cfg, capacity);
        Ok(Self {
            cfg,
            layers,
            messages,
            embed,
            head,
            past,
            state,
            tokens: Vec::new(),
        })
    }

    /// Runs one token through all 48 layer executions and returns its logits.
    pub fn step(&mut self, device: &runtime::Device<'d>, token: u32) -> anyhow::Result<Vec<f32>> {
        let hid = self.cfg.hidden_size;
        let kw = self.cfg.sconv_kernel_size - 1;
        let kv = self.cfg.num_key_value_heads * self.cfg.head_dim;
        self.past.set(self.state.len);
        // Block states belong to the token being run, not to the sequence: the
        // reference starts each forward pass with an empty stack and fills it at the
        // boundaries it crosses. Keeping the previous token's would both feed the
        // layer graphs a longer tensor than the shape they were compiled for and
        // retrieve from the wrong states. The KV cache and the convolution windows do
        // carry across tokens; these do not.
        self.state.reset_blocks(Vec::new());

        // INKLING2_TRACE names a file to append each layer execution's output to, in
        // order, for bisecting against the reference's own trace.
        let trace_to = std::env::var("INKLING2_TRACE").ok();
        let mut trace: Vec<f32> = Vec::new();
        // What a boundary commits is the layer's INPUT, not its output, and the push
        // happens after the layer has run, by which point `x` is the output.
        let mut layer_input: Vec<f32> = Vec::new();
        let mut dumped_messages = false;

        let ids = [token as f32];
        let mut x = self.embed.run(device, &[&ids])?.remove(0);

        // The messages of the pass being run, 12 loop layers by 4 each, and the states
        // the core was entered with.
        let core_start = self.cfg.loop_config.core_start;
        let mut messages: Vec<Vec<f32>> = Vec::new();
        let mut saved: Option<(Vec<f32>, Vec<Vec<f32>>)> = None;
        let zeros_msg = vec![0.0f32; hid];
        let ones = vec![1.0f32; hid];
        let zeros = vec![0.0f32; hid];

        for action in actions(&self.cfg) {
            match action {
                Action::SaveCore => saved = Some((x.clone(), self.state.blocks.clone())),
                Action::RestoreCore => {
                    let (hx, hb) = saved
                        .clone()
                        .expect("the core was saved before a later pass");
                    x = hx;
                    self.state.reset_blocks(hb);
                }
                Action::Messages(_) => {
                    // The reference retrieves from the blocks this pass appended, less
                    // its first, with the previous pass's final hidden state stacked on.
                    let want = message_states(&self.cfg).max(1);
                    let all = self.state.blocks.clone();
                    let take: Vec<f32> = all
                        .iter()
                        .rev()
                        .take(want)
                        .rev()
                        .flat_map(|b| b.iter().copied())
                        .collect();
                    let states = if take.len() == want * hid {
                        take
                    } else {
                        vec![0.0f32; want * hid]
                    };
                    messages = self.messages.run(device, &[&states, &x])?;
                    // Only the first pass's messages: the dump is a single file, and
                    // the reference dumps pass 1, so writing every pass leaves pass 2's
                    // behind and the comparison silently compares different passes.
                    if let (Some(path), true) = (&trace_to, !dumped_messages) {
                        dumped_messages = true;
                        // The first messages any layer receives, beside the state they
                        // were retrieved from, for bisecting against the reference.
                        let dump = |suffix: &str, v: &[f32]| {
                            let bytes: Vec<u8> = v.iter().flat_map(|f| f.to_le_bytes()).collect();
                            let _ = std::fs::write(path.replace(".bin", suffix), bytes);
                        };
                        if messages.len() >= 4 {
                            let flat: Vec<f32> = messages[..4].iter().flatten().copied().collect();
                            dump(".msg.bin", &flat);
                        }
                        dump(".prevx.bin", &x);
                        dump(".prevblocks.bin", &states);
                    }
                }
                Action::PushBlock(_) => self.state.push_block(&layer_input),
                Action::Run(step) => {
                    layer_input.clone_from(&x);
                    let slot = step.cache;
                    let n_states = self.state.n_blocks();
                    let block_in = self.state.block_input(1);
                    // At a boundary the MLP's stack already holds this layer's input;
                    // the host appends it here rather than the graph, because the
                    // attention before it must still see the shallower stack.
                    let mlp_in = if step.is_boundary {
                        let mut v = block_in.clone();
                        v.extend_from_slice(&x);
                        v
                    } else {
                        block_in.clone()
                    };
                    let conv = &self.state.conv[slot];
                    let (kc, vc) = (&self.state.k[slot], &self.state.v[slot]);
                    let mut feeds: Vec<&[f32]> = vec![
                        &x, &block_in, &mlp_in, &conv[0], &conv[1], &conv[2], &conv[3], kc, vc,
                    ];
                    let idx = step.layer - core_start;
                    let on_pass = step.pass.is_some_and(|p| p > 0);
                    if self.cfg.loop_config.core_start <= step.layer
                        && step.layer < self.cfg.loop_config.core_end
                    {
                        for m in 0..4 {
                            feeds.push(match (on_pass, messages.get(idx * 4 + m)) {
                                (true, Some(v)) => v.as_slice(),
                                _ => &zeros_msg,
                            });
                        }
                        feeds.push(if on_pass { &ones } else { &zeros });
                    }
                    anyhow::ensure!(
                        n_states * hid == block_in.len(),
                        "layer {} expected {n_states} block states, got {} floats",
                        step.layer,
                        block_in.len()
                    );
                    // The KV caches are the large inputs, 2 MB each at 512 tokens, and
                    // the layer only reads them: passed in place they cost nothing,
                    // where copying them was 200 MB a token.
                    let feeds: Vec<Feed<'_>> = feeds
                        .into_iter()
                        .enumerate()
                        .map(|(i, f)| if i == 7 || i == 8 { Feed::Borrow(f) } else { Feed::Copy(f) })
                        .collect();
                    let mut out = self.layers[step.layer].run_feeds(device, &feeds)?;
                    let (conv3, conv2) = (out.remove(6), out.remove(5));
                    let (conv1, conv0) = (out.remove(4), out.remove(3));
                    let (v_new, k_new) = (out.remove(2), out.remove(1));
                    x = out.remove(0);
                    self.state.append_kv(slot, &k_new, &v_new)?;
                    for (which, data) in [conv0, conv1, conv2, conv3].iter().enumerate() {
                        let width = if which < 2 { kv } else { hid };
                        self.state.roll_conv(slot, which, data, width);
                    }
                    if trace_to.is_some() {
                        trace.extend_from_slice(&x);
                    }
                    let _ = kw;
                }
            }
        }

        if let Some(path) = trace_to {
            // One file per position, so a sequence's traces do not overwrite one
            // another and it is never ambiguous which token a trace belongs to.
            let bytes: Vec<u8> = trace.iter().flat_map(|v| v.to_le_bytes()).collect();
            let at = self.state.len;
            for p in [path.replace(".bin", &format!(".{at}.bin")), path.clone()] {
                std::fs::write(&p, &bytes)
                    .with_context(|| format!("writing the execution trace to {p}"))?;
            }
        }
        self.state.len += 1;
        self.past.set(self.state.len);
        self.tokens.push(token);
        let head_blocks = self.state.block_input(1);
        Ok(self.head.run(device, &[&head_blocks, &x])?.remove(0))
    }

    /// The most likely next token, masked to the real vocabulary.
    ///
    /// The unembedding is padded out to `vocab_size` and those rows are not trained, so
    /// an unmasked argmax can land on a padding slot.
    pub fn pick(&self, logits: &[f32]) -> u32 {
        let n = self.cfg.unpadded_vocab_size;
        argmax(&logits[..n.min(logits.len())])
    }

    /// How many tokens are in the cache.
    pub fn len(&self) -> usize {
        self.state.len
    }

    /// Whether nothing has been fed yet.
    pub fn is_empty(&self) -> bool {
        self.state.len == 0
    }

    /// The plan this engine walks, for reporting.
    pub fn executions(&self) -> usize {
        plan(&self.cfg).len()
    }
}
