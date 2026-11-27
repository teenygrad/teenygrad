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

/// What a worker needs to read one graph's parameters: plain data, so it can
/// cross to a loader thread while the graph itself stays with the device.
struct ReadJob {
    graph: usize,
    names: Vec<String>,
    shapes: Vec<Vec<usize>>,
    layer: usize,
    /// Each slot's parameter buffer as host memory, when the backend has it there:
    /// the worker writes the values in place, and only the rest come back.
    dst: Vec<Option<HostDst>>,
}

/// A parameter buffer's storage, handed to one loader thread to write.
///
/// The buffer belongs to a [`Pending`] graph the main thread holds and does not
/// touch until the job's result comes back, and its heap storage never moves, so
/// one thread at a time writes it.
struct HostDst {
    ptr: *mut f32,
    len: usize,
}

// Safety: see the type's documentation; the job is the only path to the buffer
// while it is out.
unsafe impl Send for HostDst {}

/// What a loader thread sends back for one graph: the values it could not write in
/// place, and how many it did.
struct ReadDone<'a> {
    graph: usize,
    rest: anyhow::Result<Vec<(String, std::borrow::Cow<'a, [f32]>)>>,
    written: usize,
}

impl ReadJob {
    /// Reads the job's slots and writes every one that has a host buffer in place.
    fn run<'a>(self, ck: &'a Checkpoint, private: &mut Option<Checkpoint>) -> ReadDone<'a> {
        let mut written = 0;
        let rest = super::load::read_shared(ck, private, &self.names, &self.shapes, self.layer).and_then(|values| {
            let mut rest = Vec::new();
            for ((slot, value), dst) in values.into_iter().zip(&self.dst) {
                match dst {
                    Some(d) => {
                        anyhow::ensure!(
                            value.len() == d.len,
                            "{slot}: read {} values for a buffer of {}",
                            value.len(),
                            d.len
                        );
                        // Safety: `HostDst` -- this thread is the buffer's only user
                        // until the result is sent.
                        unsafe { std::slice::from_raw_parts_mut(d.ptr, d.len) }.copy_from_slice(&value);
                        written += 1;
                    }
                    None => rest.push((slot, value)),
                }
            }
            prefault(&rest);
            Ok(rest)
        });
        ReadDone { graph: self.graph, rest, written }
    }
}

/// A compiled graph whose parameters have not been loaded yet.
struct Pending<'d> {
    model: RuntimeModel<'d>,
    outs: Vec<usize>,
    slots: Vec<(String, usize, usize)>,
    layer: usize,
}

/// Loader threads reading parameters while the main thread compiles. The reads are
/// page faults on the mapped checkpoint and the occasional repack, which a single
/// thread did at about 0.5 GB/s; eight cores share them.
const LOAD_WORKERS: usize = 6;

impl<'d> Pending<'d> {
    /// Lowers `graph` and compiles it. `layer` is what [`checkpoint_name`] resolves
    /// slot names against; graphs that are not a layer (the embedding, the head, the
    /// messages) pass 0, since their slots name top-level tensors.
    fn compile(
        device: &runtime::Device<'d>,
        graph: &Graph,
        outputs: &[&SymTensor],
        rows: usize,
        ck: &Checkpoint,
        layer: usize,
        index: usize,
    ) -> anyhow::Result<(Self, ReadJob)> {
        let lowering = TritonLowering::new();
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
        let graph_to_dag = model.graph_to_dag();
        let outs = outputs.iter().map(|t| graph_to_dag[t.node_id]).collect();
        let dst = slots
            .iter()
            .zip(&shapes)
            .map(|((_, node, idx), shape)| {
                let want: usize = shape.iter().product();
                Ok(model
                    .param_host_slice(*node, *idx)?
                    .filter(|s| s.len() == want)
                    .map(|s| HostDst { ptr: s.as_mut_ptr(), len: s.len() }))
            })
            .collect::<anyhow::Result<_>>()?;
        let job = ReadJob { graph: index, names, shapes, layer, dst };
        Ok((Self { model, outs, slots, layer }, job))
    }

    /// Copies what a worker could not write in place into the graph's parameter
    /// buffers, and checks that between them every slot was loaded.
    fn fill(&mut self, done: ReadDone<'_>) -> anyhow::Result<()> {
        let values = done.rest?;
        anyhow::ensure!(
            values.len() + done.written == self.slots.len(),
            "layer {}: {} slots, {} written in place and {} returned",
            self.layer,
            self.slots.len(),
            done.written,
            values.len()
        );
        let by_name: HashMap<&str, &[f32]> =
            values.iter().map(|(n, v)| (n.as_str(), v.as_ref())).collect();
        for (name, node, idx) in &self.slots {
            if let Some(v) = by_name.get(name.as_str()) {
                self.model
                    .load_param_f32(*node, *idx, v)
                    .with_context(|| format!("loading {name} of layer {}", self.layer))?;
            }
        }
        Ok(())
    }
}

/// Touches one byte in every page of each borrowed value, so a loader thread takes
/// the page faults (the disk reads) rather than the main thread's copy after it.
fn prefault(values: &[(String, std::borrow::Cow<'_, [f32]>)]) {
    let mut sink = 0u32;
    for (_, v) in values {
        if let std::borrow::Cow::Borrowed(slice) = v {
            for chunk in slice.chunks(1024) {
                // Safety: reading an element of a live slice.
                sink ^= unsafe { std::ptr::read_volatile(chunk.as_ptr()) }.to_bits();
            }
        }
    }
    std::hint::black_box(sink);
}

impl<'d> Compiled<'d> {
    fn run(
        &mut self,
        device: &runtime::Device<'d>,
        feeds: &[&[f32]],
    ) -> anyhow::Result<Vec<Vec<f32>>> {
        self.model.run(device, feeds, &self.outs)
    }

    /// Makes later runs process `rows` tokens; see [`RuntimeModel::set_rows`].
    fn set_rows(&mut self, rows: usize) -> anyhow::Result<()> {
        self.model.set_rows(rows)
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

    /// A graph whose first input is `shape`, `None` for a dimension that follows the
    /// row count: the token count, for a graph that runs on several tokens at once.
    fn new_dyn(shape: &[Option<usize>]) -> Self {
        let (first, graph) = SymTensor::input(DtypeRepr::F32, shape.to_vec());
        Self { graph, first }
    }

    fn input(&self, shape: &[usize]) -> SymTensor {
        let shape: Vec<Option<usize>> = shape.iter().map(|&d| Some(d)).collect();
        self.input_dyn(&shape)
    }

    fn input_dyn(&self, shape: &[Option<usize>]) -> SymTensor {
        let shape: Vec<Option<usize>> = shape.to_vec();
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

/// The most tokens one [`Engine::step_many`] processes: what every per-token graph
/// is compiled for. A compiled graph runs at any count up to this with one copy of
/// its weights, so this sizes only the activation buffers.
pub const MAX_TOKENS: usize = 16;

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
    /// Whether [`Self::step_many`] keeps what [`Self::truncate`] needs to undo it.
    /// Off by default: it costs a copy of every convolution window a step.
    keep_undo: bool,
    /// How to undo the last step, when [`Self::keep_undo`] is on.
    undo: Option<StepUndo>,
}

/// What a step changed that cannot be undone by shortening the sequence: the
/// convolution windows. The KV caches need no record, since attention reads only up
/// to the sequence length and the next step overwrites what lies past it.
struct StepUndo {
    /// The sequence length before the step.
    start: usize,
    /// Every cache slot's four windows before the step.
    conv_before: Vec<[Vec<f32>; 4]>,
    /// The rows each slot's windows were rolled forward by, per window, in order.
    conv_rows: Vec<[Vec<f32>; 4]>,
}

impl<'d> Engine<'d> {
    /// Keeps, from now on, what [`Self::truncate`] needs to take back the latest step's
    /// tokens. Speculative decoding turns this on; plain decoding has no need of it.
    pub fn keep_undo(&mut self, on: bool) {
        self.keep_undo = on;
        if !on {
            self.undo = None;
        }
    }

    /// Takes the sequence back to `len` tokens, within the latest step: as if that
    /// step had been given only its first `len - start` tokens.
    ///
    /// This is how rejected draft tokens are discarded. Fails if undo is not kept or
    /// `len` lies outside the latest step.
    pub fn truncate(&mut self, len: usize) -> anyhow::Result<()> {
        let undo = self
            .undo
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("truncate needs keep_undo before the step"))?;
        anyhow::ensure!(
            (undo.start..=self.state.len).contains(&len),
            "can only truncate to {}..={}, not {len}",
            undo.start,
            self.state.len
        );
        let keep = len - undo.start;
        let hid = self.cfg.hidden_size;
        let kv = self.cfg.num_key_value_heads * self.cfg.head_dim;
        for (slot, (before, rows)) in undo.conv_before.iter().zip(&undo.conv_rows).enumerate() {
            for which in 0..4 {
                let width = if which < 2 { kv } else { hid };
                self.state.conv[slot][which].clone_from(&before[which]);
                let kept = (keep * width).min(rows[which].len());
                self.state.roll_conv(slot, which, &rows[which][..kept], width);
            }
        }
        self.state.len = len;
        self.past.set(len);
        self.tokens.truncate(len);
        Ok(())
    }


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

        // Every graph that runs per token leaves the token count unknown (`None`) and
        // is compiled for MAX_TOKENS, so one step can take a whole chunk of a prompt.
        // Every shard mapped up front, so loader threads can lend tensors out of a
        // shared checkpoint while this thread compiles the next graph.
        ck.map_all()?;
        let ck: &Checkpoint = ck;
        let timing = std::env::var_os("INKLING2_LOAD_TIMING").is_some();
        let t_load = std::time::Instant::now();
        let (job_tx, job_rx) = std::sync::mpsc::channel::<ReadJob>();
        let job_rx = std::sync::Mutex::new(job_rx);
        let (embed, layers, messages, head) = std::thread::scope(|scope| -> anyhow::Result<_> {
            let (res_tx, res_rx) = std::sync::mpsc::channel::<ReadDone<'_>>();
            for _ in 0..LOAD_WORKERS {
                let (jobs, results) = (&job_rx, res_tx.clone());
                scope.spawn(move || {
                    let mut private = None;
                    loop {
                        let job = match jobs.lock().map(|rx| rx.recv()) {
                            Ok(Ok(job)) => job,
                            _ => break,
                        };
                        if results.send(job.run(ck, &mut private)).is_err() {
                            break;
                        }
                    }
                });
            }
            drop(res_tx);

            let mut pending: Vec<Option<Pending<'d>>> = Vec::new();
            let mut filled = 0usize;
            // Where the main thread's time goes, for INKLING2_LOAD_TIMING.
            let (t_compile, t_fill) = (std::cell::Cell::new(0.0f64), std::cell::Cell::new(0.0f64));
            let fill = |pending: &mut Vec<Option<Pending<'d>>>, done: ReadDone<'_>| -> anyhow::Result<()> {
                let p = pending[done.graph].as_mut().expect("a graph is compiled before it is read");
                let t = std::time::Instant::now();
                p.fill(done)?;
                t_fill.set(t_fill.get() + t.elapsed().as_secs_f64());
                Ok(())
            };
            let mut stage = |g: &Graph, outs: &[&SymTensor], layer: usize| -> anyhow::Result<usize> {
                let index = pending.len();
                let t = std::time::Instant::now();
                let (p, job) = Pending::compile(device, g, outs, MAX_TOKENS, ck, layer, index)?;
                t_compile.set(t_compile.get() + t.elapsed().as_secs_f64());
                pending.push(Some(p));
                job_tx.send(job).map_err(|_| anyhow::anyhow!("the loader threads stopped"))?;
                // Copy what the workers have finished meanwhile, so the pages they
                // faulted in are still cached when they are copied.
                while let Ok(done) = res_rx.try_recv() {
                    fill(&mut pending, done)?;
                    filled += 1;
                }
                Ok(index)
            };
            let t = None;
            let embed = {
                let b = Builder::new_dyn(&[t]);
                let out = build_embed::<f32>(&cfg, &b.first);
                let g: &Graph = &b.graph.borrow();
                stage(g, &[&out], 0)?
            };

            let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
            for layer in 0..cfg.num_hidden_layers {
                let n_states = blocks[layer];
                let b = Builder::new_dyn(&[t, Some(hid)]);
                let boundary = layer % cfg.n_layers_per_attn_res_block == 0;
                let mlp_states = n_states + usize::from(boundary);
                // [T, n, hidden]: the same bytes as the [T * n, hidden] the block
                // attention reads, with the token count leading so it follows the rows.
                let block_states = b.input_dyn(&[t, Some(n_states), Some(hid)]);
                let mlp_block_states = b.input_dyn(&[t, Some(mlp_states), Some(hid)]);
                let k_hist = b.input(&[kw, kv]);
                let v_hist = b.input(&[kw, kv]);
                let attn_hist = b.input(&[kw, hid]);
                let mlp_hist = b.input(&[kw, hid]);
                let k_cache = b.input(&[capacity, kv]);
                let v_cache = b.input(&[capacity, kv]);
                let msgs: Option<Vec<SymTensor>> = core
                    .contains(&layer)
                    .then(|| (0..4).map(|_| b.input_dyn(&[t, Some(hid)])).collect());
                let scale = msgs.as_ref().map(|_| b.input_dyn(&[t, Some(hid)]));
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
                layers.push(stage(g, &outs, layer)?);
            }

            let messages = {
                let n = message_states(&cfg).max(1);
                let b = Builder::new_dyn(&[t, Some(n), Some(hid)]);
                let partial = b.input_dyn(&[t, Some(hid)]);
                let built = build_message_block::<f32>(&cfg, &b.first, &partial, n);
                let flat: Vec<&SymTensor> = built.iter().flatten().collect();
                let g: &Graph = &b.graph.borrow();
                stage(g, &flat, 0)?
            };

            // The head sees the block states the last layer left behind: every boundary
            // the token crossed, which is one per `n_layers_per_attn_res_block` over the
            // whole stack plus the first.
            // Every boundary the token crossed is one block state. The core's later
            // passes restart from its entry state, so each boundary contributes once
            // however many times its layer runs.
            let per = cfg.n_layers_per_attn_res_block;
            let head_states = (0..cfg.num_hidden_layers).filter(|l| l % per == 0).count();
            // Per token like the layers, so a verify step can have every row's logits.
            let head = {
                let b = Builder::new_dyn(&[t, Some(head_states.max(1)), Some(hid)]);
                let partial = b.input_dyn(&[t, Some(hid)]);
                let out =
                    build_head::<f32>(&cfg, &partial, &b.first, head_states, mup_width_multiplier);
                let g: &Graph = &b.graph.borrow();
                stage(g, &[&out], 0)?
            };

            drop(stage);
            drop(job_tx);
            for done in res_rx.iter() {
                fill(&mut pending, done)?;
                filled += 1;
            }
            anyhow::ensure!(filled == pending.len(), "{filled} of {} graphs loaded", pending.len());
            let mut take = |i: usize| -> Compiled<'d> {
                let p = pending[i].take().expect("each graph is taken once");
                Compiled { model: p.model, outs: p.outs }
            };
            if timing {
                eprintln!(
                    "load: compiling {:.1} s, copying {:.1} s on the main thread",
                    t_compile.get(),
                    t_fill.get()
                );
            }
            let layers: Vec<Compiled<'d>> = layers.into_iter().map(&mut take).collect();
            Ok((take(embed), layers, take(messages), take(head)))
        })?;
        if timing {
            eprintln!("load: {:.1} s with {LOAD_WORKERS} loader threads", t_load.elapsed().as_secs_f64());
        }

        if let Ok(path) = std::env::var("INKLING2_DUMP_SHAPES") {
            let mut text = String::new();
            for (name, graph) in [("embed", &embed), ("messages", &messages), ("head", &head)] {
                for (node, shape) in graph.model.node_shapes() {
                    text += &format!("{name}\t{node}\t{shape:?}\n");
                }
            }
            for (i, graph) in layers.iter().enumerate() {
                for (node, shape) in graph.model.node_shapes() {
                    text += &format!("layer{i}\t{node}\t{shape:?}\n");
                }
            }
            std::fs::write(path, text)?;
        }
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
            keep_undo: false,
            undo: None,
        })
    }

    /// Runs one token through all 48 layer executions and returns its logits.
    pub fn step(&mut self, device: &runtime::Device<'d>, token: u32) -> anyhow::Result<Vec<f32>> {
        self.step_many(device, &[token])
    }

    /// Runs `tokens`, up to [`MAX_TOKENS`] of them, through all 48 layer executions
    /// at once, and returns the last one's logits.
    ///
    /// The same computation as [`Self::step`] on each token in turn: attention is
    /// causal within the step, and every per-token state (block states, messages) is
    /// kept per token. What changes is that each weight is read once for all of them,
    /// which is what makes a prompt fast.
    pub fn step_many(
        &mut self,
        device: &runtime::Device<'d>,
        tokens: &[u32],
    ) -> anyhow::Result<Vec<f32>> {
        let mut rows = self.step_rows(device, tokens, false)?;
        Ok(rows.pop().expect("a step has at least one token"))
    }

    /// [`Self::step_many`], returning every token's logits rather than the last's:
    /// what checking a run of drafted tokens needs.
    pub fn step_many_logits(
        &mut self,
        device: &runtime::Device<'d>,
        tokens: &[u32],
    ) -> anyhow::Result<Vec<Vec<f32>>> {
        self.step_rows(device, tokens, true)
    }

    fn step_rows(
        &mut self,
        device: &runtime::Device<'d>,
        tokens: &[u32],
        all_rows: bool,
    ) -> anyhow::Result<Vec<Vec<f32>>> {
        let t = tokens.len();
        anyhow::ensure!(
            (1..=MAX_TOKENS).contains(&t),
            "a step takes 1 to {MAX_TOKENS} tokens, not {t}"
        );
        for graph in self.layers.iter_mut().chain([&mut self.embed, &mut self.messages]) {
            graph.set_rows(t)?;
        }
        self.undo = self.keep_undo.then(|| StepUndo {
            start: self.state.len,
            conv_before: self.state.conv.clone(),
            conv_rows: vec![Default::default(); self.state.conv.len()],
        });
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

        let ids: Vec<f32> = tokens.iter().map(|&id| id as f32).collect();
        let mut x = self.embed.run(device, &[&ids])?.remove(0);

        // The messages of the pass being run, 12 loop layers by 4 each, and the states
        // the core was entered with.
        let core_start = self.cfg.loop_config.core_start;
        let mut messages: Vec<Vec<f32>> = Vec::new();
        let mut saved: Option<(Vec<f32>, Vec<Vec<f32>>)> = None;
        let zeros_msg = vec![0.0f32; t * hid];
        let ones = vec![1.0f32; t * hid];
        let zeros = vec![0.0f32; t * hid];

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
                    let n = self.state.blocks.len();
                    let states = if n >= want {
                        interleave_blocks(&self.state.blocks[n - want..], t, hid)
                    } else {
                        vec![0.0f32; t * want * hid]
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
                    let block_in = self.state.block_input(t);
                    // At a boundary the MLP's stack already holds this layer's input;
                    // the host appends it here rather than the graph, because the
                    // attention before it must still see the shallower stack. Each
                    // token's own input goes on its own stack.
                    let mlp_in = if step.is_boundary {
                        let mut stack = self.state.blocks.clone();
                        stack.push(x.clone());
                        interleave_blocks(&stack, t, hid)
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
                        t * n_states * hid == block_in.len(),
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
                        if let Some(undo) = self.undo.as_mut() {
                            undo.conv_rows[slot][which].clone_from(data);
                        }
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
            // The step's tokens and the block states the head reads, so a host-side
            // reference can rebuild the final-norm hidden and line traces up with ids.
            let ids: Vec<u8> = tokens.iter().flat_map(|v| v.to_le_bytes()).collect();
            let blocks: Vec<u8> = interleave_blocks(&self.state.blocks, t, hid)
                .iter()
                .chain(x.iter())
                .flat_map(|v| v.to_le_bytes())
                .collect();
            for (suffix, data) in [(".ids.bin", &ids), (".blocks.bin", &blocks)] {
                let p = path.replace(".bin", &format!(".{at}{suffix}"));
                std::fs::write(&p, data).with_context(|| format!("writing {p}"))?;
            }
        }
        self.state.len += t;
        self.past.set(self.state.len);
        self.tokens.extend_from_slice(tokens);
        if all_rows {
            self.head.set_rows(t)?;
            let head_blocks = interleave_blocks(&self.state.blocks, t, hid);
            let logits = self.head.run(device, &[&head_blocks, &x])?.remove(0);
            let vocab = logits.len() / t;
            return Ok(logits.chunks(vocab).map(<[f32]>::to_vec).collect());
        }
        // Only the last token's logits are wanted: its block states and hidden state.
        self.head.set_rows(1)?;
        let last = |v: &[f32]| v[(t - 1) * hid..t * hid].to_vec();
        let head_blocks: Vec<f32> = self.state.blocks.iter().flat_map(|b| last(b)).collect();
        Ok(vec![self.head.run(device, &[&head_blocks, &last(&x)])?.remove(0)])
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

/// `blocks`, each `[T, hidden]` (one row per token), as `[T, blocks, hidden]`: each
/// token's states together, which is how the block attention reads a stack.
fn interleave_blocks(blocks: &[Vec<f32>], tokens: usize, hidden: usize) -> Vec<f32> {
    let n = blocks.len();
    let mut out = vec![0.0f32; tokens * n * hidden];
    for (s, block) in blocks.iter().enumerate() {
        for t in 0..tokens {
            let to = (t * n + s) * hidden;
            out[to..to + hidden].copy_from_slice(&block[t * hidden..(t + 1) * hidden]);
        }
    }
    out
}
