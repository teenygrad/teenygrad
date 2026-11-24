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

//! The token loop: walks [`super::runner::plan`], carrying state between layers and
//! passes, and emits one token at a time.
//!
//! A step is 48 layer executions from 24 sets of weights. Each is its own compiled
//! graph, cached by `(layer, tokens)`, so prompt processing and decoding compile two
//! graphs per distinct layer and then reuse them: the past length reaches the
//! kernels through a shared [`PastLen`] rather than through the graph's shape.
//!
//! Between layers the host carries the hidden states, the block states and the
//! convolution windows; between tokens it carries those and the KV caches. The three
//! passes of the looped core are siblings, not a chain: each restarts from the hidden
//! and block states as they were before the core, and takes messages built from the
//! previous pass's output.

use std::collections::HashMap;

use super::config::Inkling2TextConfig;
use super::kernels::attention::PastLen;
use super::runner::{State, Step, argmax, plan};

/// Which graph a step needs: the layer, and how many tokens it processes.
///
/// Two layers of the same index in different passes share a graph; only the caches
/// and the messages differ, and both are inputs rather than shape.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct GraphKey {
    /// The model layer.
    pub layer: usize,
    /// Tokens in this step.
    pub tokens: usize,
    /// Whether the layer takes messages from a previous pass, which changes the
    /// graph: the controllers are recorded only then.
    pub messages: bool,
}

/// How many distinct graphs a sequence compiles.
///
/// Prompt processing and decoding differ in token count, and the looped core's later
/// passes add the controllers, so a full sequence needs at most
/// `2 * (layers + core_layers)` graphs, not `2 * 48`.
pub fn graph_keys(cfg: &Inkling2TextConfig, prompt_tokens: usize) -> Vec<GraphKey> {
    let mut keys = Vec::new();
    let mut seen = std::collections::BTreeSet::new();
    for tokens in [prompt_tokens, 1] {
        if tokens == 0 {
            continue;
        }
        for step in plan(cfg) {
            let key = GraphKey {
                layer: step.layer,
                tokens,
                messages: step.takes_messages,
            };
            if seen.insert((key.layer, key.tokens, key.messages)) {
                keys.push(key);
            }
        }
    }
    keys
}

/// How many committed block states the cross-pass message stack attends over.
///
/// The reference slices `block_states[num_pre_loop_blocks + 1:]`: the blocks a pass
/// appended, less its first. With the partial (the previous pass's final hidden state)
/// stacked on top, that is what the messages are retrieved from.
pub fn message_states(cfg: &Inkling2TextConfig) -> usize {
    let l = &cfg.loop_config;
    let per = cfg.n_layers_per_attn_res_block;
    let in_core = (l.core_start..l.core_end).filter(|layer| layer % per == 0).count();
    // The pre-core blocks are skipped by the slice, along with one more.
    in_core.saturating_sub(1)
}

/// The order a step walks, with the saved states the looped core restarts from.
///
/// Returned rather than executed so the sequencing can be tested without a device:
/// the bookkeeping is where this gets subtle, not the arithmetic.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Action {
    /// Run a layer's graph.
    Run(Step),
    /// Append the layer's input to the block states, which a boundary layer does
    /// after running.
    PushBlock(usize),
    /// Remember the hidden and block states, before the first pass of the core.
    SaveCore,
    /// Restore them, which every later pass starts with.
    RestoreCore,
    /// Build the messages for pass `n` from the previous pass's output.
    Messages(usize),
}

/// The actions one step performs, in order.
pub fn actions(cfg: &Inkling2TextConfig) -> Vec<Action> {
    let l = &cfg.loop_config;
    let mut out = Vec::new();
    let mut saved = false;
    let mut last_pass = None;
    for step in plan(cfg) {
        if step.pass.is_some() && !saved {
            out.push(Action::SaveCore);
            saved = true;
        }
        if let Some(p) = step.pass {
            if last_pass != Some(p) && p > 0 {
                out.push(Action::RestoreCore);
                out.push(Action::Messages(p));
            }
            last_pass = Some(p);
        }
        out.push(Action::Run(step));
        if step.is_boundary {
            out.push(Action::PushBlock(step.layer));
        }
    }
    let _ = l;
    out
}

/// A sequence in progress: the host state, the past length the kernels read, and the
/// graphs compiled so far.
pub struct Session {
    /// The model's text configuration.
    pub cfg: Inkling2TextConfig,
    /// Caches, convolution windows and block states.
    pub state: State,
    /// Shared with every attention and relative-bias op.
    pub past: PastLen,
    /// Tokens produced so far, prompt included.
    pub tokens: Vec<u32>,
    compiled: HashMap<GraphKey, ()>,
}

impl Session {
    /// A session over a sequence of at most `capacity` tokens.
    pub fn new(cfg: Inkling2TextConfig, capacity: usize) -> Self {
        let state = State::new(&cfg, capacity);
        Self {
            cfg,
            state,
            past: PastLen::default(),
            tokens: Vec::new(),
            compiled: HashMap::new(),
        }
    }

    /// Records that a graph has been compiled, and whether it was already.
    pub fn note_compiled(&mut self, key: GraphKey) -> bool {
        self.compiled.insert(key, ()).is_none()
    }

    /// How many graphs are compiled.
    pub fn compiled_count(&self) -> usize {
        self.compiled.len()
    }

    /// Advances the cache cursor and the shared past length by `n` tokens.
    ///
    /// Called once per step, after every layer has run: all 48 executions of a step
    /// attend over the same past, and the cursor must not move under them.
    pub fn advance(&mut self, n: usize) {
        self.state.len += n;
        self.past.set(self.state.len);
    }

    /// Picks the next token and records it.
    pub fn take(&mut self, logits: &[f32]) -> u32 {
        let id = argmax(logits);
        self.tokens.push(id);
        id
    }

    /// Whether `id` ends generation.
    pub fn is_eos(&self, id: u32, eos: u32) -> bool {
        let _ = self;
        id == eos
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg() -> Inkling2TextConfig {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures/inkling2-mini-config.json");
        super::super::Inkling2Config::load(&path)
            .expect("config")
            .text_config
    }

    #[test]
    fn test_a_step_saves_the_core_once_and_restores_it_per_later_pass() {
        let c = cfg();
        let acts = actions(&c);
        let saves = acts.iter().filter(|a| **a == Action::SaveCore).count();
        let restores = acts.iter().filter(|a| **a == Action::RestoreCore).count();
        assert_eq!(saves, 1, "the core's entry state is saved once");
        assert_eq!(
            restores,
            c.loop_config.num_passes - 1,
            "every pass after the first restarts from it"
        );
        // The save comes before the first core layer and after the last pre-core one.
        let save_at = acts
            .iter()
            .position(|a| *a == Action::SaveCore)
            .expect("a save");
        let first_core = acts
            .iter()
            .position(|a| matches!(a, Action::Run(s) if s.pass == Some(0)))
            .expect("a core layer");
        assert!(
            save_at < first_core,
            "the state is saved before the core runs"
        );
    }

    #[test]
    fn test_the_message_stack_is_the_pass_blocks_less_the_first() {
        let c = cfg();
        // The core spans layers 6..18 with a boundary every 6, so a pass appends two
        // blocks (at 6 and 12) and the messages see the second alone, plus the
        // previous pass's final hidden state as the partial.
        assert_eq!(message_states(&c), 1);
    }

    #[test]
    fn test_messages_are_built_once_per_later_pass_before_its_layers() {
        let c = cfg();
        let acts = actions(&c);
        let msgs: Vec<usize> = acts
            .iter()
            .filter_map(|a| match a {
                Action::Messages(p) => Some(*p),
                _ => None,
            })
            .collect();
        assert_eq!(
            msgs,
            vec![1, 2],
            "one message build per pass after the first"
        );
        // Each precedes that pass's first layer.
        for p in [1usize, 2] {
            let at = acts
                .iter()
                .position(|a| *a == Action::Messages(p))
                .expect("a message build");
            let layer = acts
                .iter()
                .position(|a| matches!(a, Action::Run(s) if s.pass == Some(p)))
                .expect("a pass layer");
            assert!(at < layer, "messages for pass {p} come before its layers");
        }
    }

    #[test]
    fn test_block_states_are_pushed_at_boundaries_only() {
        let c = cfg();
        let acts = actions(&c);
        let pushes: Vec<usize> = acts
            .iter()
            .filter_map(|a| match a {
                Action::PushBlock(l) => Some(*l),
                _ => None,
            })
            .collect();
        // A boundary every n_layers_per_attn_res_block layers, over 48 executions.
        let expected = plan(&c).iter().filter(|s| s.is_boundary).count();
        assert_eq!(pushes.len(), expected);
        assert!(
            pushes
                .iter()
                .all(|l| l % c.n_layers_per_attn_res_block == 0),
            "only boundary layers push: {pushes:?}"
        );
        // Each push follows its own layer's run.
        for (i, a) in acts.iter().enumerate() {
            if let Action::PushBlock(l) = a {
                assert!(
                    matches!(acts[i - 1], Action::Run(s) if s.layer == *l),
                    "a push must follow that layer's run"
                );
            }
        }
    }

    #[test]
    fn test_a_sequence_compiles_two_graphs_per_distinct_layer_shape() {
        let c = cfg();
        let keys = graph_keys(&c, 16);
        // 24 layers at prompt width and again at one token, plus the core's layers a
        // second time each because their later passes take messages.
        let core = c.loop_config.core_end - c.loop_config.core_start;
        assert_eq!(keys.len(), 2 * (c.num_hidden_layers + core));
        assert_eq!(keys.len(), 72, "not 2 * 48: passes share a graph");
        // Decoding alone compiles half of them.
        let decode_only = graph_keys(&c, 0);
        assert_eq!(decode_only.len(), c.num_hidden_layers + core);
    }

    #[test]
    fn test_the_cursor_moves_once_a_step_not_once_a_layer() {
        let c = cfg();
        let mut s = Session::new(c, 64);
        assert_eq!(s.past.get(), 0);
        s.advance(5); // a five-token prompt
        assert_eq!(s.state.len, 5);
        assert_eq!(s.past.get(), 5, "the shared past follows the cursor");
        s.advance(1);
        assert_eq!(s.past.get(), 6);
    }

    #[test]
    fn test_graphs_are_compiled_once() {
        let c = cfg();
        let mut s = Session::new(c, 16);
        let k = GraphKey {
            layer: 7,
            tokens: 1,
            messages: true,
        };
        assert!(s.note_compiled(k), "the first time is new");
        assert!(!s.note_compiled(k), "the second is not");
        assert_eq!(s.compiled_count(), 1);
    }

    #[test]
    fn test_take_records_the_token_it_picks() {
        let c = cfg();
        let eos = 200006u32;
        let mut s = Session::new(c, 16);
        let mut logits = vec![0.0f32; 16];
        logits[3] = 1.0;
        assert_eq!(s.take(&logits), 3);
        assert_eq!(s.tokens, vec![3]);
        assert!(!s.is_eos(3, eos));
        assert!(s.is_eos(eos, eos));
    }
}
