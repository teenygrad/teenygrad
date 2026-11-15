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

//! The Gemma 3 decoder (`Gemma3ForCausalLM`) as a teeny graph that processes one
//! step of `T` new tokens against a KV cache.
//!
//! Every weight slot is named by its `state_dict` key (`model.layers.3.mlp.up_proj.weight`),
//! so checkpoint tensors load by name. The output projection is tied to
//! `model.embed_tokens`: its slot is `lm_head.weight`, which loads the same tensor.
//!
//! Step graph inputs, in creation (and therefore feed) order:
//!
//! | index | input | shape |
//! |---|---|---|
//! | 0 | `ids` (token ids as f32) | `[T]` |
//! | 1, 2 | `cos_local`, `sin_local` (sliding layers' RoPE) | `[T, head_dim]` |
//! | 3, 4 | `cos_global`, `sin_global` (full layers' RoPE) | `[T, head_dim]` |
//! | 5 + 2l, 6 + 2l | `k_cache_l`, `v_cache_l` | `[cache, kv_heads * head_dim]` |
//!
//! Outputs: `logits` `[1, vocab]` (the last token's), then `k_l`, `v_l` per layer
//! `[T, kv_heads * head_dim]`, the step's keys (normalised and rotated) and values
//! for the host to append to the cache.

use teeny_core::dtype::FloatBytes;
use teeny_core::graph::{CustomData, DtypeRepr, Graph, Op, Shape, SymTensor};
use teeny_core::name_scope::name_scope;

use crate::config::GemmaConfig;
use crate::kernels::{
    attention::{AttentionSpec, GemmaAttentionOp, PastLen},
    embed::GemmaEmbedOp,
    gelu_mul::GemmaGeluMulOp,
    last_row::GemmaLastRowOp,
    linear::Linear4Op,
    rmsnorm::GemmaRmsNormOp,
    rope::GemmaRopeOp,
};

/// Lanes along `K` in the linear kernel.
const LINEAR_BLOCK: i32 = 128;

fn wrap(like: &SymTensor, node_id: usize, shape: Shape) -> SymTensor {
    SymTensor {
        node_id,
        graph: like.graph.clone(),
        dtype: like.dtype,
        shape,
    }
}

/// Adds another `Input` placeholder to `like`'s graph.
fn input_like(like: &SymTensor, shape: &[usize]) -> SymTensor {
    let shape: Shape = shape.iter().map(|&d| Some(d)).collect();
    let node_id =
        like.graph
            .borrow_mut()
            .add_node(Op::Input, vec![], DtypeRepr::F32, shape.clone());
    wrap(like, node_id, shape)
}

fn add(a: &SymTensor, b: &SymTensor) -> SymTensor {
    assert_eq!(a.shape, b.shape, "add: operand shapes differ");
    let shape = a.shape.clone();
    let node_id =
        a.graph
            .borrow_mut()
            .add_node(Op::Add, vec![a.node_id, b.node_id], a.dtype, shape.clone());
    wrap(a, node_id, shape)
}

fn linear<D: FloatBytes + Send + Sync + 'static>(
    name: &str,
    x: &SymTensor,
    k: usize,
    n: usize,
) -> SymTensor {
    let _g = name_scope(name);
    x.record_custom(
        CustomData::new(Linear4Op::<D>::new(k, n, LINEAR_BLOCK)),
        &[],
        None,
    )
}

fn rmsnorm<D: FloatBytes + Send + Sync + 'static>(
    name: &str,
    x: &SymTensor,
    row: usize,
) -> SymTensor {
    let _g = name_scope(name);
    x.record_custom(CustomData::new(GemmaRmsNormOp::<D>::new(row)), &[], None)
}

/// The inputs of one step graph, in feed order.
pub struct StepInputs {
    /// `(name, shape)` per input.
    pub shapes: Vec<(String, Vec<usize>)>,
}

/// The step graph's inputs for `t` new tokens and a cache of `cache` rows.
pub fn step_inputs(cfg: &GemmaConfig, t: usize, cache: usize) -> StepInputs {
    let hd = cfg.head_dim;
    let kv = cfg.num_key_value_heads * hd;
    let mut shapes = vec![
        ("ids".to_string(), vec![t]),
        ("cos_local".to_string(), vec![t, hd]),
        ("sin_local".to_string(), vec![t, hd]),
        ("cos_global".to_string(), vec![t, hd]),
        ("sin_global".to_string(), vec![t, hd]),
    ];
    for l in 0..cfg.num_hidden_layers {
        shapes.push((format!("k_cache_{l}"), vec![cache, kv]));
        shapes.push((format!("v_cache_{l}"), vec![cache, kv]));
    }
    StepInputs { shapes }
}

/// A recorded step graph.
pub struct StepGraph {
    /// The graph.
    pub graph: std::rc::Rc<std::cell::RefCell<Graph>>,
    /// Outputs: `logits`, then `k_l`, `v_l` per layer.
    pub outputs: Vec<(String, SymTensor)>,
}

/// Records one step of `t` tokens against a `cache`-row KV cache. Every attention op
/// reads the past length from `past`.
pub fn build_step<D: FloatBytes + Send + Sync + 'static>(
    cfg: &GemmaConfig,
    t: usize,
    cache: usize,
    past: &PastLen,
) -> StepGraph {
    let ins = step_inputs(cfg, t, cache);
    let (ids, graph) = SymTensor::input(DtypeRepr::F32, vec![Some(t)]);
    let rest: Vec<SymTensor> = ins.shapes[1..]
        .iter()
        .map(|(_, s)| input_like(&ids, s))
        .collect();
    let (cos_l, sin_l, cos_g, sin_g) = (&rest[0], &rest[1], &rest[2], &rest[3]);

    let (hid, hd) = (cfg.hidden_size, cfg.head_dim);
    let (nh, nkv) = (cfg.num_attention_heads, cfg.num_key_value_heads);
    let mut outputs = Vec::new();

    let _model = name_scope("model");
    let mut x = {
        let _g = name_scope("embed_tokens");
        ids.record_custom(
            CustomData::new(GemmaEmbedOp::<D>::new(cfg.vocab_size, hid)),
            &[],
            None,
        )
    };

    let mut kv_out = Vec::new();
    for l in 0..cfg.num_hidden_layers {
        let _layer = name_scope(format!("layers.{l}"));
        let sliding = cfg.is_sliding(l);
        let (cos, sin) = if sliding {
            (cos_l, sin_l)
        } else {
            (cos_g, sin_g)
        };

        // Attention block.
        let h = rmsnorm::<D>("input_layernorm", &x, hid);
        let attn = {
            let _a = name_scope("self_attn");
            let q = linear::<D>("q_proj", &h, hid, nh * hd);
            let k = linear::<D>("k_proj", &h, hid, nkv * hd);
            let v = linear::<D>("v_proj", &h, hid, nkv * hd);
            let q = rmsnorm::<D>("q_norm", &q, hd);
            let k = rmsnorm::<D>("k_norm", &k, hd);
            let q = q.record_custom(
                CustomData::new(GemmaRopeOp::<D>::new(nh, hd)),
                &[cos, sin],
                None,
            );
            let k = k.record_custom(
                CustomData::new(GemmaRopeOp::<D>::new(nkv, hd)),
                &[cos, sin],
                None,
            );
            let spec = AttentionSpec {
                heads: nh,
                kv_heads: nkv,
                head_dim: hd,
                window: sliding.then_some(cfg.sliding_window),
                scale: cfg.query_pre_attn_scalar.powf(-0.5) as f32,
            };
            let o = q.record_custom(
                CustomData::new(GemmaAttentionOp::<D>::new(spec, past.clone())),
                &[&k, &v, &rest[4 + 2 * l], &rest[5 + 2 * l]],
                None,
            );
            kv_out.push((format!("k_{l}"), k));
            kv_out.push((format!("v_{l}"), v));
            linear::<D>("o_proj", &o, nh * hd, hid)
        };
        let attn = rmsnorm::<D>("post_attention_layernorm", &attn, hid);
        x = add(&x, &attn);

        // MLP block.
        let h = rmsnorm::<D>("pre_feedforward_layernorm", &x, hid);
        let mlp = {
            let _m = name_scope("mlp");
            let gate = linear::<D>("gate_proj", &h, hid, cfg.intermediate_size);
            let up = linear::<D>("up_proj", &h, hid, cfg.intermediate_size);
            let act = gate.record_custom(
                CustomData::new(GemmaGeluMulOp::<D>::default()),
                &[&up],
                None,
            );
            linear::<D>("down_proj", &act, cfg.intermediate_size, hid)
        };
        let mlp = rmsnorm::<D>("post_feedforward_layernorm", &mlp, hid);
        x = add(&x, &mlp);
    }

    let x = rmsnorm::<D>("norm", &x, hid);
    drop(_model);
    let last = x.record_custom(CustomData::new(GemmaLastRowOp::<D>::default()), &[], None);
    let logits = linear::<D>("lm_head", &last, hid, cfg.vocab_size);

    outputs.push(("logits".to_string(), logits));
    outputs.extend(kv_out);
    StepGraph { graph, outputs }
}
