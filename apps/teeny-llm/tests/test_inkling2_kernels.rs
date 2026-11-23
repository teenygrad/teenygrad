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

//! Inkling-2's own kernels, each as a one-op graph compiled through teenyc, run on the
//! device (CUDA, or RISC-V under qemu) and compared with a host reference of the
//! formula in the kernel's module docs. Needs no model weights.

#![cfg(all(feature = "inkling2", any(feature = "cuda", feature = "qemu")))]
use std::cell::RefCell;
use std::rc::Rc;

use teeny_core::graph::{CustomData, CustomOp, DtypeRepr, Graph, Op, SymTensor};
use teeny_core::model::LoweringMode;
use teeny_core::name_scope::name_scope;
use teeny_kernels::graph::TritonLowering;
use teeny_llm_core::models::inkling2::kernels::sconv::SconvOp;
use teeny_test::model::{RuntimeModel, runtime};

/// Deterministic values in `[-1, 1)`.
fn values(n: usize, seed: u64) -> Vec<f32> {
    let mut s = seed
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    (0..n)
        .map(|_| {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((s >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
        })
        .collect()
}

/// A graph whose inputs are created in feed order.
struct OneOp {
    graph: Rc<RefCell<Graph>>,
    first: SymTensor,
}

impl OneOp {
    fn new(shape: &[usize]) -> Self {
        let (first, graph) =
            SymTensor::input(DtypeRepr::F32, shape.iter().map(|&d| Some(d)).collect());
        Self { graph, first }
    }

    // Used by the multi-input kernels this file will grow; kept so the harness
    // stays identical to test_llm_kernels.rs.
    #[allow(dead_code)]
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

    /// Records `op` on the first input plus `others`, compiles, loads `params` (in slot
    /// order), runs on `feeds` and returns the op's output.
    fn run(
        &self,
        op: impl CustomOp + 'static,
        others: &[&SymTensor],
        feeds: &[&[f32]],
        params: &[&[f32]],
    ) -> Vec<f32> {
        // Parameter slots are named after the node's scope, as the model names its weights.
        let out = {
            let _scope = name_scope("op");
            self.first.record_custom(CustomData::new(op), others, None)
        };
        let graph: &Graph = &self.graph.borrow();
        let lowering = TritonLowering::new();
        let (_, graph_to_dag) = lowering
            .lower_with_mapping(graph, LoweringMode::Inference)
            .expect("lowering the one-op graph");
        let device = runtime::open().expect("opening the device");
        let target = runtime::default_target(&device).expect("the device's target");
        let rows = self.first.shape[0].unwrap_or(1);
        let mut model = RuntimeModel::compile(
            &device,
            graph,
            &lowering,
            LoweringMode::Inference,
            &target,
            rows,
            false,
        )
        .expect("compiling the op through teenyc");
        let slots: Vec<(String, usize, usize)> = model.param_info_named().collect();
        assert_eq!(slots.len(), params.len(), "parameter slots: {slots:?}");
        for ((_, node, idx), data) in slots.iter().zip(params) {
            model
                .load_param_f32(*node, *idx, data)
                .expect("loading a parameter");
        }
        let mut got = model
            .run(&device, feeds, &[graph_to_dag[out.node_id]])
            .expect("running the op");
        got.remove(0)
    }
}

/// Asserts `got` matches `want` to `tol`, relative to each value's magnitude (at least 1).
fn assert_close(what: &str, got: &[f32], want: &[f32], tol: f32) {
    assert_eq!(got.len(), want.len(), "{what}: output length");
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        assert!(
            (g - w).abs() <= tol * w.abs().max(1.0),
            "{what}: element {i} is {g}, expected {w} (tolerance {tol})"
        );
    }
}

/// `y[t, c] = sum(k) w[c, k] * x[t + k, c]`, a causal depthwise conv of width 4.
///
/// The input carries its own history: `T + K - 1` rows in, `T` rows out.
#[test]
fn test_sconv_matches_a_host_convolution() {
    const K: usize = 4;
    let (t, c) = (5usize, 300usize);
    let x = values((t + K - 1) * c, 11);
    let w = values(c * K, 12);
    let want: Vec<f32> = (0..t * c)
        .map(|i| {
            let (row, ch) = (i / c, i % c);
            (0..K).map(|k| w[ch * K + k] * x[(row + k) * c + ch]).sum()
        })
        .collect();
    let g = OneOp::new(&[t + K - 1, c]);
    let got = g.run(SconvOp::<f32>::new(c, K), &[], &[&x], &[&w]);
    assert_close("sconv", &got, &want, 1e-5);
}

/// One token with no history is the filter's last tap alone, which is what decoding
/// does at the very start of a sequence.
#[test]
fn test_sconv_with_zero_history_keeps_only_the_last_tap() {
    const K: usize = 4;
    let c = 128usize;
    let mut x = vec![0.0f32; K * c];
    let row = values(c, 13);
    x[(K - 1) * c..].copy_from_slice(&row); // the only non-zero token
    let w = values(c * K, 14);
    let want: Vec<f32> = (0..c).map(|ch| w[ch * K + (K - 1)] * row[ch]).collect();
    let g = OneOp::new(&[K, c]);
    let got = g.run(SconvOp::<f32>::new(c, K), &[], &[&x], &[&w]);
    assert_close("sconv first token", &got, &want, 1e-5);
}

/// A channel count that is not a multiple of the block, so the tail is masked.
#[test]
fn test_sconv_masks_a_ragged_channel_tail() {
    const K: usize = 4;
    let (t, c) = (2usize, 257usize); // BLOCK is 256
    let x = values((t + K - 1) * c, 15);
    let w = values(c * K, 16);
    let want: Vec<f32> = (0..t * c)
        .map(|i| {
            let (row, ch) = (i / c, i % c);
            (0..K).map(|k| w[ch * K + k] * x[(row + k) * c + ch]).sum()
        })
        .collect();
    let g = OneOp::new(&[t + K - 1, c]);
    let got = g.run(SconvOp::<f32>::new(c, K), &[], &[&x], &[&w]);
    assert_close("sconv ragged", &got, &want, 1e-5);
}
