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

// TritonLowering test for Op::Attention.
//
// Op::Attention is a YOLO-specific op that requires a YoloLowering middleware
// (defined in vision-rs) to decompose into PSA sub-nodes. TritonLowering alone
// returns an error for this op, which this test verifies.

use teeny_core::{
    graph::{DtypeRepr, Op, SymTensor},
    model::{Lowering, LoweringMode},
};
use teeny_kernels::graph::TritonLowering;

fn build_attention_graph(c: usize, num_heads: usize, key_dim: usize) -> teeny_core::graph::Graph {
    let (x, graph) = SymTensor::input(DtypeRepr::F32, vec![Some(2), Some(c), Some(4), Some(4)]);
    let shape = x.shape.clone();
    let _ = x.graph.borrow_mut().add_node(
        Op::Attention {
            c,
            num_heads,
            key_dim,
        },
        vec![x.node_id],
        x.dtype,
        shape,
    );
    drop(x);
    std::rc::Rc::try_unwrap(graph).ok().unwrap().into_inner()
}

// TritonLowering must return an error for Op::Attention — use YoloLowering from
// vision-rs to lower models that contain C2PSA / attention blocks.
#[test]
fn test_attention_triton_lowering_returns_error() {
    let graph = build_attention_graph(128, 2, 32);
    let lowering = TritonLowering::new();
    let result = lowering.lower(&graph, LoweringMode::Inference);
    assert!(
        result.is_err(),
        "TritonLowering should return Err for Op::Attention; use YoloLowering instead"
    );
}
