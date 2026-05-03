/*
 * Copyright (c) 2026 Teenygrad.
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

// TritonLowering DAG tests for Op::Attention.
//
// These tests verify that TritonLowering correctly decomposes one Op::Attention
// into exactly 13 DAG sub-nodes (plus the original Input = 14 total):
//
//   1.  QKV Conv2d        [B, qkv_h, H, W]
//   2.  QKV BatchNorm2d   [B, qkv_h, H, W]
//   3.  Pack QKV          [4, BH, N, KEY_DIM]
//   4.  FA2 V_lo          [BH, N, KEY_DIM]
//   5.  FA2 V_hi          [BH, N, KEY_DIM]
//   6.  Merge attn        [B, c, H, W]
//   7.  Extract V NCHW    [B, c, H, W]
//   8.  PE DWConv         [B, c, H, W]
//   9.  PE BatchNorm2d    [B, c, H, W]
//   10. Add (attn + pe)   [B, c, H, W]
//   11. Proj Conv2d       [B, c, H, W]
//   12. Proj BatchNorm2d  [B, c, H, W]
//   13. Residual add      [B, c, H, W]
//
// No CUDA required — the Lowering trait operates on graph IR only.

use teeny_core::{
    graph::{DtypeRepr, Op, SymTensor},
    model::{Lowering, LoweringMode},
};
use teeny_kernels::graph::TritonLowering;

fn build_attention_graph(c: usize, num_heads: usize, key_dim: usize) -> teeny_core::graph::Graph {
    let (x, graph) = SymTensor::input(
        DtypeRepr::F32,
        vec![Some(2), Some(c), Some(4), Some(4)],
    );
    let shape = x.shape.clone();
    let node_id = x.graph.borrow_mut().add_node(
        Op::Attention { c, num_heads, key_dim },
        vec![x.node_id],
        x.dtype,
        shape,
    );
    let _ = node_id;
    // Return the owned graph by dropping the Rc ref.
    drop(x);
    std::rc::Rc::try_unwrap(graph).ok().unwrap().into_inner()
}

// ── DAG node count ────────────────────────────────────────────────────────────

#[test]
fn test_attention_lowering_dag_node_count() -> anyhow::Result<()> {
    // YOLO26n C2PSA parameters: c=128, num_heads=2, key_dim=32.
    let graph = build_attention_graph(128, 2, 32);
    let lowering = TritonLowering::new();
    let dag = lowering.lower(&graph, LoweringMode::Inference)?;

    // 1 Input + 13 sub-nodes = 14
    assert_eq!(dag.len(), 14, "expected 14 DAG nodes (Input + 13 Attention sub-nodes)");
    Ok(())
}

// ── Output shape of the residual-add node ─────────────────────────────────────

#[test]
fn test_attention_lowering_output_shape() -> anyhow::Result<()> {
    let c = 128;
    let graph = build_attention_graph(c, 2, 32);
    let lowering = TritonLowering::new();
    let dag = lowering.lower(&graph, LoweringMode::Inference)?;

    // The last DAG node (residual add, index 13) must have shape [2, c, 4, 4].
    let last = dag.node(dag.len() - 1);
    let shape = last.value.output_shape();
    assert_eq!(shape, &[Some(2), Some(c), Some(4), Some(4)],
        "output shape mismatch: {shape:?}");
    Ok(())
}

// ── Dynamic batch dimension propagates ───────────────────────────────────────

#[test]
fn test_attention_lowering_dynamic_batch() -> anyhow::Result<()> {
    let (x, graph) = SymTensor::input(
        DtypeRepr::F32,
        vec![None, Some(128), Some(4), Some(4)],
    );
    let shape = x.shape.clone();
    let _ = x.graph.borrow_mut().add_node(
        Op::Attention { c: 128, num_heads: 2, key_dim: 32 },
        vec![x.node_id],
        x.dtype,
        shape,
    );
    drop(x);
    let graph = std::rc::Rc::try_unwrap(graph).ok().unwrap().into_inner();

    let dag = TritonLowering::new().lower(&graph, LoweringMode::Inference)?;
    let last = dag.node(dag.len() - 1);
    assert_eq!(last.value.output_shape()[0], None, "batch dim should stay dynamic (None)");
    Ok(())
}
