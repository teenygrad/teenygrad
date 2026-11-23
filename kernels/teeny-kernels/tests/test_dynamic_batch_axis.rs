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

//! A softmax or reduction over the innermost axis needs no transpose, so a dynamic
//! extent in front of it (a batch axis) must not stop the graph lowering. Only a
//! transpose needs every extent, because it has to copy the tensor. The traced MNIST
//! graph, `[None, 10]` into a softmax, was refused before this.

use teeny_core::graph::{DtypeRepr, Graph, Op, Shape};
use teeny_core::model::LoweringMode;
use teeny_kernels::graph::TritonLowering;

fn unary(op: Op, input: Shape, output: Shape) -> Graph {
    let mut g = Graph::new();
    let x = g.add_node(Op::Input, vec![], DtypeRepr::F32, input);
    g.add_node(op, vec![x], DtypeRepr::F32, output);
    g
}

fn lower(g: &Graph) -> anyhow::Result<usize> {
    let (dag, _) = TritonLowering::new().lower_with_mapping(g, LoweringMode::Inference)?;
    Ok(dag.len())
}

#[test]
fn test_an_innermost_softmax_lowers_with_a_dynamic_batch() {
    for op in [Op::Softmax { dim: -1 }, Op::LogSoftmax { axis: 1 }] {
        let shape: Shape = vec![None, Some(10)];
        let g = unary(op.clone(), shape.clone(), shape);
        // The input and the softmax: no transposes around it.
        assert_eq!(lower(&g).unwrap_or_else(|e| panic!("{op:?}: {e}")), 2);
    }
}

#[test]
fn test_an_innermost_reduction_lowers_with_a_dynamic_batch() {
    let g = unary(
        Op::ReduceMean {
            axes: vec![-1],
            keepdims: true,
            noop_with_empty_axes: false,
        },
        vec![None, Some(4), Some(10)],
        vec![None, Some(4), Some(1)],
    );
    assert_eq!(lower(&g).expect("an innermost reduction needs no copy"), 2);
}

#[test]
fn test_an_outer_softmax_with_a_dynamic_extent_still_asks_for_one() {
    let shape: Shape = vec![None, Some(3), Some(10)];
    let g = unary(Op::Softmax { dim: 1 }, shape.clone(), shape);
    let err = lower(&g)
        .map(|_| ())
        .expect_err("the transpose needs every extent");
    assert!(err.to_string().contains("materialize"), "{err}");
}
