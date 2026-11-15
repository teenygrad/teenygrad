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

//! Rewrites of imported ONNX ops into ones the lowering already runs.

use std::collections::BTreeMap;

use teeny_core::graph::{ConstantTensor, Graph, Op};

/// Rewrites every `Gemm(A, B[, C])` as `MatMul(A', B') [+ C]`, where `A'` and `B'` are
/// `A` and `B` behind a `Transpose` when `transA` / `transB` ask for one.
///
/// The lowering has one kernel for `MatMul` and `Gemm` alike and reads neither
/// transpose flag nor the bias, so a `Gemm` from a classifier head (`transB = 1`, bias
/// set: PyTorch's `nn.Linear`) would contract over the wrong axis or drop its bias.
/// Expressed in ops that are lowered (and, for a constant weight, folded: the transposed
/// weight becomes an ordinary constant), it runs as ONNX defines it.
///
/// The `Gemm` node's own index becomes the final node of its replacement, so every
/// consumer keeps reading it. `alpha` and `beta` other than 1 are refused: exporters
/// emit 1, and a scale would need a constant operand this IR does not carry.
///
/// Returns how many `Gemm` nodes were rewritten.
pub fn decompose_gemm(graph: &mut Graph) -> Result<usize, String> {
    let mut count = 0;
    for g in 0..graph.nodes.len() {
        let Op::Gemm {
            alpha,
            beta,
            trans_a,
            trans_b,
        } = graph.nodes[g].op.clone()
        else {
            continue;
        };
        let label = graph
            .names
            .get(&g)
            .cloned()
            .unwrap_or_else(|| format!("#{g}"));
        let inputs = graph.nodes[g].inputs.clone();
        if inputs.len() < 2 {
            return Err(format!(
                "Gemm '{label}': needs at least two operands, has {}",
                inputs.len()
            ));
        }
        let has_bias = inputs.len() >= 3;
        if alpha != 1.0 || (has_bias && beta != 1.0) {
            return Err(format!(
                "Gemm '{label}': alpha {alpha} / beta {beta} are not supported (only 1)"
            ));
        }
        let (dtype, shape) = (graph.nodes[g].dtype, graph.nodes[g].shape.clone());

        let operand = |i: usize, transpose: bool, graph: &mut Graph| -> usize {
            if !transpose {
                return i;
            }
            let src = &graph.nodes[i];
            let shape = match src.shape.as_slice() {
                [r, c] => vec![*c, *r],
                other => other.iter().rev().copied().collect(),
            };
            let dtype = src.dtype;
            graph.add_node(Op::Transpose { perm: vec![1, 0] }, vec![i], dtype, shape)
        };
        let a = operand(inputs[0], trans_a, graph);
        let b = operand(inputs[1], trans_b, graph);

        if has_bias {
            let mm = graph.add_node(Op::MatMul, vec![a, b], dtype, shape);
            if let Some(name) = graph.names.get(&g).cloned() {
                graph.names.insert(mm, format!("{name}/matmul"));
            }
            let node = &mut graph.nodes[g];
            node.op = Op::Add;
            node.inputs = vec![mm, inputs[2]];
        } else {
            let node = &mut graph.nodes[g];
            node.op = Op::MatMul;
            node.inputs = vec![a, b];
        }
        count += 1;
    }
    Ok(count)
}

/// Rewrites `Pow(x, c)` for a constant scalar `c` of 1, 2, 3, 0.5 or -1 as exact ops:
/// `x`, `x * x`, `x * x * x`, `Sqrt(x)`, `Reciprocal(x)`.
///
/// Exporters emit these for RMSNorm's mean square (`x^2`) and tanh-GELU's cube
/// (`x^3`). The `Pow` kernel computes `exp(c * ln x)`, which is NaN for every negative
/// `x` whatever `c` is, and reads its exponent element-wise rather than broadcasting a
/// scalar, so these are both more exact and correct where the kernel is not. Any other
/// `Pow` is left for the lowering.
///
/// Returns how many were rewritten.
pub fn simplify_pow(graph: &mut Graph, constants: &BTreeMap<usize, ConstantTensor>) -> usize {
    let mut count = 0;
    for g in 0..graph.nodes.len() {
        if !matches!(graph.nodes[g].op, Op::Pow) {
            continue;
        }
        let &[x, e] = graph.nodes[g].inputs.as_slice() else {
            continue;
        };
        let Some(c) = constants
            .get(&e)
            .and_then(ConstantTensor::as_f64)
            .and_then(|v| (v.len() == 1).then(|| v[0]))
        else {
            continue;
        };
        let (dtype, shape) = (graph.nodes[g].dtype, graph.nodes[g].shape.clone());
        // The lowered DAG keeps one edge per distinct parent, so `Mul(x, x)` would reach
        // its kernel with a single operand; the second factor is a copy of `x`.
        let copy = |graph: &mut Graph| {
            let xs = graph.nodes[x].shape.clone();
            graph.add_node(Op::Identity, vec![x], dtype, xs)
        };
        let (op, inputs) = if c == 1.0 {
            (Op::Identity, vec![x])
        } else if c == 2.0 {
            (Op::Mul, vec![x, copy(graph)])
        } else if c == 3.0 {
            let x2 = copy(graph);
            let sq = graph.add_node(Op::Mul, vec![x, x2], dtype, shape.clone());
            (Op::Mul, vec![sq, x])
        } else if c == 0.5 {
            (Op::Sqrt, vec![x])
        } else if c == -1.0 {
            (Op::Reciprocal, vec![x])
        } else {
            continue;
        };
        let node = &mut graph.nodes[g];
        node.op = op;
        node.inputs = inputs;
        count += 1;
    }
    count
}
