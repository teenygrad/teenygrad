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

use teeny_core::graph::{ConstantMap, ConstantTensor, DtypeRepr, Graph, Op};

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

/// Makes `And`, `Or`, `Xor`, comparison and `Where` operands the output's shape, and a `Where`
/// condition the output's dtype, by inserting `Expand` and `Cast` nodes.
///
/// Their kernels read every operand at the output's own offsets, so ONNX's broadcasting
/// (an attention mask `[1, 1, 1, S]` against `[1, 1, S, S]`) would read past the
/// smaller operand; `Expand` lowers to a strided copy that does broadcast. The `Where`
/// kernel also reads its condition in the output's dtype, so a Bool condition (one byte
/// per element) is cast to it first; arithmetic forms of `Where` are avoided because a
/// mask of `-inf` turns them into NaN.
///
/// Runs after materialising, when shapes are known: an operand or output whose shape is
/// not fully static is left alone. Each `Expand`'s target shape is a new constant,
/// added to `constants`. Returns how many nodes were inserted.
pub fn expand_for_broadcast(graph: &mut Graph, constants: &mut ConstantMap) -> usize {
    let static_shape = |s: &[Option<usize>]| -> Option<Vec<usize>> { s.iter().copied().collect() };
    let mut inserted = 0;
    for g in 0..graph.nodes.len() {
        let is_where = matches!(graph.nodes[g].op, Op::Where);
        if !(is_where
            || matches!(
                graph.nodes[g].op,
                Op::And
                    | Op::Or
                    | Op::Xor
                    | Op::Equal
                    | Op::Greater
                    | Op::GreaterOrEqual
                    | Op::Less
                    | Op::LessOrEqual
            ))
        {
            continue;
        }
        let Some(out) = static_shape(&graph.nodes[g].shape) else {
            continue;
        };
        let out_dtype = graph.nodes[g].dtype;
        let mut inputs = graph.nodes[g].inputs.clone();
        for (pos, input) in inputs.iter_mut().enumerate() {
            let src = &graph.nodes[*input];
            let Some(shape) = static_shape(&src.shape) else {
                continue;
            };
            let dtype = src.dtype;
            // A rank-0 constant (Where's `0` and `-inf`, say) becomes a constant of the
            // output's shape outright: Expand cannot take a rank-0 operand, since an
            // empty shape here also means "unknown".
            if shape.is_empty() && !out.is_empty() {
                if let Some(scalar) = constants.get(input).cloned() {
                    let n: usize = out.iter().product();
                    let full: Vec<Option<usize>> = out.iter().map(|&d| Some(d)).collect();
                    let filled = graph.add_node(
                        Op::Constant {
                            dtype,
                            shape: full.clone(),
                        },
                        vec![],
                        dtype,
                        full.clone(),
                    );
                    constants.insert(
                        filled,
                        ConstantTensor {
                            dtype,
                            shape: full,
                            bytes: scalar.bytes.repeat(n),
                        },
                    );
                    *input = filled;
                    inserted += 1;
                    continue;
                }
                continue;
            }
            if shape != out {
                // The target shape, as the constant Expand's second operand.
                let dims: Vec<u8> = out.iter().flat_map(|&d| (d as i64).to_le_bytes()).collect();
                let shape_node = graph.add_node(
                    Op::Constant {
                        dtype: DtypeRepr::I64,
                        shape: vec![Some(out.len())],
                    },
                    vec![],
                    DtypeRepr::I64,
                    vec![Some(out.len())],
                );
                constants.insert(
                    shape_node,
                    ConstantTensor {
                        dtype: DtypeRepr::I64,
                        shape: vec![Some(out.len())],
                        bytes: dims,
                    },
                );
                let full: Vec<Option<usize>> = out.iter().map(|&d| Some(d)).collect();
                *input = graph.add_node(Op::Expand, vec![*input, shape_node], dtype, full);
                inserted += 2;
            }
            if is_where && pos == 0 && dtype == DtypeRepr::Bool && out_dtype != DtypeRepr::Bool {
                let full: Vec<Option<usize>> = out.iter().map(|&d| Some(d)).collect();
                *input = graph.add_node(Op::Cast { to: out_dtype }, vec![*input], out_dtype, full);
                inserted += 1;
            }
        }
        graph.nodes[g].inputs = inputs;
    }
    inserted
}

/// Gives each repeated operand of a node (`Mul(x, x)`) its own `Identity` copy.
///
/// The lowered DAG keeps one edge per distinct parent, so a node reading the same tensor
/// twice reaches its kernel with one operand fewer than it takes. Returns how many copies
/// were inserted.
pub fn dedupe_operands(graph: &mut Graph) -> usize {
    let mut inserted = 0;
    for g in 0..graph.nodes.len() {
        let mut inputs = graph.nodes[g].inputs.clone();
        let mut seen = Vec::new();
        for input in inputs.iter_mut() {
            if seen.contains(input) {
                let src = &graph.nodes[*input];
                let (dtype, shape) = (src.dtype, src.shape.clone());
                *input = graph.add_node(Op::Identity, vec![*input], dtype, shape);
                inserted += 1;
            }
            seen.push(*input);
        }
        graph.nodes[g].inputs = inputs;
    }
    inserted
}

/// Computes each comparison returning Bool at its operands' dtype, then casts to Bool.
///
/// ONNX comparisons return bools, but their kernels write `0`/`1` in the operands'
/// dtype; a node recorded as Bool would give them a one-byte buffer they write four-byte
/// values into. The comparison becomes a node at the operands' dtype, and the original
/// node a `Cast` of it to Bool, so consumers keep reading the same node. Returns how many
/// were rewritten.
pub fn compare_then_cast(graph: &mut Graph) -> usize {
    let mut count = 0;
    for g in 0..graph.nodes.len() {
        let op = graph.nodes[g].op.clone();
        if !matches!(
            op,
            Op::Equal | Op::Greater | Op::GreaterOrEqual | Op::Less | Op::LessOrEqual
        ) || graph.nodes[g].dtype != DtypeRepr::Bool
        {
            continue;
        }
        let Some(&first) = graph.nodes[g].inputs.first() else {
            continue;
        };
        let operand_dtype = graph.nodes[first].dtype;
        if operand_dtype == DtypeRepr::Bool {
            continue;
        }
        let inputs = graph.nodes[g].inputs.clone();
        let shape = graph.nodes[g].shape.clone();
        let cmp = graph.add_node(op, inputs, operand_dtype, shape);
        if let Some(name) = graph.names.get(&g).cloned() {
            graph.names.insert(cmp, format!("{name}/compare"));
        }
        let node = &mut graph.nodes[g];
        node.op = Op::Cast {
            to: DtypeRepr::Bool,
        };
        node.inputs = vec![cmp];
        count += 1;
    }
    count
}
