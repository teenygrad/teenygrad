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

//! Resolving `If` nodes whose condition is known at compile time.
//!
//! An exported model is full of branches the author never wrote. Tracing a
//! `x.squeeze(1)` emits `if x.shape[1] == 1 { squeeze(x, 1) } else { x }`, because the
//! trace cannot know the extent. In a static export that extent *is* known, so the
//! branch is decidable and only one side of it is real code.
//!
//! Resolving it has to happen in three stages, and the order is forced:
//!
//! 1. the import keeps both branch bodies without importing either
//!    ([`crate::onnx::IfBranches`]),
//! 2. [`crate::materialize`] folds the condition to a constant,
//! 3. this pass imports the taken body and makes the `If` a pass-through of its result.
//!
//! The reason the bodies cannot both be imported at step 1, with a select afterwards,
//! is that a guarded branch is usually only *valid* under its guard: the squeeze above
//! is an error whenever the condition is false. Importing both would put an operation
//! in the graph that cannot run, and lowering has no way to know it must not.
//!
//! The reason the decision cannot be made at step 1 either is that the condition is
//! computed, not stored. It is `Equal(Gather(Shape(x), 1), 1)` and so on: a constant
//! only in the sense that folding can reach it.
//!
//! A branch body may itself contain an `If`, whose own condition only becomes constant
//! once the body is in the graph and folded. This pass therefore does one round and
//! reports what it did; a caller with nested branches alternates it with
//! [`crate::materialize::materialize_constants`] until it reports no progress. See
//! [`resolve_if_branches`].

use std::collections::BTreeMap;

use teeny_core::graph::{ConstantMap, Op};

use crate::errors::Result;
use crate::materialize::{
    MaterializeConfig, MaterializeReport, lowering_constants, materialize_constants,
};
use crate::onnx::{Import, ImportCtx};

/// What one round of [`resolve_if_branches`] did.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct BranchReport {
    /// `If` nodes replaced by their taken branch.
    pub resolved: usize,
    /// `If` nodes left alone because the condition did not fold to a value.
    pub unresolved_condition: usize,
    /// `If` nodes left alone because they have more than one output, which the
    /// single-output graph IR cannot represent.
    pub declined_multi_output: usize,
}

impl BranchReport {
    /// Whether this round changed the graph, and so whether another round, after
    /// folding again, could resolve more.
    pub fn made_progress(&self) -> bool {
        self.resolved > 0
    }
}

/// Replaces every `If` whose condition is a known constant with its taken branch.
///
/// `folded` is the constant map from [`crate::materialize::materialize_constants`];
/// values already in `import.tensors` are consulted too, so a condition that was
/// constant in the file needs no folding pass to be seen.
///
/// The `If` node itself is kept, rewritten to an `Op::Identity` reading the branch's
/// result, rather than deleted with its consumers rewired. Keeping it means every
/// index already recorded against it stays valid, and the node had to become something
/// lowerable in any case. The cost is one copy.
///
/// Branch nodes are appended, so they sit after their consumer in `graph.nodes`. That
/// is sound because the graph is addressed by [`teeny_core::graph::Graph::topological_sort`]
/// rather than by index order, and the result is still acyclic: a branch body can read
/// the enclosing scope but never the `If`'s own output.
pub fn resolve_if_branches(import: &mut Import, folded: &ConstantMap) -> Result<BranchReport> {
    let mut report = BranchReport::default();
    if import.branches.is_empty() {
        return Ok(report);
    }

    // Resolving an `If` consumes its entry, so take the whole set and put back the
    // ones this round could not do.
    let pending = std::mem::take(&mut import.branches);
    let mut deferred = BTreeMap::new();

    for (if_id, branches) in pending {
        let Some(&cond_id) = import.graph.nodes[if_id].inputs.first() else {
            report.unresolved_condition += 1;
            deferred.insert(if_id, branches);
            continue;
        };

        // A condition folded this round lives in `folded`; one that was constant in
        // the file lives in `tensors`.
        let cond = folded
            .get(&cond_id)
            .or_else(|| import.tensors.get(&cond_id))
            .and_then(|t| t.as_i64())
            .and_then(|v| v.first().copied());
        let Some(cond) = cond else {
            report.unresolved_condition += 1;
            deferred.insert(if_id, branches);
            continue;
        };

        // An ONNX branch body's outputs are the `If`'s outputs. A node here has one
        // output, so a multi-output `If` cannot be represented and is left as it is
        // rather than silently reduced to its first output.
        if branches.taken(cond != 0).output.len() != 1 {
            report.declined_multi_output += 1;
            deferred.insert(if_id, branches);
            continue;
        }

        let out_id = {
            let body = branches.taken(cond != 0);
            let mut ctx = ImportCtx::reopen(
                std::mem::take(&mut import.graph),
                std::mem::take(&mut import.tensors),
                std::mem::take(&mut import.value_to_node),
                import.value_info_map.clone(),
                body,
                import.opset,
            )?;
            // A branch body has no inputs of its own, per the ONNX spec, but it may
            // carry initializers.
            ctx.add_initializers_and_inputs(body)?;
            for node in &body.node {
                ctx.import_node(node)?;
            }
            let out_name = &body.output[0].name;
            let out_id = ctx.value_to_node.get(out_name).copied();
            import.graph = ctx.graph;
            import.tensors = ctx.tensors;
            import.value_to_node = ctx.value_to_node;
            import.value_info_map = ctx.value_info_map;

            match out_id {
                Some(id) => id,
                None => {
                    // The body names an output no node in it produces. That is a
                    // malformed subgraph; leave the `If` alone so the failure surfaces
                    // as an unlowerable op rather than a silently wrong graph.
                    report.unresolved_condition += 1;
                    deferred.insert(if_id, branches);
                    continue;
                }
            }
        };

        let (dtype, shape) = {
            let out = &import.graph.nodes[out_id];
            (out.dtype, out.shape.clone())
        };
        let node = &mut import.graph.nodes[if_id];
        node.op = Op::Identity;
        node.inputs = vec![out_id];
        node.dtype = dtype;
        node.shape = shape;
        report.resolved += 1;
    }

    import.branches = deferred;
    Ok(report)
}

/// Folds constants and resolves branches, repeating until neither makes progress, and
/// returns the constant map lowering should be given.
///
/// This is the entry point to prefer. Doing the two passes by hand works, but the
/// seeding is easy to get wrong in a way that produces a graph lowering rejects
/// rather than an obvious error:
///
/// * a later folding round has to be seeded with the *merged* map, because an earlier
///   round rewrote the nodes it folded to `Op::Constant` while their values stayed in
///   its return value, and
/// * resolving a branch *adds* constants to `import.tensors`, since the body brings
///   its own, so a seed captured before resolution is already stale.
///
/// Rounds are needed at all because a branch body can contain another `If`, whose own
/// condition cannot fold until the body is in the graph.
pub fn materialize_and_resolve(
    import: &mut Import,
    config: &MaterializeConfig,
) -> Result<(ConstantMap, MaterializeReport, BranchReport)> {
    crate::rewrite::decompose_gemm(&mut import.graph)
        .map_err(crate::errors::Error::InvalidModel)?;
    crate::rewrite::simplify_pow(&mut import.graph, &import.tensors);
    let mut folds = ConstantMap::new();
    let mut totals = BranchReport::default();
    // The first round's folding report is the one worth returning: it describes
    // folding the imported graph. A later round sees a graph most of whose constants
    // are already folded, so its counts describe only the branch bodies.
    let mut first_report: Option<MaterializeReport> = None;
    loop {
        // `import.tensors` is re-read every round: resolution adds the taken body's
        // own constants to it.
        let mut seed = import.tensors.clone();
        seed.extend(folds.iter().map(|(k, v)| (*k, v.clone())));

        let (folded, materialize_report) = materialize_constants(&mut import.graph, &seed, config);
        let materialize_report = first_report.get_or_insert(materialize_report).clone();
        folds.extend(folded.iter().map(|(k, v)| (*k, v.clone())));

        let report = resolve_if_branches(import, &folded)?;
        totals.resolved += report.resolved;
        totals.unresolved_condition = report.unresolved_condition;
        totals.declined_multi_output = report.declined_multi_output;

        if !report.made_progress() {
            let all = lowering_constants(&import.graph, &seed, &folds);
            return Ok((all, materialize_report, totals));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Onnx;
    use crate::onnx::onnx_proto3::{
        AttributeProto, GraphProto, ModelProto, NodeProto, OperatorSetIdProto, TensorProto,
        TensorShapeProto, TypeProto, ValueInfoProto, attribute_proto, tensor_proto,
        tensor_shape_proto,
    };
    use protobuf::Message;

    fn i64_tensor(name: &str, vals: &[i64]) -> TensorProto {
        let mut t = TensorProto::new();
        t.name = name.to_string();
        t.data_type = tensor_proto::DataType::INT64 as i32;
        t.dims = vec![vals.len() as i64];
        t.int64_data = vals.to_vec();
        t
    }

    fn node(op: &str, name: &str, inputs: &[&str], outputs: &[&str]) -> NodeProto {
        let mut n = NodeProto::new();
        n.op_type = op.to_string();
        n.name = name.to_string();
        n.input = inputs.iter().map(|s| s.to_string()).collect();
        n.output = outputs.iter().map(|s| s.to_string()).collect();
        n
    }

    fn constant_node(name: &str, out: &str, vals: &[i64]) -> NodeProto {
        let mut n = node("Constant", name, &[], &[out]);
        let mut a = AttributeProto::new();
        a.name = "value".to_string();
        a.type_ = protobuf::EnumOrUnknown::new(attribute_proto::AttributeType::TENSOR);
        a.t = Some(i64_tensor("", vals)).into();
        n.attribute.push(a);
        n
    }

    fn graph_attr(name: &str, g: GraphProto) -> AttributeProto {
        let mut a = AttributeProto::new();
        a.name = name.to_string();
        a.type_ = protobuf::EnumOrUnknown::new(attribute_proto::AttributeType::GRAPH);
        a.g = Some(g).into();
        a
    }

    /// A float-typed `ValueInfoProto`, optionally with a static shape. The importer
    /// rejects an untyped one, so even a value whose shape does not matter needs a
    /// type here.
    fn value_info(name: &str, dims: Option<&[i64]>) -> ValueInfoProto {
        let mut v = ValueInfoProto::new();
        v.name = name.to_string();
        let mut ty = TypeProto::new();
        let tt = ty.mut_tensor_type();
        tt.elem_type = tensor_proto::DataType::FLOAT as i32;
        if let Some(dims) = dims {
            let mut shape = TensorShapeProto::new();
            for d in dims {
                let mut dim = tensor_shape_proto::Dimension::new();
                dim.set_dim_value(*d);
                shape.dim.push(dim);
            }
            tt.shape = Some(shape).into();
        }
        v.type_ = Some(ty).into();
        v
    }

    /// A model shaped like a traced `squeeze(dim)` guard.
    ///
    /// `x` is an input of the given shape; the condition is
    /// `Equal(Gather(Shape(x), 1), 1)`, which is how an exporter writes "is axis 1 a
    /// unit axis". The then-branch squeezes axis 1, the else-branch passes `x`
    /// through, and the result is negated so the `If` has a consumer whose wiring can
    /// be checked.
    fn guard_model(dims: &[i64]) -> Vec<u8> {
        let mut g = GraphProto::new();

        g.input.push(value_info("x", Some(dims)));
        g.output.push(value_info("out", None));

        g.node.push(node("Shape", "/Shape", &["x"], &["shape"]));
        g.node.push(constant_node("/axis", "axis", &[1]));
        g.node
            .push(node("Gather", "/Gather", &["shape", "axis"], &["dim1"]));
        g.node.push(constant_node("/one", "one", &[1]));
        g.node
            .push(node("Equal", "/Equal", &["dim1", "one"], &["cond"]));

        let mut then_g = GraphProto::new();
        then_g.output.push(value_info("then_out", None));
        then_g.node.push(constant_node("/sq_axes", "sq_axes", &[1]));
        then_g.node.push(node(
            "Squeeze",
            "/Squeeze",
            &["x", "sq_axes"],
            &["then_out"],
        ));

        let mut else_g = GraphProto::new();
        else_g.output.push(value_info("else_out", None));
        else_g
            .node
            .push(node("Identity", "/Identity", &["x"], &["else_out"]));

        let mut if_node = node("If", "/If", &["cond"], &["if_out"]);
        if_node.attribute.push(graph_attr("then_branch", then_g));
        if_node.attribute.push(graph_attr("else_branch", else_g));
        g.node.push(if_node);
        g.node.push(node("Neg", "/Neg", &["if_out"], &["out"]));

        let mut m = ModelProto::new();
        m.graph = Some(g).into();
        let mut op = OperatorSetIdProto::new();
        op.version = 17;
        m.opset_import.push(op);
        m.write_to_bytes().expect("serialise")
    }

    /// Runs the public pipeline and returns the import, the constants lowering should
    /// be given, and the branch report.
    fn resolve_to_fixpoint(bytes: &[u8]) -> (Import, ConstantMap, BranchReport) {
        let mut import = Onnx::import_from_reader(&mut &bytes[..]).expect("import");
        let (all, _, branch) =
            materialize_and_resolve(&mut import, &MaterializeConfig::default()).expect("resolve");
        (import, all, branch)
    }

    /// Materialises, then resolves, returning the import and both reports.
    fn import_and_resolve(bytes: &[u8]) -> (Import, BranchReport) {
        let mut import = Onnx::import_from_reader(&mut &bytes[..]).expect("import");
        let (folded, _) = materialize_constants(
            &mut import.graph,
            &import.tensors.clone(),
            &MaterializeConfig::default(),
        );
        let report = resolve_if_branches(&mut import, &folded).expect("resolve");
        (import, report)
    }

    /// The `If` node, and what it now reads.
    fn if_node(import: &Import) -> (&Op, Vec<&Op>) {
        let id = *import
            .graph
            .names
            .iter()
            .find(|(_, n)| *n == "/If")
            .expect("the If node keeps its name")
            .0;
        let node = &import.graph.nodes[id];
        (
            &node.op,
            node.inputs
                .iter()
                .map(|&i| &import.graph.nodes[i].op)
                .collect(),
        )
    }

    /// With a unit axis 1, the condition is true and the squeeze is real code.
    #[test]
    fn a_true_guard_resolves_to_the_then_branch() {
        let (import, report) = import_and_resolve(&guard_model(&[2, 1, 3]));
        assert_eq!(report.resolved, 1, "{report:?}");
        assert_eq!(report.unresolved_condition, 0);
        assert!(import.branches.is_empty(), "the If should be consumed");

        let (op, reads) = if_node(&import);
        assert!(
            matches!(op, Op::Identity),
            "the If should pass through: {op:?}"
        );
        assert_eq!(reads.len(), 1);
        assert!(
            matches!(reads[0], Op::Squeeze { .. }),
            "should read the then-branch squeeze, got {:?}",
            reads[0]
        );
    }

    /// With a non-unit axis 1, the condition is false. The squeeze must not be in the
    /// graph at all: it would be an error on this path, which is the whole reason both
    /// branches cannot be imported and selected afterwards.
    #[test]
    fn a_false_guard_resolves_to_the_else_branch_and_omits_the_squeeze() {
        let (import, report) = import_and_resolve(&guard_model(&[2, 4, 3]));
        assert_eq!(report.resolved, 1, "{report:?}");

        let (op, reads) = if_node(&import);
        assert!(matches!(op, Op::Identity));
        assert!(
            matches!(reads[0], Op::Identity),
            "should read the else-branch identity, got {:?}",
            reads[0]
        );
        assert!(
            !import
                .graph
                .nodes
                .iter()
                .any(|n| matches!(n.op, Op::Squeeze { .. })),
            "the untaken branch's squeeze must not be imported"
        );
    }

    /// The branch body's `Squeeze` reads `x` from the enclosing graph, which is the
    /// scoping rule that makes importing a subgraph work at all.
    #[test]
    fn a_branch_body_captures_from_the_enclosing_scope() {
        let (import, _) = import_and_resolve(&guard_model(&[2, 1, 3]));
        let squeeze = import
            .graph
            .nodes
            .iter()
            .find(|n| matches!(n.op, Op::Squeeze { .. }))
            .expect("the squeeze should be imported");
        let x = *import.value_to_node.get("x").expect("x is in scope");
        assert!(
            squeeze.inputs.contains(&x),
            "the squeeze should read the outer x, reads {:?}",
            squeeze.inputs
        );
        // Opset 13 moved Squeeze's axes to an operand, and the operand is a Constant
        // inside the branch body, so resolving it proves the body's own nodes land in
        // the same tensor map the importer reads axes from.
        assert!(
            matches!(&squeeze.op, Op::Squeeze { axes } if axes == &[1]),
            "axes should resolve from the branch's own Constant: {:?}",
            squeeze.op
        );
    }

    /// The graph stays acyclic and sortable even though the branch nodes are appended
    /// after the consumer that reads them.
    #[test]
    fn the_resolved_graph_still_sorts_topologically() {
        let (import, _) = import_and_resolve(&guard_model(&[2, 1, 3]));
        let order = import.graph.topological_sort();
        assert_eq!(order.len(), import.graph.nodes.len());
        let position: std::collections::BTreeMap<usize, usize> =
            order.iter().enumerate().map(|(p, &n)| (n, p)).collect();
        for (i, node) in import.graph.nodes.iter().enumerate() {
            for &input in &node.inputs {
                assert!(
                    position[&input] < position[&i],
                    "node {i} is sorted before its input {input}"
                );
            }
        }
    }

    /// A genuinely dynamic condition is left alone rather than guessed at. The ONNX
    /// backend suite's own `test_if` has `cond` as a graph input, so nothing can fold
    /// it.
    #[test]
    fn a_dynamic_condition_is_left_unresolved() {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("onnx/onnx/backend/test/data/node/test_if/model.onnx");
        if !path.exists() {
            eprintln!("skipping: {} is not checked out", path.display());
            return;
        }
        let bytes = std::fs::read(&path).expect("read fixture");
        let (import, report) = import_and_resolve(&bytes);
        assert_eq!(report.resolved, 0, "{report:?}");
        assert_eq!(report.unresolved_condition, 1, "{report:?}");
        assert_eq!(
            import.branches.len(),
            1,
            "the If should be kept for a later round"
        );
        assert!(
            import.graph.nodes.iter().any(|n| matches!(n.op, Op::If)),
            "the If should still be an If"
        );
    }

    /// Every constant in the resolved graph has a value, including the ones the branch
    /// body brought in.
    ///
    /// This is the check that pins the round ordering down. A second `materialize`
    /// round must be seeded with the *merged* map: the first round rewrote the nodes
    /// it folded to `Op::Constant`, but their values are in its own return value, not
    /// in `tensors`. Seeding the second round from `tensors` alone drops them, and
    /// lowering then meets a constant with no value.
    #[test]
    fn every_constant_in_the_resolved_graph_has_a_value() {
        let (import, all, _) = resolve_to_fixpoint(&guard_model(&[2, 1, 3]));
        for (i, node) in import.graph.nodes.iter().enumerate() {
            if matches!(node.op, Op::Constant { .. }) {
                assert!(
                    all.contains_key(&i),
                    "constant {i} ({:?}) has no value",
                    import.graph.names.get(&i)
                );
            }
        }
    }

    /// Seeding a later round from `tensors` alone loses the earlier round's folds,
    /// which is why [`resolve_to_fixpoint`] threads the merged map through. Stated as
    /// a test so the ordering cannot be quietly broken.
    #[test]
    fn a_second_round_seeded_from_tensors_alone_loses_folded_values() {
        let (mut import, _) = import_and_resolve(&guard_model(&[2, 1, 3]));
        let tensors = import.tensors.clone();
        let (folded, _) =
            materialize_constants(&mut import.graph, &tensors, &MaterializeConfig::default());
        let all = lowering_constants(&import.graph, &tensors, &folded);
        let missing = import
            .graph
            .nodes
            .iter()
            .enumerate()
            .filter(|(i, n)| matches!(n.op, Op::Constant { .. }) && !all.contains_key(i))
            .count();
        assert!(
            missing > 0,
            "expected the mis-seeded round to lose values; if this now passes, \
             materialize has become idempotent and resolve_to_fixpoint can be simplified"
        );
    }
}
