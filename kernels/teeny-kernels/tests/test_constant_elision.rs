//! Lowering of graphs whose shape operands are materialised constants.
//!
//! An exporter emits a reshape target, slice bounds or a pad list as an ordinary
//! graph edge, so the operand has to be *something* by the time lowering runs. It is
//! not tensor data and never needs a buffer, so lowering reads its value and leaves
//! it out of the DAG. These tests pin that behaviour, and the distinction from a
//! constant that really does flow data.

use std::collections::BTreeMap;

use teeny_core::graph::{ConstantMap, ConstantTensor, DtypeRepr, Graph, Op};
use teeny_core::model::{Lowering, LoweringMode};
use teeny_kernels::graph::TritonLowering;

fn i64_tensor(shape: &[usize], vals: &[i64]) -> ConstantTensor {
    let mut bytes = Vec::new();
    for v in vals {
        bytes.extend_from_slice(&v.to_le_bytes());
    }
    ConstantTensor {
        dtype: DtypeRepr::I64,
        shape: shape.iter().map(|&d| Some(d)).collect(),
        bytes,
    }
}

/// Lowers `graph`, expecting it to fail, and returns the error text.
///
/// `Dag` does not implement `Debug`, so `expect_err` is unavailable.
fn lowering_error(graph: &Graph, why: &str) -> String {
    match TritonLowering::new().lower(graph, LoweringMode::Inference) {
        Ok(_) => panic!("expected lowering to fail: {why}"),
        Err(e) => e.to_string(),
    }
}

/// `x[2,3] -> Reshape(x, [3,2])`, with the target as a constant operand.
fn reshape_graph() -> (Graph, usize, usize, usize) {
    let mut g = Graph::new();
    let x = g.add_node(Op::Input, vec![], DtypeRepr::F32, vec![Some(2), Some(3)]);
    let target = g.add_node(
        Op::Constant {
            dtype: DtypeRepr::I64,
            shape: vec![Some(2)],
        },
        vec![],
        DtypeRepr::I64,
        vec![Some(2)],
    );
    let rs = g.add_node(
        Op::Reshape,
        vec![x, target],
        DtypeRepr::F32,
        vec![Some(3), Some(2)],
    );
    (g, x, target, rs)
}

#[test]
fn a_shape_operand_constant_is_left_out_of_the_dag() {
    let (graph, _x, _target, _rs) = reshape_graph();

    let mut constants = ConstantMap::new();
    constants.insert(_target, i64_tensor(&[2], &[3, 2]));

    let lowering = TritonLowering::new().with_constants(constants);
    let dag = lowering
        .lower(&graph, LoweringMode::Inference)
        .expect("a Reshape whose target is a materialised constant should lower");

    // Three graph nodes, two DAG nodes: the shape operand is metadata, not a buffer.
    assert_eq!(graph.nodes.len(), 3);
    assert_eq!(
        dag.len(),
        2,
        "the shape operand should not become a DAG node"
    );
}

/// The elision is about how the operand is *consumed*, not about having its value:
/// a shape operand is metadata whether or not a value was supplied for it.
#[test]
fn a_shape_operand_is_elided_even_without_a_supplied_value() {
    let (graph, ..) = reshape_graph();

    let lowering = TritonLowering::new();
    let dag = lowering
        .lower(&graph, LoweringMode::Inference)
        .expect("the shape operand is metadata regardless of whether a value was given");
    assert_eq!(dag.len(), 2);
}

/// A constant added to an activation flows data, so it still needs a buffer. That is
/// not implemented, and the error should say which node and why, rather than pointing
/// at a materialisation pass that already ran.
#[test]
fn a_data_constant_reports_itself_by_name() {
    let mut g = Graph::new();
    let x = g.add_node(Op::Input, vec![], DtypeRepr::F32, vec![Some(4)]);
    let bias = g.add_node(
        Op::Constant {
            dtype: DtypeRepr::F32,
            shape: vec![Some(4)],
        },
        vec![],
        DtypeRepr::F32,
        vec![Some(4)],
    );
    g.names.insert(bias, "my_bias".to_string());
    let _add = g.add_node(Op::Add, vec![x, bias], DtypeRepr::F32, vec![Some(4)]);

    let msg = lowering_error(&g, "a data constant has no buffer yet");
    assert!(msg.contains("my_bias"), "should name the node: {msg}");
    assert!(
        msg.contains("flows tensor data"),
        "should say why it is not elidable: {msg}"
    );
}

/// A constant used in both a shape position and a data position cannot be elided:
/// something reads it as a buffer.
#[test]
fn a_constant_used_as_both_shape_and_data_is_kept() {
    let mut g = Graph::new();
    let x = g.add_node(Op::Input, vec![], DtypeRepr::I64, vec![Some(2)]);
    let shared = g.add_node(
        Op::Constant {
            dtype: DtypeRepr::I64,
            shape: vec![Some(2)],
        },
        vec![],
        DtypeRepr::I64,
        vec![Some(2)],
    );
    // Shape position.
    let _rs = g.add_node(
        Op::Reshape,
        vec![x, shared],
        DtypeRepr::I64,
        vec![Some(2), Some(1)],
    );
    // Data position.
    let _add = g.add_node(Op::Add, vec![x, shared], DtypeRepr::I64, vec![Some(2)]);

    let msg = lowering_error(&g, "the data use means it still needs a buffer");
    assert!(msg.contains("flows tensor data"), "{msg}");
}

/// A `Squeeze`/`Unsqueeze` axes operand is metadata too: the importer lifts it into
/// the op, leaving the edge redundant.
#[test]
fn an_axes_operand_is_elided() {
    let mut g = Graph::new();
    let x = g.add_node(
        Op::Input,
        vec![],
        DtypeRepr::F32,
        vec![Some(2), Some(1), Some(3)],
    );
    let axes = g.add_node(
        Op::Constant {
            dtype: DtypeRepr::I64,
            shape: vec![Some(1)],
        },
        vec![],
        DtypeRepr::I64,
        vec![Some(1)],
    );
    let _sq = g.add_node(
        Op::Squeeze { axes: vec![1] },
        vec![x, axes],
        DtypeRepr::F32,
        vec![Some(2), Some(3)],
    );

    let mut constants = BTreeMap::new();
    constants.insert(axes, i64_tensor(&[1], &[1]));

    let dag = TritonLowering::new()
        .with_constants(constants)
        .lower(&g, LoweringMode::Inference)
        .expect("an axes operand should not need a buffer");
    assert_eq!(dag.len(), 2);
}

/// The graph's result is kept even when nothing consumes it, so it can be read back.
#[test]
fn a_constant_that_is_the_graphs_result_is_kept() {
    let mut g = Graph::new();
    let only = g.add_node(
        Op::Constant {
            dtype: DtypeRepr::F32,
            shape: vec![Some(1)],
        },
        vec![],
        DtypeRepr::F32,
        vec![Some(1)],
    );
    g.names.insert(only, "answer".to_string());

    let msg = lowering_error(&g, "the result must be materialised, so it is not elidable");
    assert!(msg.contains("answer"), "{msg}");
}
