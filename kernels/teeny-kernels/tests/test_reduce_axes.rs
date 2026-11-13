//! A reduction over an axis the kernels cannot reach is refused, not computed.
//!
//! The reduction kernels walk a flattened `[outer, inner]` view and infer the split from
//! the output shape, so they can only reduce the trailing axes. Reducing any other axis
//! sums adjacent elements instead of strided ones, which is a wrong answer of the right
//! shape and dtype: nothing downstream can tell. Until the lowering can transpose the
//! axes innermost, which needs `Op::Transpose`, it says so instead.

use teeny_core::graph::{DtypeRepr, Graph, Op};
use teeny_core::model::{Lowering, LoweringMode};
use teeny_kernels::graph::TritonLowering;

/// `[2, 3, 4, 5]` reduced along `axes`, with the output shape ONNX gives.
fn graph_reducing(axes: &[i64], keepdims: bool) -> Graph {
    let shape = [2usize, 3, 4, 5];
    let mut g = Graph::new();
    let x = g.add_node(
        Op::Input,
        vec![],
        DtypeRepr::F32,
        shape.iter().map(|d| Some(*d)).collect(),
    );
    let reduced: Vec<usize> = axes
        .iter()
        .map(|a| if *a < 0 { a + shape.len() as i64 } else { *a } as usize)
        .collect();
    let out_shape: Vec<Option<usize>> = shape
        .iter()
        .enumerate()
        .filter_map(|(i, d)| match (reduced.contains(&i), keepdims) {
            (true, true) => Some(Some(1)),
            (true, false) => None,
            (false, _) => Some(Some(*d)),
        })
        .collect();
    g.add_node(
        Op::ReduceSum {
            axes: axes.to_vec(),
            keepdims,
            noop_with_empty_axes: false,
        },
        vec![x],
        DtypeRepr::F32,
        out_shape,
    );
    g
}

fn lower(g: &Graph) -> anyhow::Result<()> {
    TritonLowering::new()
        .lower(g, LoweringMode::Inference)
        .map(|_| ())
}

#[test]
fn the_innermost_axis_lowers() {
    lower(&graph_reducing(&[3], true)).expect("the last axis is what the kernel reduces");
    lower(&graph_reducing(&[-1], false)).expect("a negative axis resolves to the last one");
    lower(&graph_reducing(&[2, 3], true)).expect("a trailing block is fine");
    lower(&graph_reducing(&[], true)).expect("every axis is a trailing block");
}

#[test]
fn any_other_axis_is_refused() {
    for axes in [vec![0i64], vec![1], vec![2], vec![0, 1], vec![1, 3]] {
        let err = lower(&graph_reducing(&axes, true))
            .expect_err(&format!("{axes:?} is not the innermost axis"));
        let msg = err.to_string();
        assert!(
            msg.contains("innermost"),
            "{axes:?} should say why it was refused, said: {msg}"
        );
    }
}

#[test]
fn an_axis_outside_the_rank_is_refused() {
    let err = lower(&graph_reducing(&[4], true)).expect_err("rank 4 has no axis 4");
    assert!(err.to_string().contains("within"), "{err}");
}

/// An export may put the axes in an operand instead of the attribute, from opset 13 for
/// `ReduceSum`. One the importer could not resolve leaves the list empty, which would
/// otherwise be read as "reduce everything".
#[test]
fn unresolved_axes_from_an_operand_are_refused() {
    let mut g = Graph::new();
    let x = g.add_node(
        Op::Input,
        vec![],
        DtypeRepr::F32,
        vec![Some(2), Some(3), Some(4)],
    );
    let axes = g.add_node(Op::Input, vec![], DtypeRepr::I64, vec![Some(1)]);
    g.add_node(
        Op::ReduceSum {
            axes: vec![],
            keepdims: true,
            noop_with_empty_axes: false,
        },
        vec![x, axes],
        DtypeRepr::F32,
        vec![Some(2), Some(3), Some(1)],
    );
    let err = lower(&g).expect_err("an unresolved operand cannot mean every axis");
    assert!(err.to_string().contains("computed operand"), "{err}");
}
