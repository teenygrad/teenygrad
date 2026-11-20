// Cast: the only one that lowers is the one that changes nothing.
//
// Exporters emit casts in quantity between a dtype and itself. All 300 in the
// FoundationStereo export are F32 to F32, so refusing them for want of a conversion
// kernel held up the whole graph for conversions none of them asks for.
//
//   cargo test -p teeny-kernels --test test_cast

use teeny_core::graph::{DtypeRepr, Graph, Op};
use teeny_core::model::{Lowering, LoweringMode};
use teeny_kernels::graph::TritonLowering;

fn cast_graph(from: DtypeRepr, to: DtypeRepr, node_dtype: DtypeRepr) -> Graph {
    let mut g = Graph::new();
    let x = g.add_node(Op::Input, vec![], from, vec![Some(6)]);
    g.add_node(Op::Cast { to }, vec![x], node_dtype, vec![Some(6)]);
    g
}

fn lower(from: DtypeRepr, to: DtypeRepr, node_dtype: DtypeRepr) -> anyhow::Result<usize> {
    TritonLowering::new()
        .lower(&cast_graph(from, to, node_dtype), LoweringMode::Inference)
        .map(|dag| dag.len())
}

#[test]
fn a_cast_to_the_same_dtype_is_a_copy() {
    assert_eq!(
        lower(DtypeRepr::F32, DtypeRepr::F32, DtypeRepr::F32).expect("f32 to f32"),
        2
    );
    assert_eq!(
        lower(DtypeRepr::I64, DtypeRepr::I64, DtypeRepr::I64).expect("i64 to i64"),
        2
    );
}

/// A real conversion is refused with both dtypes named, rather than copying the bytes
/// across and calling them the new dtype.
#[test]
fn a_converting_cast_says_what_it_would_have_to_convert() {
    let msg = match lower(DtypeRepr::I64, DtypeRepr::F32, DtypeRepr::F32) {
        Ok(_) => panic!("an I64 to F32 cast should not lower as a copy"),
        Err(e) => e.to_string(),
    };
    assert!(msg.contains("from I64 to F32"), "{msg}");
}

/// A node whose own dtype disagrees with what it claims to cast to is refused too: the
/// output buffer's width comes from the node, so a copy would write the wrong amount.
#[test]
fn a_node_dtype_that_disagrees_with_the_target_is_refused() {
    let msg = match lower(DtypeRepr::F32, DtypeRepr::F32, DtypeRepr::I64) {
        Ok(_) => panic!("a disagreeing node dtype should not lower"),
        Err(e) => e.to_string(),
    };
    assert!(msg.contains("node dtype I64"), "{msg}");
}
