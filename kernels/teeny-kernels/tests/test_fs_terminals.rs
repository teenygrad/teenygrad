//! Which terminal node is the FoundationStereo export's disparity?
//!
//! `test_fs_riscv_run` asks the executor for every terminal and then takes whichever
//! output happens to have the reference's length. If more than one terminal has that
//! length, that picks by accident, which would explain a result that is exactly zero
//! rather than merely wrong. The export declares its output as `disp`, with the shape
//! `[0, 1, 0, 0]`: the exporter left three dims unset, so there is no `value_info` for
//! it and the importer gives the node its input's shape instead (teenygrad#129).
#![cfg(feature = "riscv")]

use teeny_core::model::{Lowering, LoweringMode};
use teeny_kernels::graph::TritonLowering;
use teeny_onnx::materialize::MaterializeConfig;
use teeny_onnx::{Onnx, branches::materialize_and_resolve};

fn len_of(shape: &[Option<usize>]) -> usize {
    shape.iter().map(|d| d.unwrap_or(1)).product()
}

#[test]
#[ignore = "needs FS96_ONNX"]
fn which_terminals_could_be_the_disparity() {
    let path = std::env::var("FS96_ONNX").expect("set FS96_ONNX");
    let mut import = Onnx::import_from_path(&path).expect("import");

    // The importer keeps no record of the file's declared outputs, so find `disp` by
    // the name of the value it produces.
    if let Some(&g) = import.value_to_node.get("disp") {
        println!(
            "'disp' is graph node {g}, shape {:?} (len {}), dtype {:?}",
            import.graph.nodes[g].shape,
            len_of(&import.graph.nodes[g].shape),
            import.graph.nodes[g].dtype
        );
    } else {
        println!("'disp' is not in value_to_node at all");
    }

    let (folded, _, _) =
        materialize_and_resolve(&mut import, &MaterializeConfig::default()).expect("resolve");
    let lowering = TritonLowering::new().with_constants(folded);
    let (dag, graph_to_dag) = lowering
        .lower_with_mapping(&import.graph, LoweringMode::Inference)
        .expect("lower");

    let mut dag_to_graph = vec![usize::MAX; dag.len()];
    for (g, &d) in graph_to_dag.iter().enumerate() {
        if d < dag.len() {
            dag_to_graph[d] = g;
        }
    }

    let terminals: Vec<usize> = (0..dag.len())
        .filter(|&d| dag.node(d).children.is_empty())
        .collect();

    println!("\n{} terminal DAG nodes:", terminals.len());
    for &d in &terminals {
        let g = dag_to_graph[d];
        if g == usize::MAX {
            println!("  dag {d:>6} (no graph node)");
            continue;
        }
        let n = &import.graph.nodes[g];
        println!(
            "  dag {d:>6} graph {g:>6} len {:>8} {:?} {:?} {}",
            len_of(&n.shape),
            n.shape,
            n.dtype,
            import.graph.names.get(&g).cloned().unwrap_or_default()
        );
    }

    let want = 12288; // the reference disparity's length, 1 * 1 * 96 * 128
    let candidates: Vec<usize> = terminals
        .iter()
        .copied()
        .filter(|&d| dag_to_graph[d] != usize::MAX && len_of(&import.graph.nodes[dag_to_graph[d]].shape) == want)
        .collect();
    println!(
        "\nterminals of length {want}: {} -- the run picks the first of these",
        candidates.len()
    );
    for &d in &candidates {
        let g = dag_to_graph[d];
        println!(
            "  dag {d} graph {g} {}",
            import.graph.names.get(&g).cloned().unwrap_or_default()
        );
    }
}
