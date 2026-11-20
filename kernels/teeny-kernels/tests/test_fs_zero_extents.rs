//! Does any node in the FoundationStereo export have a zero-length extent?
//!
//! The CUDA run of the 96x128 model fails at `/cost_agg/conv3_up/conv/Slice_1` with
//! shape `[0, 112, 6, 8]` and a grid of `[0, 1, 1]`, which CUDA rejects. A grid of no
//! blocks launches nothing on RISC-V, leaving the node's buffer at the zero the pool
//! put there, which would make everything downstream zero and is the first candidate
//! for the all-zero disparity.
#![cfg(feature = "riscv")]

use teeny_core::model::{Lowering, LoweringMode};
use teeny_kernels::graph::TritonLowering;
use teeny_onnx::materialize::MaterializeConfig;
use teeny_onnx::{Onnx, branches::materialize_and_resolve};

#[test]
#[ignore = "needs FS96_ONNX"]
fn no_node_should_have_a_zero_extent() {
    let path = std::env::var("FS96_ONNX").expect("set FS96_ONNX");
    let mut import = Onnx::import_from_path(&path).expect("import");
    let (folded, _, _) =
        materialize_and_resolve(&mut import, &MaterializeConfig::default()).expect("resolve");

    // In the imported graph, before lowering.
    let mut graph_zeros: Vec<usize> = Vec::new();
    for (i, n) in import.graph.nodes.iter().enumerate() {
        if n.shape.iter().any(|d| *d == Some(0)) {
            graph_zeros.push(i);
        }
    }
    println!(
        "{} of {} graph nodes have a zero extent",
        graph_zeros.len(),
        import.graph.nodes.len()
    );
    for &i in graph_zeros.iter().take(40) {
        println!(
            "  graph {i:>6} {:?} {:?} {}",
            import.graph.nodes[i].shape,
            import.graph.nodes[i].op,
            import.graph.names.get(&i).cloned().unwrap_or_default()
        );
    }

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

    // Everything the output depends on, walking parents from `disp`.
    let disp_graph = *import.value_to_node.get("disp").expect("disp");
    let disp_dag = graph_to_dag[disp_graph];
    let mut needed = vec![false; dag.len()];
    let mut stack = vec![disp_dag];
    while let Some(d) = stack.pop() {
        if needed[d] {
            continue;
        }
        needed[d] = true;
        for &p in &dag.node(d).parents {
            stack.push(p);
        }
    }
    let reachable = needed.iter().filter(|x| **x).count();
    println!("\n`disp` depends on {reachable} of {} DAG nodes", dag.len());

    let mut on_path = 0usize;
    for d in 0..dag.len() {
        let g = dag_to_graph[d];
        if g == usize::MAX {
            continue;
        }
        if import.graph.nodes[g].shape.iter().any(|x| *x == Some(0)) && needed[d] {
            if on_path < 20 {
                println!(
                    "  on the path: dag {d:>6} graph {g:>6} {:?} {}",
                    import.graph.nodes[g].shape,
                    import.graph.names.get(&g).cloned().unwrap_or_default()
                );
            }
            on_path += 1;
        }
    }
    println!("\n{on_path} zero-extent nodes are on `disp`'s path");
    assert_eq!(on_path, 0, "the output depends on {on_path} empty nodes");
}
