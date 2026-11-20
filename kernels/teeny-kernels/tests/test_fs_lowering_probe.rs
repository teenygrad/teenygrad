// Reports how far FoundationStereo's ONNX graph gets through lowering.
//
// Imports the model, lowers it, and prints the first op with no lowering — so progress
// is measured against the real model rather than a count of match arms.
//
//   FS_ONNX=<path> cargo test -p teeny-kernels --test test_fs_lowering_probe \
//       -- --nocapture --ignored

use teeny_core::model::LoweringMode;
use teeny_kernels::graph::TritonLowering;

#[test]
#[ignore = "needs FS_ONNX pointing at a FoundationStereo .onnx export"]
fn probe_fs_lowering() {
    let path = std::env::var("FS_ONNX").expect("set FS_ONNX to a FoundationStereo .onnx");
    let (graph, tensors) = teeny_onnx::Onnx::from_path_with_tensors(&path).expect("import failed");
    let total: usize = tensors.values().map(|t| t.bytes.len()).sum();
    println!(
        "tensors recovered: {} ({:.1} MB)",
        tensors.len(),
        total as f64 / 1e6
    );
    if let Some((idx, t)) = tensors.iter().next() {
        println!(
            "  e.g. node {idx} ({}): dtype={:?} shape={:?} bytes={}",
            graph
                .names
                .get(idx)
                .map(|s| s.as_str())
                .unwrap_or("<unnamed>"),
            t.dtype,
            t.shape,
            t.bytes.len()
        );
    }
    println!("imported {} graph nodes", graph.nodes.len());
    let inputs = graph
        .nodes
        .iter()
        .enumerate()
        .filter(|(_, n)| matches!(n.op, teeny_core::graph::Op::Input))
        .count();
    let named_inputs = graph
        .nodes
        .iter()
        .enumerate()
        .filter(|(i, n)| {
            matches!(n.op, teeny_core::graph::Op::Input) && graph.names.contains_key(i)
        })
        .count();
    println!("Op::Input nodes: {inputs}, of which named: {named_inputs}");
    for (i, n) in graph.nodes.iter().enumerate().take(400) {
        if matches!(n.op, teeny_core::graph::Op::Input)
            && let Some(nm) = graph.names.get(&i)
        {
            println!("  e.g. node {i}: {nm}  shape={:?}", n.shape);
            break;
        }
    }

    let lowering = TritonLowering::new();
    match lowering.lower_with_mapping(&graph, LoweringMode::Inference) {
        Ok((dag, _)) => println!("LOWERED OK — {} dag nodes", dag.len()),
        Err(e) => println!("first blocker: {e}"),
    }
}
