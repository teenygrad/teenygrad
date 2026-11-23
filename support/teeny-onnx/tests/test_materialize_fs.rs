//! Runs the materialisation pass over a real exported model.
//!
//! Ignored by default: it needs the FoundationStereo export, which is ~1.75 GB and
//! lives outside the repo. Point `FS_ONNX` at an `.onnx` file and run with
//! `--ignored` to see what the pass folds and what it still cannot.

use std::collections::BTreeMap;
use std::time::Instant;

use teeny_core::graph::Op;
use teeny_onnx::Onnx;
use teeny_onnx::materialize::{MaterializeConfig, materialize_constants};

#[test]
#[ignore = "needs a large external .onnx export via FS_ONNX"]
fn test_materialize_shrinks_a_real_export() {
    let path = std::env::var("FS_ONNX").expect("set FS_ONNX to an .onnx path");

    let t0 = Instant::now();
    let (mut graph, tensors) = Onnx::from_path_with_tensors(&path).expect("import");
    println!(
        "imported {} nodes, {} tensors in {:.1}s",
        graph.nodes.len(),
        tensors.len(),
        t0.elapsed().as_secs_f64()
    );

    // Opset 13 moved Squeeze/Unsqueeze `axes` to an operand. If the importer did not
    // resolve it, every one of these nodes would carry an empty `axes`, which means a
    // different operation.
    let (mut axed, mut unaxed) = (0, 0);
    for n in &graph.nodes {
        if let Op::Squeeze { axes } | Op::Unsqueeze { axes } = &n.op {
            if axes.is_empty() {
                unaxed += 1;
            } else {
                axed += 1;
            }
        }
    }
    println!("Squeeze/Unsqueeze: {axed} with axes resolved, {unaxed} without");
    assert_eq!(unaxed, 0, "every axes operand in this export is a constant");

    // How many shapes the importer did not pin down, before the pass runs. Compare
    // with what ONNX's own inference put in the file: a gap means the importer is
    // dropping shapes the model already states.
    // Only an explicit `None` dimension counts as unresolved. An empty shape does
    // not: `Shape` cannot tell a genuine rank-0 scalar from an unestablished rank, and
    // this model has thousands of scalar constants.
    let dynamic_at_import = graph
        .nodes
        .iter()
        .filter(|n| n.shape.iter().any(Option::is_none))
        .count();
    let rank0_at_import = graph.nodes.iter().filter(|n| n.shape.is_empty()).count();
    println!(
        "at import: {dynamic_at_import} nodes with a dynamic dimension, \
         {rank0_at_import} with an empty shape (scalar or unknown rank), of {} nodes",
        graph.nodes.len()
    );

    let before: BTreeMap<String, usize> = tally(&graph);

    let t1 = Instant::now();
    let (folded, report) =
        materialize_constants(&mut graph, &tensors, &MaterializeConfig::default());
    println!(
        "folded {} nodes ({} values) in {:.2}s, {} skipped as too large",
        report.folded,
        folded.len(),
        t1.elapsed().as_secs_f64(),
        report.skipped_too_large
    );
    println!(
        "declined: {} Shape with unknown operand rank, {} out of range",
        report.shape_of_unknown_rank, report.declined_out_of_range
    );
    println!(
        "shapes: {} refined, {} still unresolved of {} nodes",
        report.shapes_refined,
        report.shapes_unresolved,
        graph.nodes.len()
    );

    // Which ops still have a dynamic dimension: the next shape rules worth writing.
    {
        let mut by_op: BTreeMap<String, usize> = BTreeMap::new();
        for n in &graph.nodes {
            if n.shape.iter().any(Option::is_none) {
                let label = format!("{:?}", n.op);
                let label = label.split([' ', '(']).next().unwrap_or("?").to_string();
                *by_op.entry(label).or_insert(0) += 1;
            }
        }
        let mut v: Vec<(&String, &usize)> = by_op.iter().collect();
        v.sort_by(|a, b| b.1.cmp(a.1));
        println!("still dynamic, by op:");
        for (op, n) in v.iter().take(15) {
            println!("  {n:>5}  {op}");
        }

        // The roots: dynamic although every operand is static, so inference stopped
        // here rather than inheriting the problem.
        let is_dyn = |n: &teeny_core::graph::GraphNode| n.shape.iter().any(Option::is_none);
        let mut roots: BTreeMap<String, usize> = BTreeMap::new();
        for n in &graph.nodes {
            if !is_dyn(n) {
                continue;
            }
            if n.inputs.iter().any(|&i| is_dyn(&graph.nodes[i])) {
                continue;
            }
            let label = format!("{:?}", n.op);
            let label = label.split([' ', '(']).next().unwrap_or("?").to_string();
            *roots.entry(label).or_insert(0) += 1;
        }
        let mut r: Vec<(&String, &usize)> = roots.iter().collect();
        r.sort_by(|a, b| b.1.cmp(a.1));
        println!("roots (dynamic with all-static inputs):");
        for (op, n) in r {
            println!("  {n:>5}  {op}");
        }
    }

    let after = tally(&graph);
    println!("\ntop unsupported ops (candidates to implement next):");
    for (op, n) in report.unsupported_ranked().iter().take(25) {
        println!("  {n:>6}  {op}");
    }

    println!("\nop counts, before -> after (changed only):");
    let mut keys: Vec<&String> = before.keys().chain(after.keys()).collect();
    keys.sort();
    keys.dedup();
    for k in keys {
        let (b, a) = (
            before.get(k).copied().unwrap_or(0),
            after.get(k).copied().unwrap_or(0),
        );
        if b != a {
            println!("  {k:<22} {b:>6} -> {a:>6}");
        }
    }

    assert!(report.folded > 0, "nothing folded in a real export");
}

fn tally(graph: &teeny_core::graph::Graph) -> BTreeMap<String, usize> {
    let mut m = BTreeMap::new();
    for n in &graph.nodes {
        let label = format!("{:?}", n.op);
        let label = label.split([' ', '(']).next().unwrap_or("?").to_string();
        *m.entry(label).or_insert(0) += 1;
    }
    let _ = Op::Identity;
    m
}
