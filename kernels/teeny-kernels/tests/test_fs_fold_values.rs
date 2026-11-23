//! What the constant fold produced for named shape operands.
//!
//! FoundationStereo's crop-then-pad chains compute their pads and slice bounds from
//! `Shape` nodes, so every one of them is a folded constant rather than a literal in
//! the file. A fold that is wrong there produces a correctly-shaped tensor of padding,
//! which is exactly zero and passes every shape check.
//!
//! `FS_FOLD_NAMES` is a comma-separated list of node names to print.
#![cfg(feature = "riscv")]

use teeny_onnx::materialize::MaterializeConfig;
use teeny_onnx::{Onnx, branches::materialize_and_resolve};

#[test]
#[ignore = "needs FS96_ONNX and FS_FOLD_NAMES"]
fn test_what_the_fold_made_of_these_operands() {
    let path = std::env::var("FS96_ONNX").expect("set FS96_ONNX");
    let names: Vec<String> = std::env::var("FS_FOLD_NAMES")
        .expect("set FS_FOLD_NAMES")
        .split(',')
        .map(|s| s.trim().to_string())
        .collect();

    let mut import = Onnx::import_from_path(&path).expect("import");
    let (folded, _, _) =
        materialize_and_resolve(&mut import, &MaterializeConfig::default()).expect("resolve");

    for name in &names {
        // A node's name and the name of the value it produces differ in an export, so
        // try both.
        let by_value = import.value_to_node.get(name).copied();
        let by_node = import
            .graph
            .names
            .iter()
            .find(|(_, n)| *n == name)
            .map(|(i, _)| *i);
        let Some(idx) = by_value.or(by_node) else {
            println!("{name}: no node");
            continue;
        };
        let node = &import.graph.nodes[idx];
        print!(
            "{name}: graph {idx} {:?} shape {:?} dtype {:?}",
            node.op, node.shape, node.dtype
        );
        match folded.get(&idx) {
            Some(c) => {
                let ints = c.to_i64_vec();
                let floats = c.to_f64_vec();
                println!(
                    "\n  folded: dtype {:?} shape {:?} i64 {:?} f64 {:?}",
                    c.dtype, c.shape, ints, floats
                );
            }
            None => println!("\n  not folded"),
        }
        for &input in &node.inputs {
            let iname = import.graph.names.get(&input).cloned().unwrap_or_default();
            println!(
                "    from graph {input} {:?} {iname} folded {:?}",
                import.graph.nodes[input].op,
                folded.get(&input).map(|c| c.to_i64_vec())
            );
        }
    }
}
