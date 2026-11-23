//! `CumSum` along an axis that is not the innermost one.
//!
//! ONNX carries the axis as an operand, always, so the importer cannot see it: it is a
//! `Constant` node, and `materialize::adopt_axis_operands` writes it onto the op once
//! the fold has computed it. Before that the op kept the 0 the importer wrote and the
//! kernel accumulated along the innermost axis whatever the file asked for, which is a
//! wrong answer of the right shape.
//!
//! FoundationStereo has two of these, both on `[2, 12, 16]`: one over axis 1, which was
//! wrong, and one over axis 2, which was right by coincidence.
//!
//! `CUMSUM_ONNX` and `CUMSUM_DATA`, built by `scripts/make_cumsum_onnx.py`.
#![cfg(feature = "hardware")]

use teeny_core::model::LoweringMode;
use teeny_kernels::graph::TritonLowering;
use teeny_onnx::materialize::MaterializeConfig;
use teeny_onnx::{Onnx, branches::materialize_and_resolve};
use teeny_test::model::RuntimeModel;

fn read_f32(path: &str) -> Vec<f32> {
    let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("reading {path}: {e}"));
    bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

#[test]
#[ignore = "needs CUMSUM_ONNX and CUMSUM_DATA"]
fn test_a_cumulative_sum_follows_its_axis() {
    let path = std::env::var("CUMSUM_ONNX").expect("set CUMSUM_ONNX");
    let data = std::env::var("CUMSUM_DATA").expect("set CUMSUM_DATA");
    let stem = std::path::Path::new(&path)
        .file_stem()
        .and_then(|s| s.to_str())
        .expect("a file stem");
    let x = read_f32(&format!("{data}/{stem}_x.f32"));
    let want = read_f32(&format!("{data}/{stem}_ref.f32"));

    let mut import = Onnx::import_from_path(&path).expect("import");
    let (folded, report, _) =
        materialize_and_resolve(&mut import, &MaterializeConfig::default()).expect("resolve");
    println!(
        "axis operands adopted: {}, reduce axes adopted: {}",
        report.axis_operands_adopted, report.reduce_axes_adopted
    );
    for (i, n) in import.graph.nodes.iter().enumerate() {
        if matches!(n.op, teeny_core::graph::Op::CumSum { .. }) {
            println!("  node {i}: {:?}", n.op);
        }
    }
    let lowering = TritonLowering::new().with_constants(folded);

    let (device, target) = {
        #[cfg(feature = "cuda")]
        {
            let device = teeny_runtime::open().expect("device");
            let target = teeny_runtime::default_target(&device).expect("target");
            (device, target)
        }
        #[cfg(not(feature = "cuda"))]
        {
            use teeny_riscv::compiler::target::Capability;
            (
                teeny_runtime::Device::new(Default::default()),
                teeny_runtime::Target::new(Capability::GenericRvv1_0),
            )
        }
    };

    let mut model = RuntimeModel::compile(
        &device,
        &import.graph,
        &lowering,
        LoweringMode::Inference,
        &target,
        1,
        false,
    )
    .expect("compile");

    let outputs = model.terminal_node_indices_sorted_by_size();
    let got = model.run(&device, &[&x], &outputs).expect("run");
    let out = got
        .iter()
        .find(|o| o.len() == want.len())
        .expect("an output of the reference's length");

    println!(
        "{stem}: {} values, min {:.6e}, max {:.6e}",
        out.len(),
        out.iter().fold(f32::INFINITY, |m, v| m.min(*v)),
        out.iter().fold(f32::NEG_INFINITY, |m, v| m.max(*v))
    );
    for (i, (g, w)) in out.iter().zip(&want).enumerate().take(12) {
        println!("  {i:>3}  got {g:>14.6e}  want {w:>14.6e}");
    }

    for (i, (g, w)) in out.iter().zip(&want).enumerate() {
        assert!(
            (g - w).abs() < 1e-5,
            "element {i}: got {g:.6e}, want {w:.6e}"
        );
    }
}
