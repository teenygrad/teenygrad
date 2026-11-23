//! `Softmax` through the ONNX import, at ranks and axes other than 2-D-over-the-last.
//!
//! Two faults, which the three models separate. `SoftmaxRuntimeOp::grid` is
//! `[output_shape[0], 1, 1]`, so it launches one CTA per entry of the *first* axis
//! rather than one per row, and the lowering takes `n_cols` from the last axis and
//! drops `Op::Softmax`'s `dim`. A 2-D softmax over the last axis is the case where both
//! happen to be right.
//!
//! `SOFTMAX_ONNX` and `SOFTMAX_DATA` name the model and the fixtures, built by
//! `kernels/teeny-kernels/tests/fixtures/onnx/make_softmax_onnx.py`.
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
#[ignore = "needs SOFTMAX_ONNX and SOFTMAX_DATA"]
fn a_softmax_sums_to_one_along_its_axis() {
    let path = std::env::var("SOFTMAX_ONNX").expect("set SOFTMAX_ONNX");
    let data = std::env::var("SOFTMAX_DATA").expect("set SOFTMAX_DATA");
    let stem = std::path::Path::new(&path)
        .file_stem()
        .and_then(|s| s.to_str())
        .expect("a file stem");
    let x = read_f32(&format!("{data}/{stem}_x.f32"));
    let want = read_f32(&format!("{data}/{stem}_ref.f32"));

    let mut import = Onnx::import_from_path(&path).expect("import");
    let (folded, _, _) =
        materialize_and_resolve(&mut import, &MaterializeConfig::default()).expect("resolve");
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

    let zeros = out.iter().filter(|v| **v == 0.0).count();
    println!(
        "{stem}: {} values, {zeros} exactly zero, min {:.6e}, max {:.6e}",
        out.len(),
        out.iter().fold(f32::INFINITY, |m, v| m.min(*v)),
        out.iter().fold(0.0f32, |m, v| m.max(*v))
    );
    for (i, (g, w)) in out.iter().zip(&want).enumerate().take(12) {
        println!("  {i:>3}  got {g:>14.6e}  want {w:>14.6e}");
    }

    // A softmax is strictly positive: every element is exp of something finite over a
    // positive sum, so an exact zero means the kernel never wrote there.
    assert_eq!(
        zeros,
        0,
        "{zeros} of {} elements are exactly zero, so they were never written",
        out.len()
    );
    for (i, (g, w)) in out.iter().zip(&want).enumerate() {
        assert!(
            (g - w).abs() < 1e-5,
            "element {i}: got {g:.6e}, want {w:.6e}"
        );
    }
}
