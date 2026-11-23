//! A reduction whose axes arrive as an operand rather than an attribute.
//!
//! ONNX kept `axes` as an attribute until opset 13 for `ReduceSum` and opset 18 for the
//! rest, and as an operand after. The importer reads only the attribute form, so the
//! op's `axes` stayed empty for an operand, and an empty list means *every* axis unless
//! `noop_with_empty_axes` says otherwise: a reduction over one axis became a reduction
//! over all of them, which is a wrong answer with a wrong shape that shape inference
//! then agreed with. `teeny_onnx::materialize::adopt_reduce_axes` moves the operand's
//! value onto the op.
//!
//! All 10 `ReduceSum` nodes in the FoundationStereo export are this shape, with an
//! operand of `[2]` and no attribute, while its 24 `ReduceL2` nodes use the attribute.
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
#[ignore = "needs RSUM_ONNX and RSUM_DATA"]
fn test_the_axes_operand_decides_what_is_reduced() {
    let path = std::env::var("RSUM_ONNX").expect("set RSUM_ONNX");
    let data = std::env::var("RSUM_DATA").expect("set RSUM_DATA");
    let x = read_f32(&format!("{data}/rsum_operand_x.f32"));
    let want = read_f32(&format!("{data}/rsum_operand_ref.f32"));

    let mut import = Onnx::import_from_path(&path).expect("import");
    let (folded, report, _) =
        materialize_and_resolve(&mut import, &MaterializeConfig::default()).expect("resolve");
    println!(
        "{} reduction(s) took their axes from an operand",
        report.reduce_axes_adopted
    );
    assert_eq!(
        report.reduce_axes_adopted, 1,
        "the axes operand should have been adopted"
    );

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
        .unwrap_or_else(|| {
            panic!(
                "no output of {} values; got {:?}. Reducing every axis would give one.",
                want.len(),
                got.iter().map(|o| o.len()).collect::<Vec<_>>()
            )
        });

    for (i, (g, w)) in out.iter().zip(&want).enumerate() {
        assert!(
            (g - w).abs() < 1e-4,
            "element {i}: got {g:.6e}, want {w:.6e}"
        );
    }
}
