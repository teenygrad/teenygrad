//! `x / clip(||x||, eps)` through the ONNX import, as FoundationStereo writes it.
//!
//! A normalized vector's elements are at most one, since a norm is at least the
//! magnitude of any element it covers. In the FoundationStereo export this comes out at
//! 1.5e13: the reduction is right, matching a recomputation from its own input to
//! 7.6e-6, but the `Expand` that broadcasts the norm back over the reduced axis hands
//! elements a different group's norm, and the ones that land on an all-zero group get
//! the clip's floor of 1e-12 instead.
//!
//! `scripts/make_l2norm_onnx.py` builds two models that differ only in where the
//! reduced axis sits. `l2norm.onnx`, `[1, 6, 4, 1]` reducing axis 2, has it innermost in
//! effect and is exact to 1e-7. `l2norm5d.onnx`, `[1, 2, 5, 3, 4]` reducing the axis of
//! extent 5, gives it a stride and comes out at 2.70, identically on CUDA and on RISC-V.
//!
//! The cause is that every `Op::Reduce*` arm in the lowering matches `{ .. }` and drops
//! its axes, while `reduction.rs` reduces a flattened `[outer, inner]` view and says the
//! caller must reshape first. Nothing does, so a reduce over any axis but the innermost
//! sums adjacent elements instead of strided ones, with the right output shape either
//! way. One group is all zero in both models, as the cost volume's padding is, so the
//! clip floor is exercised too.
//!
//! The paths come from `L2NORM_ONNX` and `L2NORM_DATA`.
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
#[ignore = "needs L2NORM_ONNX and L2NORM_DATA"]
fn a_normalized_tensor_has_no_element_above_one() {
    let path = std::env::var("L2NORM_ONNX").expect("set L2NORM_ONNX");
    let data = std::env::var("L2NORM_DATA").expect("set L2NORM_DATA");
    let x = read_f32(&format!("{data}/l2norm5d_x.f32"));
    let want = read_f32(&format!("{data}/l2norm5d_ref.f32"));

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

    let max = out.iter().fold(0.0f32, |m, v| m.max(v.abs()));
    let worst = out
        .iter()
        .zip(&want)
        .enumerate()
        .max_by(|(_, (a, b)), (_, (c, d))| (*a - *b).abs().total_cmp(&(*c - *d).abs()))
        .map(|(i, (a, b))| (i, *a, *b))
        .expect("non-empty");
    println!("max |out| = {max:.4e}");
    println!(
        "worst element {}: got {:.6e}, want {:.6e}",
        worst.0, worst.1, worst.2
    );
    for (i, (g, w)) in out.iter().zip(&want).enumerate().take(24) {
        println!("  {i:>3}  got {g:>14.6e}  want {w:>14.6e}");
    }

    assert!(
        max <= 1.001,
        "a normalized element cannot exceed one, and the largest is {max:.4e}"
    );
    for (i, (g, w)) in out.iter().zip(&want).enumerate() {
        assert!(
            (g - w).abs() < 1e-4,
            "element {i}: got {g:.6e}, want {w:.6e}"
        );
    }
}
