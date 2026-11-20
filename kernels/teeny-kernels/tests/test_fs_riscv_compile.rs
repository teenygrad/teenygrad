// Compiling the whole FoundationStereo export for RISC-V.
//
// The lowering tests next door say every node in the export turns into a kernel. This
// says every one of those kernels compiles for riscv64 through teenyc, at the shapes
// the export actually uses, which is the step between "lowers" and "runs".
//
// Many nodes share a kernel: what differs between two convolutions of different shapes
// is the arguments, not the source. So this compiles each distinct source once and
// reports both counts.
//
//   FS_ONNX=/path/to/fs.onnx TEENYC_QEMU_RISCV64=/usr/bin/qemu-riscv64-static \
//     cargo test -p teeny-kernels --no-default-features --features qemu,hardware \
//     --test test_fs_riscv_compile -- --ignored --nocapture

#![cfg(all(feature = "hardware", feature = "riscv", not(feature = "cuda")))]

use std::collections::BTreeMap;
use std::time::Instant;

use teeny_core::model::{Lowering, LoweringMode};
use teeny_kernels::graph::TritonLowering;
use teeny_onnx::materialize::MaterializeConfig;
use teeny_onnx::{Onnx, branches::materialize_and_resolve};
use teeny_riscv::compiler::target::Capability;
use teeny_runtime::Target;
use teeny_test::ExecKernel;

/// What a forward pass of the export would cost in activation buffers.
///
/// Compiles the model for riscv64 and asks the executor what its own pool would do,
/// without running it: the peak depends on buffers being matched by exact size, not
/// just on the graph's liveness, so this is the number that says whether a pass fits.
#[test]
#[ignore = "needs FS_ONNX and a few minutes to compile"]
fn what_a_pass_would_hold() {
    use teeny_test::model::RuntimeModel;

    let path = std::env::var("FS_ONNX").expect("set FS_ONNX to an .onnx path");
    let mut import = Onnx::import_from_path(&path).expect("import");
    let (folded, _, _) =
        materialize_and_resolve(&mut import, &MaterializeConfig::default()).expect("resolve");

    let device = teeny_runtime::Device::new(Default::default());
    let target = Target::new(Capability::SpacemitK3);
    let started = Instant::now();
    let model = RuntimeModel::compile(
        &device,
        &import.graph,
        &TritonLowering::new().with_constants(folded),
        LoweringMode::Inference,
        &target,
        1,
        false,
    )
    .expect("compiling the model");
    println!("compiled in {:.1?}", started.elapsed());

    let outputs = model.terminal_node_indices_sorted_by_size();
    let stats = model.planned_buffer_stats(&outputs);
    let mib = |words: usize| (words * 4) as f64 / (1 << 20) as f64;
    println!(
        "activations: {} buffers, {} reused, {:.2} GiB allocated, {:.2} GiB peak",
        stats.acquired,
        stats.reused,
        mib(stats.allocated_words) / 1024.0,
        mib(stats.peak_words) / 1024.0,
    );
}

/// Which of the export's kernels have no runtime dispatch.
///
/// A kernel whose `RuntimeOp` is one of the stubs in the lowering compiles and then
/// panics when the executor asks it to pack arguments, so "lowers" is not the same as
/// "runs". This asks every node for its argument count, which is the first thing the
/// executor does, and reports the ones that refuse.
#[test]
#[ignore = "needs FS_ONNX"]
fn which_kernels_have_no_runtime_dispatch() {
    let path = std::env::var("FS_ONNX").expect("set FS_ONNX to an .onnx path");
    let mut import = Onnx::import_from_path(&path).expect("import");
    let (folded, _, _) =
        materialize_and_resolve(&mut import, &MaterializeConfig::default()).expect("resolve");
    let dag = TritonLowering::new()
        .with_constants(folded)
        .lower(&import.graph, LoweringMode::Inference)
        .expect("the whole graph should lower");

    let mut stubs: BTreeMap<String, usize> = BTreeMap::new();
    for idx in 0..dag.len() {
        let op = dag.node(idx).value.as_ref();
        if op.is_input() {
            continue;
        }
        let Some(runtime_op) = op.runtime_op() else {
            *stubs
                .entry(format!("{} (no runtime op)", op.name()))
                .or_insert(0) += 1;
            continue;
        };
        // The stubs panic rather than returning, so each one is asked in isolation.
        let name = op.name().to_string();
        let answered = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            runtime_op.n_activation_inputs()
        }));
        if answered.is_err() {
            *stubs.entry(name).or_insert(0) += 1;
        }
    }
    if stubs.is_empty() {
        println!("every kernel in the export has a runtime dispatch");
    }
    for (name, count) in &stubs {
        println!("{count:6}  {name}");
    }
}

/// Every distinct kernel the export lowers to, compiled for riscv64.
#[test]
#[ignore = "needs FS_ONNX and a few minutes"]
fn every_kernel_in_the_export_compiles_for_riscv() {
    let path = std::env::var("FS_ONNX").expect("set FS_ONNX to an .onnx path");
    let started = Instant::now();
    let mut import = Onnx::import_from_path(&path).expect("import");
    let (folded, _, _) =
        materialize_and_resolve(&mut import, &MaterializeConfig::default()).expect("resolve");
    println!("imported and folded in {:.1?}", started.elapsed());

    let started = Instant::now();
    let dag = TritonLowering::new()
        .with_constants(folded)
        .lower(&import.graph, LoweringMode::Inference)
        .expect("the whole graph should lower");
    println!("lowered {} nodes in {:.1?}", dag.len(), started.elapsed());

    // One entry per distinct kernel source, with how many nodes share it.
    let mut by_source: BTreeMap<String, (usize, usize)> = BTreeMap::new();
    for idx in 0..dag.len() {
        let op = dag.node(idx).value.as_ref();
        if op.is_input() {
            continue;
        }
        let source = op.forward_kernel_source().to_string();
        if source.is_empty() {
            continue;
        }
        by_source
            .entry(source)
            .and_modify(|(count, _)| *count += 1)
            .or_insert((1, idx));
    }
    println!("{} distinct kernels", by_source.len());

    let target = Target::new(Capability::SpacemitK3);
    let mut failures: Vec<(String, String)> = Vec::new();
    let started = Instant::now();
    for (n, (count, idx)) in by_source.values().enumerate() {
        let op = dag.node(*idx).value.as_ref();
        match teeny_runtime::compile_kernel(&ExecKernel(op), &target, false, false) {
            Ok(_) => {}
            Err(e) => failures.push((
                op.name().to_string(),
                e.to_string()
                    .lines()
                    .take(2)
                    .collect::<Vec<_>>()
                    .join(" | "),
            )),
        }
        if n % 25 == 0 {
            println!(
                "  {n}/{} kernels, {:.0?} elapsed (this one covers {count} nodes)",
                by_source.len(),
                started.elapsed()
            );
        }
    }
    println!(
        "compiled {} kernels in {:.1?}, {} failed",
        by_source.len() - failures.len(),
        started.elapsed(),
        failures.len()
    );
    for (name, why) in &failures {
        println!("  FAILED {name}: {why}");
    }
    assert!(
        failures.is_empty(),
        "{} kernels did not compile",
        failures.len()
    );
}
