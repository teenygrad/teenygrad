// Proves a data constant's values reach the device and arrive at the output.
//
// Eliding shape-metadata constants is checked in test_constant_elision.rs without a
// device, because nothing has to execute. A constant that flows data is the opposite
// case: the only way to know its bytes made it to the right buffer, in the right
// order and the right dtype, is to run the graph and compare against values computed
// on the host.
//
// The execution test needs a real device for the compiled-in backend, so it is gated
// on `hardware`, per the convention in teeny-kernels' Cargo.toml:
//
//   cargo test -p teeny-kernels --features cuda,hardware --test test_constant_binding
//   cargo test -p teeny-kernels --no-default-features --features qemu,hardware \
//       --test test_constant_binding

#![cfg(any(feature = "cuda", feature = "riscv"))]

use teeny_core::graph::{ConstantMap, ConstantTensor, DtypeRepr, Graph, Op};
use teeny_core::model::LoweringMode;
use teeny_kernels::graph::TritonLowering;
#[cfg(feature = "hardware")]
use teeny_runtime::Target;
#[cfg(feature = "hardware")]
use teeny_test::model::RuntimeModel;

fn f32_tensor(shape: &[usize], vals: &[f32]) -> ConstantTensor {
    let mut bytes = Vec::new();
    for v in vals {
        bytes.extend_from_slice(&v.to_le_bytes());
    }
    ConstantTensor {
        dtype: DtypeRepr::F32,
        shape: shape.iter().map(|&d| Some(d)).collect(),
        bytes,
    }
}

#[cfg(all(feature = "cuda", feature = "hardware"))]
fn device_and_target<'a>() -> (teeny_runtime::Device<'a>, Target) {
    let env = teeny_test::cuda::setup_cuda_env().expect("cuda env");
    let target = Target::new(env.capability);
    (env.device, target)
}

#[cfg(all(feature = "riscv", feature = "hardware", not(feature = "cuda")))]
fn device_and_target<'a>() -> (teeny_runtime::Device<'a>, Target) {
    use teeny_riscv::compiler::target::Capability;
    (
        teeny_runtime::Device::new(Default::default()),
        Target::new(Capability::SpacemitK3),
    )
}

/// `y = x + c`, with `c` a materialised constant rather than a fed input.
///
/// The constant's values are deliberately not round numbers and not symmetric, so a
/// byte-order mistake, an off-by-one or a buffer shared with the input would all show
/// up as a mismatch rather than coincidentally passing.
#[test]
#[cfg(feature = "hardware")]
fn a_bound_constant_reaches_the_output() {
    dotenv::dotenv().ok();
    let (device, target) = device_and_target();

    const N: usize = 8;
    let c: Vec<f32> = vec![1.5, -2.25, 0.125, 7.0, -0.5, 3.375, -9.75, 0.0625];
    let x: Vec<f32> = (0..N).map(|i| i as f32 * 0.5 - 1.0).collect();

    let mut graph = Graph::new();
    let x_node = graph.add_node(Op::Input, vec![], DtypeRepr::F32, vec![Some(N)]);
    let c_node = graph.add_node(
        Op::Constant {
            dtype: DtypeRepr::F32,
            shape: vec![Some(N)],
        },
        vec![],
        DtypeRepr::F32,
        vec![Some(N)],
    );
    graph.names.insert(c_node, "folded_bias".to_string());
    let _add = graph.add_node(Op::Add, vec![x_node, c_node], DtypeRepr::F32, vec![Some(N)]);

    let mut constants = ConstantMap::new();
    constants.insert(c_node, f32_tensor(&[N], &c));

    let lowering = TritonLowering::new().with_constants(constants);
    let mut model = RuntimeModel::compile(
        &device,
        &graph,
        &lowering,
        LoweringMode::Inference,
        &target,
        1,
        false,
    )
    .expect("a data constant with its value supplied should compile");

    // The constant is a real DAG node with a parameter slot, already populated from
    // `param_init_data`. Nothing binds it by hand, which is the point: it is visible
    // as a parameter but needs no checkpoint.
    let slots: Vec<String> = model.param_info_named().map(|(name, _, _)| name).collect();
    assert!(
        slots.iter().any(|s| s == "folded_bias.value"),
        "the constant should expose a named slot, got {slots:?}"
    );

    // The constant takes no feed: it is not an op-less node. Only `x` is fed.
    let feeds: Vec<&[f32]> = vec![&x];
    let outputs = model.terminal_node_indices_sorted_by_size();
    let got = model.run(&device, &feeds, &outputs).expect("run failed");

    let out = got
        .iter()
        .find(|o| o.len() == N)
        .unwrap_or_else(|| panic!("no output of length {N}: {:?}", lens(&got)));

    let mut max = 0.0f32;
    for i in 0..N {
        let want = x[i] + c[i];
        let d = (out[i] - want).abs();
        if d > max {
            max = d;
        }
    }
    println!("x + c: max |device - host| = {max:e}");
    assert!(
        max < 1e-6,
        "constant did not arrive intact: {max:e}, got {out:?}"
    );
}

/// A constant whose bytes do not fill its shape is caught at lowering, not on the
/// device, where it would read past the end of its slot.
#[test]
fn a_constant_with_short_bytes_is_rejected() {
    use teeny_core::model::Lowering;

    let mut graph = Graph::new();
    let x = graph.add_node(Op::Input, vec![], DtypeRepr::F32, vec![Some(4)]);
    let c = graph.add_node(
        Op::Constant {
            dtype: DtypeRepr::F32,
            shape: vec![Some(4)],
        },
        vec![],
        DtypeRepr::F32,
        vec![Some(4)],
    );
    let _add = graph.add_node(Op::Add, vec![x, c], DtypeRepr::F32, vec![Some(4)]);

    let mut constants = ConstantMap::new();
    // Three elements for a four-element tensor.
    constants.insert(c, f32_tensor(&[4], &[1.0, 2.0, 3.0]));

    let msg = match TritonLowering::new()
        .with_constants(constants)
        .lower(&graph, LoweringMode::Inference)
    {
        Ok(_) => panic!("a short constant should not lower"),
        Err(e) => e.to_string(),
    };
    assert!(
        msg.contains("do not fill"),
        "should report the size disagreement: {msg}"
    );
}

#[cfg(feature = "hardware")]
fn lens(v: &[Vec<f32>]) -> Vec<usize> {
    v.iter().map(|o| o.len()).collect()
}

/// A `Bool` constant lowers, which it could not before: the dtype dispatch had no arm
/// for it, so every mask an exporter folded into a constant stopped the import.
///
/// Bool is not a `Num` dtype and never will be, so it does not go through the dispatch
/// macros. It works here because a bool is one byte, a Bool buffer is allocated at one
/// byte per element, and this kernel only moves bytes, so the `u8` instantiation copies
/// the mask verbatim and the node keeps its Bool dtype for its consumers.
#[test]
fn a_bool_constant_lowers() {
    use teeny_core::model::Lowering;

    const N: usize = 5;
    let mut graph = Graph::new();
    let c = graph.add_node(
        Op::Constant {
            dtype: DtypeRepr::Bool,
            shape: vec![Some(N)],
        },
        vec![],
        DtypeRepr::Bool,
        vec![Some(N)],
    );
    graph.names.insert(c, "folded_mask".to_string());

    let mut constants = ConstantMap::new();
    constants.insert(
        c,
        ConstantTensor {
            dtype: DtypeRepr::Bool,
            shape: vec![Some(N)],
            bytes: vec![1, 0, 1, 1, 0],
        },
    );

    TritonLowering::new()
        .with_constants(constants)
        .lower(&graph, LoweringMode::Inference)
        .expect("a Bool constant should lower");
}

/// The byte count is still checked for Bool: one byte per element, not the four a
/// carrier-field decode would have produced.
#[test]
fn a_bool_constant_is_sized_at_one_byte_per_element() {
    use teeny_core::model::Lowering;

    let mut graph = Graph::new();
    let c = graph.add_node(
        Op::Constant {
            dtype: DtypeRepr::Bool,
            shape: vec![Some(4)],
        },
        vec![],
        DtypeRepr::Bool,
        vec![Some(4)],
    );

    let mut constants = ConstantMap::new();
    constants.insert(
        c,
        ConstantTensor {
            dtype: DtypeRepr::Bool,
            shape: vec![Some(4)],
            bytes: vec![1, 0],
        },
    );

    let msg = match TritonLowering::new()
        .with_constants(constants)
        .lower(&graph, LoweringMode::Inference)
    {
        Ok(_) => panic!("two bytes should not fill a four-element bool tensor"),
        Err(e) => e.to_string(),
    };
    assert!(msg.contains("do not fill"), "unexpected error: {msg}");
}
