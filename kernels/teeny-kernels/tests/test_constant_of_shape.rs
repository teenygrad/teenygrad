// ConstantOfShape: a tensor of one repeated value, filled on the device.
//
// The constant folder evaluates these when the result is small, and declines when it
// is not: the export's largest is 55.9 million elements, which is 224 MB of graph data
// for a value that is already known. Those are the ones that reach lowering, so what
// matters here is that the fill arrives intact without the tensor ever being held as
// data.
//
//   cargo test -p teeny-kernels --test test_constant_of_shape
//   cargo test -p teeny-kernels --no-default-features --features qemu,hardware \
//       --test test_constant_of_shape

use teeny_core::graph::{ConstantMap, ConstantTensor, DtypeRepr, Graph, Op};
use teeny_core::model::{Lowering, LoweringMode};
use teeny_kernels::graph::TritonLowering;

/// A `ConstantOfShape` of `shape` filled with `fill`'s bytes, its shape operand a
/// materialised constant as the importer and the folder leave it.
fn graph_of(dtype: DtypeRepr, shape: &[usize], fill: &[u8]) -> (Graph, ConstantMap, usize) {
    let mut g = Graph::new();
    let dims = g.add_node(
        Op::Constant {
            dtype: DtypeRepr::I64,
            shape: vec![Some(shape.len())],
        },
        vec![],
        DtypeRepr::I64,
        vec![Some(shape.len())],
    );
    let out = g.add_node(
        Op::ConstantOfShape {
            dtype,
            fill: fill.to_vec(),
        },
        vec![dims],
        dtype,
        shape.iter().map(|&d| Some(d)).collect(),
    );
    g.names.insert(out, "the_fill".to_string());

    let mut bytes = Vec::new();
    for &d in shape {
        bytes.extend_from_slice(&(d as i64).to_le_bytes());
    }
    let mut constants = ConstantMap::new();
    constants.insert(
        dims,
        ConstantTensor {
            dtype: DtypeRepr::I64,
            shape: vec![Some(shape.len())],
            bytes,
        },
    );
    (g, constants, out)
}

fn lower(dtype: DtypeRepr, shape: &[usize], fill: &[u8]) -> anyhow::Result<usize> {
    let (graph, constants, _) = graph_of(dtype, shape, fill);
    TritonLowering::new()
        .with_constants(constants)
        .lower(&graph, LoweringMode::Inference)
        .map(|dag| dag.len())
}

/// The shape operand is metadata, so lowering reads it and elides the node: what is
/// left is the fill alone, with nothing carrying the dims to the device.
#[test]
fn the_shape_operand_is_elided() {
    assert_eq!(
        lower(DtypeRepr::F32, &[2, 3, 4], &0.0f32.to_le_bytes()).expect("f32"),
        1
    );
}

#[test]
fn every_buffer_dtype_fills() {
    for (dtype, width) in [
        (DtypeRepr::F32, 4),
        (DtypeRepr::F64, 8),
        (DtypeRepr::I8, 1),
        (DtypeRepr::I16, 2),
        (DtypeRepr::I32, 4),
        (DtypeRepr::I64, 8),
        (DtypeRepr::U8, 1),
        (DtypeRepr::U16, 2),
        (DtypeRepr::U32, 4),
        (DtypeRepr::U64, 8),
        (DtypeRepr::Bool, 1),
    ] {
        let fill = vec![1u8; width];
        assert_eq!(
            lower(dtype, &[4, 5], &fill).unwrap_or_else(|e| panic!("{dtype:?}: {e}")),
            1
        );
    }
}

/// ONNX fills with zero when the node has no `value` attribute, which the importer
/// records as no bytes at all rather than as a zero of some guessed width.
#[test]
fn an_absent_value_fills_with_zero() {
    assert_eq!(lower(DtypeRepr::F32, &[3], &[]).expect("default fill"), 1);
}

/// A fill that is not one element wide would be read past the end of its slot on the
/// device, so it is refused at lowering with the widths named.
#[test]
fn a_fill_that_is_not_one_element_is_refused() {
    let msg = match lower(DtypeRepr::F32, &[3], &[1, 2]) {
        Ok(_) => panic!("two bytes should not fill an f32"),
        Err(e) => e.to_string(),
    };
    assert!(msg.contains("is 2 bytes"), "{msg}");
    assert!(msg.contains("F32"), "{msg}");
}

/// Without a shape there is nothing to allocate or fill. The shapes that get here are
/// large, so guessing would mean a buffer of the wrong size rather than a small slip.
#[test]
fn an_unshaped_fill_is_refused() {
    let mut g = Graph::new();
    g.add_node(
        Op::ConstantOfShape {
            dtype: DtypeRepr::F32,
            fill: 0.0f32.to_le_bytes().to_vec(),
        },
        vec![],
        DtypeRepr::F32,
        vec![],
    );
    let msg = match TritonLowering::new().lower(&g, LoweringMode::Inference) {
        Ok(_) => panic!("an unshaped fill should not lower"),
        Err(e) => e.to_string(),
    };
    assert!(msg.contains("shape is empty"), "{msg}");
}

// ---------------------------------------------------------------------------
// Execution
// ---------------------------------------------------------------------------

#[cfg(all(feature = "hardware", any(feature = "cuda", feature = "riscv")))]
mod execution {
    use super::*;
    use teeny_runtime::Target;
    use teeny_test::model::RuntimeModel;

    #[cfg(feature = "cuda")]
    fn device_and_target<'a>() -> (teeny_runtime::Device<'a>, Target) {
        let env = teeny_test::cuda::setup_cuda_env().expect("cuda env");
        let target = Target::new(env.capability);
        (env.device, target)
    }

    #[cfg(all(feature = "riscv", not(feature = "cuda")))]
    fn device_and_target<'a>() -> (teeny_runtime::Device<'a>, Target) {
        use teeny_riscv::compiler::target::Capability;
        (
            teeny_runtime::Device::new(Default::default()),
            Target::new(Capability::SpacemitK3),
        )
    }

    /// Runs an f32 fill and returns the output.
    fn run(shape: &[usize], value: f32) -> Vec<f32> {
        let (graph, constants, _) = graph_of(DtypeRepr::F32, shape, &value.to_le_bytes());
        let (device, target) = device_and_target();
        let mut model = RuntimeModel::compile(
            &device,
            &graph,
            &TritonLowering::new().with_constants(constants),
            LoweringMode::Inference,
            &target,
            1,
            false,
        )
        .expect("compile failed");

        // The fill's slot is populated from `param_init_data`, so nothing binds it,
        // but it is still a named parameter of the model.
        let slots: Vec<String> = model.param_info_named().map(|(n, _, _)| n).collect();
        assert!(
            slots.iter().any(|s| s == "the_fill.value"),
            "the fill should expose a named slot, got {slots:?}"
        );

        let feeds: Vec<&[f32]> = vec![];
        let outputs = model.terminal_node_indices_sorted_by_size();
        let got = model.run(&device, &feeds, &outputs).expect("run failed");
        let n: usize = shape.iter().product();
        got.into_iter()
            .find(|o| o.len() == n)
            .unwrap_or_else(|| panic!("no output of length {n}"))
    }

    /// A value that is neither zero nor round: a byte-order mistake or a slot read at
    /// the wrong offset would show up rather than passing by coincidence.
    #[test]
    fn the_value_reaches_every_element() {
        let out = run(&[3, 5], -2.359375);
        assert_eq!(out.len(), 15);
        for (i, v) in out.iter().enumerate() {
            assert_eq!(*v, -2.359375, "element {i} of the fill is {v}");
        }
    }

    #[test]
    fn a_zero_fill_is_zero_everywhere() {
        assert!(run(&[2, 2, 2], 0.0).iter().all(|&v| v == 0.0));
    }

    /// More elements than one block of lanes, and not a multiple of one either, so the
    /// last block is partly masked off.
    #[test]
    fn an_output_longer_than_a_block_is_filled_whole() {
        let out = run(&[3000], 1.25);
        assert_eq!(out.len(), 3000);
        assert!(out.iter().all(|&v| v == 1.25), "the tail was not filled");
    }
}
