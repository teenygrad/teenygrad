// Pad with a constant, over any number of axes.
//
// The older pad kernels are shaped for images: the last two axes of an NCHW tensor.
// The export pads a middle axis of a rank-5 tensor, so this walks the output and works
// out where each element came from, one axis at a time.
//
//   cargo test -p teeny-kernels --test test_pad_nd
//   cargo test -p teeny-kernels --no-default-features --features qemu,hardware \
//       --test test_pad_nd

use teeny_core::graph::{ConstantMap, ConstantTensor, DtypeRepr, Graph, Op};
use teeny_core::model::{Lowering, LoweringMode};
use teeny_kernels::graph::TritonLowering;

/// The shape `pads` gives, as `[lo.., hi..]` over every axis.
fn padded(shape: &[usize], pads: &[i64]) -> Vec<usize> {
    let rank = shape.len();
    (0..rank)
        .map(|ax| (shape[ax] as i64 + pads[ax] + pads[rank + ax]) as usize)
        .collect()
}

fn i64_tensor(values: &[i64]) -> ConstantTensor {
    ConstantTensor {
        dtype: DtypeRepr::I64,
        shape: vec![Some(values.len())],
        bytes: values.iter().flat_map(|v| v.to_le_bytes()).collect(),
    }
}

/// `Pad(x, pads, value)` with the pads and the value both constant, as an exporter
/// writes them.
fn pad_graph(shape: &[usize], pads: &[i64], value: Option<f32>) -> (Graph, ConstantMap) {
    let out = padded(shape, pads);
    let mut g = Graph::new();
    let x = g.add_node(
        Op::Input,
        vec![],
        DtypeRepr::F32,
        shape.iter().map(|&d| Some(d)).collect(),
    );
    let p = g.add_node(
        Op::Constant {
            dtype: DtypeRepr::I64,
            shape: vec![Some(pads.len())],
        },
        vec![],
        DtypeRepr::I64,
        vec![Some(pads.len())],
    );
    let mut constants = ConstantMap::new();
    constants.insert(p, i64_tensor(pads));

    let mut inputs = vec![x, p];
    if let Some(v) = value {
        let c = g.add_node(
            Op::Constant {
                dtype: DtypeRepr::F32,
                shape: vec![],
            },
            vec![],
            DtypeRepr::F32,
            vec![],
        );
        constants.insert(
            c,
            ConstantTensor {
                dtype: DtypeRepr::F32,
                shape: vec![],
                bytes: v.to_le_bytes().to_vec(),
            },
        );
        inputs.push(c);
    }
    let out_node = g.add_node(
        Op::Pad {
            mode: "constant".to_string(),
        },
        inputs,
        DtypeRepr::F32,
        out.iter().map(|&d| Some(d)).collect(),
    );
    g.names.insert(out_node, "the_pad".to_string());
    (g, constants)
}

fn lower(shape: &[usize], pads: &[i64], value: Option<f32>) -> anyhow::Result<usize> {
    let (graph, constants) = pad_graph(shape, pads, value);
    TritonLowering::new()
        .with_constants(constants)
        .lower(&graph, LoweringMode::Inference)
        .map(|dag| dag.len())
}

#[test]
fn the_export_s_shapes_lower() {
    // A middle axis of a rank-5 tensor, which is the export's own form.
    assert_eq!(
        lower(
            &[1, 112, 26, 30, 40],
            &[0, 0, 1, 0, 0, 0, 0, 1, 0, 0],
            Some(0.0)
        )
        .expect("rank 5"),
        2
    );
    // No value operand at all: ONNX pads with zero.
    assert_eq!(lower(&[2, 3], &[1, 0, 1, 0], None).expect("no value"), 2);
}

/// Cropping, which is what a negative pad means, and the output extents say so.
#[test]
fn a_negative_pad_lowers() {
    assert_eq!(lower(&[4, 4], &[-1, 0, 0, -2], Some(0.0)).expect("crop"), 2);
}

#[test]
fn pads_that_do_not_give_the_output_extent_are_refused() {
    let (mut graph, constants) = pad_graph(&[2, 3], &[1, 1, 1, 1], Some(0.0));
    // Claim an output one element wider than the pads call for.
    let last = graph.nodes.len() - 1;
    graph.nodes[last].shape = vec![Some(4), Some(6)];
    let msg = match TritonLowering::new()
        .with_constants(constants)
        .lower(&graph, LoweringMode::Inference)
    {
        Ok(_) => panic!("a disagreeing output extent should not lower"),
        Err(e) => e.to_string(),
    };
    assert!(msg.contains("but the output has 6"), "{msg}");
}

/// Only a constant pad is implemented, and the other modes each need their own
/// coordinate mapping rather than a different fill.
#[test]
fn another_mode_is_refused_by_name() {
    let (mut graph, constants) = pad_graph(&[2, 3], &[1, 1, 1, 1], Some(0.0));
    let last = graph.nodes.len() - 1;
    graph.nodes[last].op = Op::Pad {
        mode: "reflect".to_string(),
    };
    let msg = match TritonLowering::new()
        .with_constants(constants)
        .lower(&graph, LoweringMode::Inference)
    {
        Ok(_) => panic!("reflect should not lower"),
        Err(e) => e.to_string(),
    };
    assert!(msg.contains("\"reflect\""), "{msg}");
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

    /// Padding on the host, by walking the output's coordinates and shifting each.
    fn reference(x: &[f32], shape: &[usize], pads: &[i64], value: f32) -> Vec<f32> {
        let rank = shape.len();
        let out = padded(shape, pads);
        let n: usize = out.iter().product();
        let mut want = vec![value; n];
        let mut in_strides = vec![1usize; rank];
        for ax in (0..rank.saturating_sub(1)).rev() {
            in_strides[ax] = in_strides[ax + 1] * shape[ax + 1];
        }
        for flat in 0..n {
            let mut rest = flat;
            let mut inside = true;
            let mut src = 0usize;
            for ax in (0..rank).rev() {
                let c = (rest % out[ax]) as i64;
                rest /= out[ax];
                let ci = c - pads[ax];
                if ci < 0 || ci >= shape[ax] as i64 {
                    inside = false;
                } else {
                    src += ci as usize * in_strides[ax];
                }
            }
            if inside {
                want[flat] = x[src];
            }
        }
        want
    }

    fn run(shape: &[usize], pads: &[i64], value: f32) {
        let n_in: usize = shape.iter().product();
        // Values distinct from the pad value, so a padded element and a copied one
        // cannot be confused.
        let x: Vec<f32> = (0..n_in).map(|i| i as f32 * 0.5 + 1.0).collect();
        let want = reference(&x, shape, pads, value);

        let (graph, constants) = pad_graph(shape, pads, Some(value));
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

        let feeds: Vec<&[f32]> = vec![&x];
        let outputs = model.terminal_node_indices_sorted_by_size();
        let got = model.run(&device, &feeds, &outputs).expect("run failed");
        let out = got
            .iter()
            .find(|o| o.len() == want.len())
            .unwrap_or_else(|| panic!("no output of length {}", want.len()));
        for i in 0..want.len() {
            assert_eq!(
                out[i], want[i],
                "{shape:?} pads {pads:?}: element {i} is {} not {}",
                out[i], want[i]
            );
        }
    }

    /// Padding both ends of both axes, with a value that is not zero so an unwritten
    /// element and a padded one are different things.
    #[test]
    fn padding_a_matrix_on_every_side() {
        run(&[3, 4], &[1, 2, 2, 1], -7.5);
    }

    /// One axis only, which is the export's case: the others have to come through
    /// untouched.
    #[test]
    fn padding_a_middle_axis_of_a_rank_four_tensor() {
        run(&[2, 3, 2, 3], &[0, 1, 0, 0, 0, 2, 0, 0], 100.0);
    }

    /// A negative pad crops, and a mix of cropping and padding on the same tensor is
    /// where a shift applied in the wrong direction shows up.
    #[test]
    fn a_negative_pad_crops() {
        run(&[4, 5], &[-1, 2, 0, -1], 0.25);
    }

    /// More elements than one block of lanes.
    #[test]
    fn a_tensor_longer_than_a_block() {
        run(&[40, 40], &[1, 1, 1, 1], -1.0);
    }
}
