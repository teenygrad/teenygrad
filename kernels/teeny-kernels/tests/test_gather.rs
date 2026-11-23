// Gather: select slices along one axis by index.
//
// The index mapping is checked against a host implementation of ONNX Gather over a
// range of shapes, axes and index patterns, element for element. The execution tests
// then run the same cases on a device.
//
//   cargo test -p teeny-kernels --test test_gather
//   cargo test -p teeny-kernels --no-default-features --features qemu,hardware \
//       --test test_gather

use teeny_core::graph::{ConstantMap, ConstantTensor, DtypeRepr, Graph, Op};
use teeny_core::model::{Lowering, LoweringMode, RuntimeOp};
use teeny_kernels::graph::TritonLowering;
use teeny_kernels::nn::tensor::gather::{GatherError, GatherRuntimeOp};

/// ONNX `Gather` on the host.
fn gather_reference(
    x: &[f32],
    in_shape: &[usize],
    axis: usize,
    indices: &[i64],
) -> (Vec<f32>, Vec<usize>) {
    let dim = in_shape[axis];
    let outer: usize = in_shape[..axis].iter().product();
    let inner: usize = in_shape[axis + 1..].iter().product();

    // A rank-1 index tensor keeps the axis; the shape is handled by the caller.
    let out_shape: Vec<usize> = in_shape[..axis]
        .iter()
        .copied()
        .chain(core::iter::once(indices.len()))
        .chain(in_shape[axis + 1..].iter().copied())
        .collect();

    let mut out = Vec::with_capacity(outer * indices.len() * inner);
    for o in 0..outer {
        for &i in indices {
            let norm = if i < 0 { i + dim as i64 } else { i } as usize;
            for n in 0..inner {
                out.push(x[o * dim * inner + norm * inner + n]);
            }
        }
    }
    (out, out_shape)
}

/// Replays the kernel's index arithmetic on the host from the scalars the runtime op
/// packs, so the mapping the device will use is checked without needing a device.
fn kernel_index_mapping(scalars: [i32; 4], idx: &[i32], n: usize) -> Vec<usize> {
    let [_, inner, out_block, outer_stride] = scalars;
    (0..n)
        .map(|p| {
            let p = p as i32;
            let o = p / out_block;
            let rest = p - o * out_block;
            let k = rest / inner;
            let i = rest - k * inner;
            (o * outer_stride + idx[k as usize] * inner + i) as usize
        })
        .collect()
}

fn case(in_shape: &[usize], axis: i64, indices: &[i64]) {
    let n_in: usize = in_shape.iter().product();
    // Distinct values, so a wrong index cannot coincide with a right one.
    let x: Vec<f32> = (0..n_in).map(|i| i as f32 + 0.5).collect();

    let rank = in_shape.len() as i64;
    let ax = (if axis < 0 { axis + rank } else { axis }) as usize;
    let (want, out_shape) = gather_reference(&x, in_shape, ax, indices);

    let rop = GatherRuntimeOp::<f32>::new(1024, in_shape, axis, indices)
        .unwrap_or_else(|e| panic!("{in_shape:?} axis {axis} {indices:?}: {e}"));

    assert_eq!(
        GatherRuntimeOp::<f32>::output_shape(in_shape, axis, &[indices.len()]),
        out_shape,
        "output shape for {in_shape:?} axis {axis}"
    );

    // Normalised indices, as the runtime op stores them.
    let dim = in_shape[ax] as i64;
    let idx: Vec<i32> = indices
        .iter()
        .map(|&i| (if i < 0 { i + dim } else { i }) as i32)
        .collect();

    let n: usize = out_shape.iter().product();
    let mapping = kernel_index_mapping(rop.kernel_scalars(n as i32), &idx, n);
    let got: Vec<f32> = mapping.iter().map(|&i| x[i]).collect();
    assert_eq!(
        got, want,
        "index mapping for {in_shape:?} axis {axis} indices {indices:?}"
    );
}

#[test]
fn test_the_index_mapping_matches_onnx_gather() {
    // 1-D: a plain select, and a permutation with repeats.
    case(&[5], 0, &[0]);
    case(&[5], 0, &[4, 0, 2, 2]);
    // 2-D along each axis.
    case(&[4, 3], 0, &[2]);
    case(&[4, 3], 0, &[3, 1, 0]);
    case(&[4, 3], 1, &[1]);
    case(&[4, 3], 1, &[2, 0]);
    // 3-D, including the middle axis, which has both outer and inner extent.
    case(&[2, 4, 3], 1, &[3, 0]);
    case(&[2, 4, 3], 2, &[2, 1, 0]);
    case(&[2, 4, 3], 0, &[1, 1]);
    // Negative axis and negative indices, both of which ONNX allows.
    case(&[4, 3], -1, &[-1, -3]);
    case(&[2, 4, 3], -2, &[-4, 2]);
    // More indices than the axis has elements: gather may repeat.
    case(&[3], 0, &[0, 1, 2, 2, 1, 0, 0]);
    // An axis of extent 1.
    case(&[2, 1, 3], 1, &[0]);
}

#[test]
fn test_an_out_of_range_index_is_rejected() {
    assert_eq!(
        GatherRuntimeOp::<f32>::new(1024, &[4, 3], 0, &[4]).err(),
        Some(GatherError::IndexOutOfRange { index: 4, dim: 4 })
    );
    assert_eq!(
        GatherRuntimeOp::<f32>::new(1024, &[4, 3], 0, &[-5]).err(),
        Some(GatherError::IndexOutOfRange { index: -5, dim: 4 })
    );
    // The boundary cases are valid.
    assert!(GatherRuntimeOp::<f32>::new(1024, &[4, 3], 0, &[3]).is_ok());
    assert!(GatherRuntimeOp::<f32>::new(1024, &[4, 3], 0, &[-4]).is_ok());
}

#[test]
fn test_an_out_of_range_axis_is_rejected() {
    assert_eq!(
        GatherRuntimeOp::<f32>::new(1024, &[4, 3], 2, &[0]).err(),
        Some(GatherError::AxisOutOfRange { axis: 2, rank: 2 })
    );
    assert_eq!(
        GatherRuntimeOp::<f32>::new(1024, &[4, 3], -3, &[0]).err(),
        Some(GatherError::AxisOutOfRange { axis: -3, rank: 2 })
    );
}

#[test]
fn test_an_empty_gathered_axis_is_rejected() {
    assert_eq!(
        GatherRuntimeOp::<f32>::new(1024, &[0, 3], 0, &[0]).err(),
        Some(GatherError::EmptyAxis)
    );
}

/// A scalar index tensor removes the axis rather than keeping it at extent 1. This is
/// the case that dominates real models: 611 of the 613 constant-index Gather nodes in
/// the FoundationStereo export select a single slice.
#[test]
fn test_a_scalar_index_removes_the_axis() {
    assert_eq!(
        GatherRuntimeOp::<f32>::output_shape(&[2, 4, 3], 1, &[]),
        vec![2, 3]
    );
    assert_eq!(
        GatherRuntimeOp::<f32>::output_shape(&[2, 4, 3], 1, &[1]),
        vec![2, 1, 3]
    );
}

/// The index slot is sized in elements of the data's dtype but holds `i32` indices, so
/// it has to be long enough whatever the dtype, and its byte count has to match the
/// slot exactly: that is `param_init_data`'s contract.
#[test]
fn test_the_index_slot_fits_whatever_the_data_dtype() {
    for n_idx in [1usize, 2, 3, 7, 8] {
        let indices: Vec<i64> = (0..n_idx as i64).map(|i| i % 4).collect();

        let f32_op = GatherRuntimeOp::<f32>::new(1024, &[4, 3], 0, &indices).unwrap();
        let shape = f32_op.param_shapes(&[], &[]);
        let bytes = f32_op.param_init_data(0).unwrap();
        assert_eq!(bytes.len(), shape[0][0] * 4, "f32 slot, {n_idx} indices");
        assert!(bytes.len() >= n_idx * 4);

        let f64_op = GatherRuntimeOp::<f64>::new(1024, &[4, 3], 0, &indices).unwrap();
        let shape = f64_op.param_shapes(&[], &[]);
        let bytes = f64_op.param_init_data(0).unwrap();
        assert_eq!(bytes.len(), shape[0][0] * 8, "f64 slot, {n_idx} indices");
        assert!(bytes.len() >= n_idx * 4);
    }
}

#[test]
fn test_lowering_takes_indices_from_a_materialised_constant() {
    let (graph, idx_node) = gather_graph(&[4, 3], 0, &[2, 0]);
    let mut constants = ConstantMap::new();
    constants.insert(idx_node, i64_tensor(&[2], &[2, 0]));

    let dag = TritonLowering::new()
        .with_constants(constants)
        .lower(&graph, LoweringMode::Inference)
        .expect("Gather should lower once its indices are known");
    // data and gather; the indices live in a parameter slot, not a DAG node.
    assert_eq!(dag.len(), 2);
}

/// Indices only known at run time lower to the runtime-index kernel, which reads them
/// from their own buffer rather than a parameter slot.
#[test]
fn test_lowering_reads_runtime_indices_from_their_buffer() {
    // An index tensor produced by an op, so no constant value exists for it: an
    // embedding lookup of token ids. It lowers to the runtime-index kernel, which takes
    // the indices as a second activation.
    let (g, _) = runtime_index_graph(&[4, 3], 0, 2);
    let dag = TritonLowering::new()
        .lower(&g, LoweringMode::Inference)
        .expect("runtime indices lower");
    assert!(dag.len() >= 3, "data, indices and the gather");
}

/// `Gather(data, Identity(indices), axis)` with `n_idx` int64 indices as a graph input,
/// so the indices are only known at run time. Returns the graph and its output node.
fn runtime_index_graph(in_shape: &[usize], axis: i64, n_idx: usize) -> (Graph, usize) {
    let mut g = Graph::new();
    let shape = |s: &[usize]| s.iter().map(|&d| Some(d)).collect::<Vec<_>>();
    let data = g.add_node(Op::Input, vec![], DtypeRepr::F32, shape(in_shape));
    let raw = g.add_node(Op::Input, vec![], DtypeRepr::I64, vec![Some(n_idx)]);
    let idx = g.add_node(Op::Identity, vec![raw], DtypeRepr::I64, vec![Some(n_idx)]);
    let out_shape = GatherRuntimeOp::<f32>::output_shape(in_shape, axis, &[n_idx]);
    let out = g.add_node(
        Op::Gather { axis },
        vec![data, idx],
        DtypeRepr::F32,
        shape(&out_shape),
    );
    (g, out)
}

#[test]
fn test_lowering_reports_a_bad_index() {
    let (graph, idx_node) = gather_graph(&[4, 3], 0, &[2]);
    let mut constants = ConstantMap::new();
    constants.insert(idx_node, i64_tensor(&[1], &[9]));

    let msg = match TritonLowering::new()
        .with_constants(constants)
        .lower(&graph, LoweringMode::Inference)
    {
        Ok(_) => panic!("an out-of-range index should not lower"),
        Err(e) => e.to_string(),
    };
    assert!(msg.contains("index 9 is out of range"), "{msg}");
}

fn i64_tensor(shape: &[usize], vals: &[i64]) -> ConstantTensor {
    let mut bytes = Vec::new();
    for v in vals {
        bytes.extend_from_slice(&v.to_le_bytes());
    }
    ConstantTensor {
        dtype: DtypeRepr::I64,
        shape: shape.iter().map(|&d| Some(d)).collect(),
        bytes,
    }
}

/// `Gather(x, indices, axis)`, returning the graph and the indices node's index.
fn gather_graph(in_shape: &[usize], axis: i64, indices: &[i64]) -> (Graph, usize) {
    let mut g = Graph::new();
    let x = g.add_node(
        Op::Input,
        vec![],
        DtypeRepr::F32,
        in_shape.iter().map(|&d| Some(d)).collect(),
    );
    let i = g.add_node(
        Op::Constant {
            dtype: DtypeRepr::I64,
            shape: vec![Some(indices.len())],
        },
        vec![],
        DtypeRepr::I64,
        vec![Some(indices.len())],
    );
    let out = GatherRuntimeOp::<f32>::output_shape(in_shape, axis, &[indices.len()]);
    g.add_node(
        Op::Gather { axis },
        vec![x, i],
        DtypeRepr::F32,
        out.iter().map(|&d| Some(d)).collect(),
    );
    (g, i)
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

    fn run_case(in_shape: &[usize], axis: i64, indices: &[i64]) {
        let n_in: usize = in_shape.iter().product();
        let x: Vec<f32> = (0..n_in).map(|i| i as f32 + 0.5).collect();
        let rank = in_shape.len() as i64;
        let ax = (if axis < 0 { axis + rank } else { axis }) as usize;
        let (want, _) = gather_reference(&x, in_shape, ax, indices);

        let (graph, idx_node) = gather_graph(in_shape, axis, indices);
        let mut constants = ConstantMap::new();
        constants.insert(idx_node, i64_tensor(&[indices.len()], indices));

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

        let wrong = (0..want.len()).filter(|&i| out[i] != want[i]).count();
        assert_eq!(
            wrong,
            0,
            "{in_shape:?} axis {axis} {indices:?}: {wrong} of {} wrong",
            want.len()
        );
        println!(
            "gather {in_shape:?} axis {axis} {indices:?} -> {} elements, exact",
            want.len()
        );
    }

    /// The same gathers with the indices fed at run time as an int64 tensor, which is
    /// how an embedding reads token ids: negative indices wrap as ONNX defines.
    fn run_runtime_case(in_shape: &[usize], axis: i64, indices: &[i64]) {
        let n_in: usize = in_shape.iter().product();
        let x: Vec<f32> = (0..n_in).map(|i| i as f32 + 0.5).collect();
        let rank = in_shape.len() as i64;
        let ax = (if axis < 0 { axis + rank } else { axis }) as usize;
        let (want, _) = gather_reference(&x, in_shape, ax, indices);

        let (graph, _) = runtime_index_graph(in_shape, axis, indices.len());
        let (device, target) = device_and_target();
        let mut model = RuntimeModel::compile(
            &device,
            &graph,
            &TritonLowering::new(),
            LoweringMode::Inference,
            &target,
            1,
            false,
        )
        .expect("compile failed");

        // The executor's buffers are f32 words; an int64 feed travels as its raw bytes.
        let idx_words: Vec<f32> = indices
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect::<Vec<u8>>()
            .chunks(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        let feeds: Vec<&[f32]> = vec![&x, &idx_words];
        let outputs = model.terminal_node_indices_sorted_by_size();
        let got = model.run(&device, &feeds, &outputs).expect("run failed");
        let out = got
            .iter()
            .find(|o| o.len() == want.len())
            .unwrap_or_else(|| panic!("no output of length {}", want.len()));
        let wrong = (0..want.len()).filter(|&i| out[i] != want[i]).count();
        assert_eq!(
            wrong,
            0,
            "runtime {in_shape:?} axis {axis} {indices:?}: {wrong} of {} wrong",
            want.len()
        );
    }

    #[test]
    fn test_gather_with_runtime_indices_runs_on_device() {
        // An embedding lookup: rows of a [vocab, hidden] table by token id.
        run_runtime_case(&[6, 4], 0, &[5, 0, 3, 3, 1]);
        run_runtime_case(&[6, 4], 0, &[-1, -6, 2]);
        run_runtime_case(&[2, 4, 3], 1, &[3, 0]);
        run_runtime_case(&[2, 4, 3], -1, &[-1, 0]);
    }

    #[test]
    fn test_gather_runs_on_device() {
        run_case(&[5], 0, &[4, 0, 2, 2]);
        run_case(&[4, 3], 0, &[3, 1, 0]);
        run_case(&[4, 3], 1, &[2, 0]);
        run_case(&[2, 4, 3], 1, &[3, 0]);
        run_case(&[2, 4, 3], -1, &[-1, 0]);
        // The single-slice select, the dominant case in real models.
        run_case(&[2, 4, 3], 1, &[2]);
    }
}
