// Tile: repeat a tensor along each axis.
//
// The index mapping is checked against a host reference that implements ONNX Tile
// directly, rather than against hand-written expected values, so the test covers
// every element of several shapes instead of a few spot checks. The execution tests
// then run the same cases on a device.
//
//   cargo test -p teeny-kernels --test test_tile
//   cargo test -p teeny-kernels --features cuda,hardware --test test_tile
//   cargo test -p teeny-kernels --no-default-features --features qemu,hardware \
//       --test test_tile

use teeny_core::graph::{ConstantMap, ConstantTensor, DtypeRepr, Graph, Op};
use teeny_core::model::{Lowering, LoweringMode, RuntimeOp};
use teeny_kernels::graph::TritonLowering;
use teeny_kernels::nn::tensor::tile::{MAX_RANK, TileRuntimeOp};

/// ONNX `Tile` on the host: `out[coords] = x[coords mod in_shape]`.
fn tile_reference(x: &[f32], in_shape: &[usize], repeats: &[i64]) -> (Vec<f32>, Vec<usize>) {
    let out_shape: Vec<usize> = in_shape
        .iter()
        .zip(repeats)
        .map(|(&d, &r)| d * r as usize)
        .collect();
    let rank = in_shape.len();

    let mut in_strides = vec![1usize; rank];
    for d in (0..rank.saturating_sub(1)).rev() {
        in_strides[d] = in_strides[d + 1] * in_shape[d + 1];
    }
    let mut out_strides = vec![1usize; rank];
    for d in (0..rank.saturating_sub(1)).rev() {
        out_strides[d] = out_strides[d + 1] * out_shape[d + 1];
    }

    let n: usize = out_shape.iter().product();
    let mut out = vec![0.0f32; n];
    for (flat, slot) in out.iter_mut().enumerate() {
        let mut src = 0usize;
        let mut rem = flat;
        for d in 0..rank {
            let coord = rem / out_strides[d];
            rem -= coord * out_strides[d];
            src += (coord % in_shape[d]) * in_strides[d];
        }
        *slot = x[src];
    }
    (out, out_shape)
}

/// Replays the kernel's index arithmetic on the host, from the scalars the runtime op
/// packs. This checks the mapping the device will use, including the left-padding of
/// lower-rank shapes to [`MAX_RANK`], without needing a device.
fn kernel_index_mapping(scalars: &[i32], n: usize) -> Vec<usize> {
    let out_extents = &scalars[1..1 + MAX_RANK];
    let in_extents = &scalars[1 + MAX_RANK..1 + 2 * MAX_RANK];

    // Strides are passed for the outer dimensions; the innermost is the literal 1.
    let passed = &scalars[1 + 2 * MAX_RANK..];
    let mut strides = [1i64; MAX_RANK];
    for d in 0..MAX_RANK - 1 {
        strides[d] = passed[d] as i64;
    }

    (0..n)
        .map(|p| {
            let mut idx = 0i64;
            let mut rem = p as i64;
            for d in (0..MAX_RANK).rev() {
                let s = out_extents[d] as i64;
                let q = rem / s;
                let c = rem - q * s;
                let e = in_extents[d] as i64;
                idx += (c - (c / e) * e) * strides[d];
                rem = q;
            }
            idx as usize
        })
        .collect()
}

fn case(in_shape: &[usize], repeats: &[i64]) {
    let n_in: usize = in_shape.iter().product();
    // Distinct values, so a wrong index cannot coincide with a right one.
    let x: Vec<f32> = (0..n_in).map(|i| i as f32 + 0.5).collect();

    let (want, out_shape) = tile_reference(&x, in_shape, repeats);
    let rop = TileRuntimeOp::<f32>::new(1024, in_shape, repeats)
        .unwrap_or_else(|| panic!("{in_shape:?} x {repeats:?} should be a valid Tile"));

    assert_eq!(
        rop.output_shape(in_shape.len()),
        out_shape,
        "output shape for {in_shape:?} x {repeats:?}"
    );

    let n: usize = out_shape.iter().product();
    let mapping = kernel_index_mapping(&rop.kernel_scalars(n as i32), n);
    let got: Vec<f32> = mapping.iter().map(|&i| x[i]).collect();
    assert_eq!(got, want, "index mapping for {in_shape:?} x {repeats:?}");
}

#[test]
fn the_index_mapping_matches_onnx_tile() {
    // 1-D, the simplest wrap.
    case(&[3], &[4]);
    // Repeating only the inner axis, only the outer, and both.
    case(&[2, 3], &[1, 3]);
    case(&[2, 3], &[3, 1]);
    case(&[2, 3], &[2, 2]);
    // A repeat of 1 is the identity on that axis.
    case(&[4, 5], &[1, 1]);
    // Rank 3 and 4, exercising the left-padding of unused dimensions.
    case(&[2, 3, 2], &[2, 1, 3]);
    case(&[2, 2, 2, 2], &[1, 2, 1, 2]);
    // An axis of extent 1 repeated, which is also what broadcasting does.
    case(&[1, 4], &[5, 1]);
    // Full rank, so nothing is padded.
    case(&[2, 1, 2, 1, 2, 1], &[1, 2, 1, 2, 1, 2]);
}

#[test]
fn invalid_repeats_are_rejected() {
    // One repeat per dimension is required.
    assert!(TileRuntimeOp::<f32>::new(1024, &[2, 3], &[2]).is_none());
    assert!(TileRuntimeOp::<f32>::new(1024, &[2, 3], &[2, 2, 2]).is_none());
    // Negative repeats are not meaningful.
    assert!(TileRuntimeOp::<f32>::new(1024, &[2, 3], &[2, -1]).is_none());
    // Beyond what the kernel indexes.
    let seven = vec![1usize; MAX_RANK + 1];
    let ones = vec![1i64; MAX_RANK + 1];
    assert!(TileRuntimeOp::<f32>::new(1024, &seven, &ones).is_none());
}

/// A repeat of 0 makes the output empty. Nothing is stored, and the kernel must not
/// divide by the zero extent while computing indices for masked lanes.
#[test]
fn a_zero_repeat_gives_an_empty_output() {
    let rop = TileRuntimeOp::<f32>::new(1024, &[2, 3], &[0, 1]).expect("0 is a valid repeat");
    assert_eq!(rop.output_shape(2), vec![0, 3]);
    assert_eq!(rop.grid(&[0, 3]), [0, 1, 1]);
    // The input extents the kernel divides by are clamped away from zero.
    let scalars = rop.kernel_scalars(0);
    let in_extents = &scalars[1 + MAX_RANK..1 + 2 * MAX_RANK];
    assert!(
        in_extents.iter().all(|&e| e >= 1),
        "extents must stay divisible: {in_extents:?}"
    );
}

/// Lowering reads the repeats from the materialised constant rather than from a
/// buffer, which is the whole point of eliding shape operands.
#[test]
fn lowering_takes_repeats_from_a_materialised_constant() {
    let (graph, repeats_node) = tile_graph(&[2, 3], &[2, 2]);
    let mut constants = ConstantMap::new();
    constants.insert(repeats_node, i64_tensor(&[2], &[2, 2]));

    let dag = TritonLowering::new()
        .with_constants(constants)
        .lower(&graph, LoweringMode::Inference)
        .expect("Tile should lower once its repeats are known");
    // input and tile; the repeats operand is metadata, not a DAG node.
    assert_eq!(dag.len(), 2);
}

#[test]
fn lowering_without_the_repeats_says_so() {
    let (graph, _) = tile_graph(&[2, 3], &[2, 2]);
    let msg = match TritonLowering::new().lower(&graph, LoweringMode::Inference) {
        Ok(_) => panic!("Tile cannot lower without its repeats"),
        Err(e) => e.to_string(),
    };
    assert!(
        msg.contains("no value for its repeats operand"),
        "should name the missing operand: {msg}"
    );
}

#[test]
fn lowering_rejects_repeats_that_do_not_match_the_input() {
    let (graph, repeats_node) = tile_graph(&[2, 3], &[2, 2]);
    let mut constants = ConstantMap::new();
    // Three repeats for a rank-2 input.
    constants.insert(repeats_node, i64_tensor(&[3], &[2, 2, 2]));

    let msg = match TritonLowering::new()
        .with_constants(constants)
        .lower(&graph, LoweringMode::Inference)
    {
        Ok(_) => panic!("mismatched repeats should not lower"),
        Err(e) => e.to_string(),
    };
    assert!(msg.contains("do not match input shape"), "{msg}");
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

/// `Tile(x, repeats)`, returning the graph and the repeats node's index.
fn tile_graph(in_shape: &[usize], repeats: &[i64]) -> (Graph, usize) {
    let mut g = Graph::new();
    let x = g.add_node(
        Op::Input,
        vec![],
        DtypeRepr::F32,
        in_shape.iter().map(|&d| Some(d)).collect(),
    );
    let r = g.add_node(
        Op::Constant {
            dtype: DtypeRepr::I64,
            shape: vec![Some(repeats.len())],
        },
        vec![],
        DtypeRepr::I64,
        vec![Some(repeats.len())],
    );
    let out: Vec<Option<usize>> = in_shape
        .iter()
        .zip(repeats)
        .map(|(&d, &k)| Some(d * k as usize))
        .collect();
    g.add_node(Op::Tile, vec![x, r], DtypeRepr::F32, out);
    (g, r)
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

    fn run_case(in_shape: &[usize], repeats: &[i64]) {
        let n_in: usize = in_shape.iter().product();
        let x: Vec<f32> = (0..n_in).map(|i| i as f32 + 0.5).collect();
        let (want, _) = tile_reference(&x, in_shape, repeats);

        let (graph, repeats_node) = tile_graph(in_shape, repeats);
        let mut constants = ConstantMap::new();
        constants.insert(repeats_node, i64_tensor(&[repeats.len()], repeats));

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

        let mismatches = (0..want.len()).filter(|&i| out[i] != want[i]).count();
        if mismatches > 0 {
            println!("  got  = {out:?}");
            println!("  want = {want:?}");
            let bad: Vec<usize> = (0..want.len()).filter(|&i| out[i] != want[i]).collect();
            println!("  bad indices = {bad:?}");
        }
        assert_eq!(
            mismatches,
            0,
            "{in_shape:?} x {repeats:?}: {mismatches} of {} elements wrong",
            want.len()
        );
        println!(
            "tile {in_shape:?} x {repeats:?} -> {} elements, exact",
            want.len()
        );
    }

    /// The index mapping is also checked against a reference above without a device.
    #[test]
    fn tile_runs_on_device() {
        run_case(&[3], &[4]);
        run_case(&[2, 3], &[2, 2]);
        run_case(&[2, 3], &[1, 3]);
        run_case(&[2, 3, 2], &[2, 1, 3]);
        run_case(&[1, 4], &[5, 1]);
    }
}
