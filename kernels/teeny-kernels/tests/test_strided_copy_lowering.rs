// Transpose, Slice and Expand, lowered onto the one strided-copy kernel.
//
// Each case is checked against a host reference that implements the ONNX operator
// directly, by replaying the index arithmetic the kernel will perform from the very
// scalars the runtime op packs. That covers every element of several shapes with no
// device, and checks the mapping rather than merely that something lowered.
//
//   cargo test -p teeny-kernels --test test_strided_copy_lowering
//   cargo test -p teeny-kernels --features cuda,hardware --test test_strided_copy_lowering
//   cargo test -p teeny-kernels --no-default-features --features qemu,hardware \
//       --test test_strided_copy_lowering

use teeny_core::graph::{ConstantMap, ConstantTensor, DtypeRepr, Graph, Op, slice_plan};
use teeny_core::model::{Lowering, LoweringMode};
use teeny_kernels::graph::TritonLowering;
use teeny_kernels::nn::tensor::strided_copy::{MAX_RANK, StridedCopyRuntimeOp};

// ---------------------------------------------------------------------------
// Host references and the kernel's own mapping
// ---------------------------------------------------------------------------

fn strides_of(shape: &[usize]) -> Vec<usize> {
    let mut s = vec![1usize; shape.len()];
    for d in (0..shape.len().saturating_sub(1)).rev() {
        s[d] = s[d + 1] * shape[d + 1];
    }
    s
}

/// Flat output index -> coordinates, row-major.
fn coords_of(flat: usize, shape: &[usize]) -> Vec<usize> {
    let st = strides_of(shape);
    let mut rem = flat;
    (0..shape.len())
        .map(|d| {
            let c = rem / st[d];
            rem -= c * st[d];
            c
        })
        .collect()
}

/// ONNX `Transpose`: `out[c] = x[c permuted back]`.
fn transpose_reference(x: &[f32], in_shape: &[usize], perm: &[usize]) -> (Vec<f32>, Vec<usize>) {
    let out_shape: Vec<usize> = perm.iter().map(|&a| in_shape[a]).collect();
    let in_st = strides_of(in_shape);
    let n: usize = out_shape.iter().product();
    let out = (0..n)
        .map(|flat| {
            let c = coords_of(flat, &out_shape);
            // Output axis d is input axis perm[d].
            let src: usize = (0..perm.len()).map(|d| c[d] * in_st[perm[d]]).sum();
            x[src]
        })
        .collect();
    (out, out_shape)
}

/// ONNX `Slice`, from an already-normalised plan.
fn slice_reference(
    x: &[f32],
    in_shape: &[usize],
    starts: &[i64],
    steps: &[i64],
    out_shape: &[usize],
) -> Vec<f32> {
    let in_st = strides_of(in_shape);
    let n: usize = out_shape.iter().product();
    (0..n)
        .map(|flat| {
            let c = coords_of(flat, out_shape);
            let src: i64 = (0..in_shape.len())
                .map(|d| (starts[d] + c[d] as i64 * steps[d]) * in_st[d] as i64)
                .sum();
            x[src as usize]
        })
        .collect()
}

/// ONNX `Expand`: trailing axes align, a unit axis repeats.
fn expand_reference(x: &[f32], in_shape: &[usize], out_shape: &[usize]) -> Vec<f32> {
    let in_st = strides_of(in_shape);
    let lead = out_shape.len() - in_shape.len();
    let n: usize = out_shape.iter().product();
    (0..n)
        .map(|flat| {
            let c = coords_of(flat, out_shape);
            let src: usize = (0..in_shape.len())
                .map(|d| {
                    let coord = if in_shape[d] == 1 { 0 } else { c[lead + d] };
                    coord * in_st[d]
                })
                .sum();
            x[src]
        })
        .collect()
}

/// Replays the kernel's index arithmetic on the host from the scalars the runtime op
/// packs: `[n, in_offset, out_extents * MAX_RANK, in_strides * MAX_RANK]`. This is the
/// mapping the device will use, left-padding of lower-rank shapes included.
fn kernel_index_mapping(scalars: &[i32], n: usize) -> Vec<i64> {
    let in_offset = scalars[1] as i64;
    let out_extents = &scalars[2..2 + MAX_RANK];
    let in_strides = &scalars[2 + MAX_RANK..2 + 2 * MAX_RANK];

    (0..n)
        .map(|p| {
            let mut idx = in_offset;
            let mut rem = p as i64;
            for d in (0..MAX_RANK).rev() {
                let s = out_extents[d] as i64;
                let q = rem / s;
                let c = rem - q * s;
                idx += c * in_strides[d] as i64;
                rem = q;
            }
            idx
        })
        .collect()
}

/// Distinct values, so a wrong index cannot coincide with a right one.
fn ramp(n: usize) -> Vec<f32> {
    (0..n).map(|i| i as f32 + 0.5).collect()
}

fn check_mapping(
    label: &str,
    x: &[f32],
    want: &[f32],
    out_shape: &[usize],
    op: &StridedCopyRuntimeOp<f32>,
) {
    let n: usize = out_shape.iter().product();
    assert_eq!(n, want.len(), "{label}: reference length disagrees");
    let mapping = kernel_index_mapping(&op.kernel_scalars(n as i32), n);
    let got: Vec<f32> = mapping
        .iter()
        .map(|&i| {
            assert!(
                i >= 0 && (i as usize) < x.len(),
                "{label}: index {i} is outside the input of {} elements",
                x.len()
            );
            x[i as usize]
        })
        .collect();
    assert_eq!(got, want, "{label}: index mapping");
}

// ---------------------------------------------------------------------------
// Mapping
// ---------------------------------------------------------------------------

fn transpose_case(in_shape: &[usize], perm: &[usize]) {
    let x = ramp(in_shape.iter().product());
    let (want, out_shape) = transpose_reference(&x, in_shape, perm);
    let op = StridedCopyRuntimeOp::<f32>::transpose(1024, in_shape, perm);
    check_mapping(
        &format!("transpose {in_shape:?} by {perm:?}"),
        &x,
        &want,
        &out_shape,
        &op,
    );
}

#[test]
fn test_the_mapping_matches_onnx_transpose() {
    // The plain 2-D swap.
    transpose_case(&[2, 3], &[1, 0]);
    // The identity permutation is a contiguous copy.
    transpose_case(&[2, 3], &[0, 1]);
    // Every permutation of rank 3, so no ordering is special-cased by luck.
    for perm in [
        [0, 1, 2],
        [0, 2, 1],
        [1, 0, 2],
        [1, 2, 0],
        [2, 0, 1],
        [2, 1, 0],
    ] {
        transpose_case(&[2, 3, 4], &perm);
    }
    // NCHW to NHWC, the one that actually turns up.
    transpose_case(&[2, 3, 4, 5], &[0, 2, 3, 1]);
    // Full rank, so nothing is left-padded.
    transpose_case(&[2, 1, 3, 1, 2, 2], &[5, 4, 3, 2, 1, 0]);
}

fn slice_case(
    in_shape: &[usize],
    starts: &[i64],
    ends: &[i64],
    axes: Option<&[i64]>,
    steps: Option<&[i64]>,
) {
    let x = ramp(in_shape.iter().product());
    let plan = slice_plan(in_shape, starts, ends, axes, steps)
        .unwrap_or_else(|| panic!("{in_shape:?} [{starts:?}..{ends:?}] should be a valid slice"));
    let want = slice_reference(&x, in_shape, &plan.starts, &plan.steps, &plan.out_shape);
    let op = StridedCopyRuntimeOp::<f32>::slice(
        1024,
        in_shape,
        &plan.out_shape,
        &plan.starts,
        &plan.steps,
    );
    check_mapping(
        &format!("slice {in_shape:?} [{starts:?}..{ends:?}] step {steps:?}"),
        &x,
        &want,
        &plan.out_shape,
        &op,
    );
}

#[test]
fn test_the_mapping_matches_onnx_slice() {
    // A plain contiguous window.
    slice_case(&[6], &[1], &[4], None, None);
    // The whole axis, and a start beyond the end (empty is legal).
    slice_case(&[6], &[0], &[6], None, None);
    // Negative indices count from the end.
    slice_case(&[6], &[-4], &[-1], None, None);
    // Ends past the extent are clamped, not an error.
    slice_case(&[6], &[2], &[100], None, None);
    // A step skips elements.
    slice_case(&[8], &[0], &[8], None, Some(&[2]));
    slice_case(&[8], &[1], &[7], None, Some(&[3]));
    // Slicing one axis of several leaves the others whole.
    slice_case(&[4, 5], &[1], &[3], Some(&[0]), None);
    slice_case(&[4, 5], &[1], &[4], Some(&[1]), None);
    // A negative axis index.
    slice_case(&[4, 5], &[1], &[4], Some(&[-1]), None);
    // Both axes at once, with steps.
    slice_case(&[4, 6], &[1, 0], &[4, 6], Some(&[0, 1]), Some(&[2, 3]));
    // Rank 4, one axis sliced.
    slice_case(&[2, 3, 4, 5], &[1], &[3], Some(&[2]), None);
}

/// A negative step reverses an axis, which the affine mapping handles as a negative
/// stride with the offset at the far end.
#[test]
fn test_a_negative_step_reverses_an_axis() {
    slice_case(&[6], &[5], &[-7], None, Some(&[-1]));
    slice_case(&[6], &[4], &[0], None, Some(&[-2]));
    slice_case(&[3, 4], &[3], &[-5], Some(&[1]), Some(&[-1]));
}

fn expand_case(in_shape: &[usize], out_shape: &[usize]) {
    let x = ramp(in_shape.iter().product());
    let want = expand_reference(&x, in_shape, out_shape);
    let op = StridedCopyRuntimeOp::<f32>::expand(1024, in_shape, out_shape);
    check_mapping(
        &format!("expand {in_shape:?} to {out_shape:?}"),
        &x,
        &want,
        out_shape,
        &op,
    );
}

#[test]
fn test_the_mapping_matches_onnx_expand() {
    // A unit axis grows.
    expand_case(&[1], &[4]);
    expand_case(&[1, 3], &[2, 3]);
    expand_case(&[2, 1], &[2, 3]);
    expand_case(&[1, 1], &[2, 3]);
    // Rank grows by prepending axes, which broadcasting aligns at the trailing end.
    expand_case(&[3], &[2, 3]);
    expand_case(&[3], &[4, 2, 3]);
    expand_case(&[2, 1], &[5, 2, 3]);
    // Nothing to do: already the target shape.
    expand_case(&[2, 3], &[2, 3]);
    // Rank 4, the shape a bias broadcast takes.
    expand_case(&[1, 3, 1, 1], &[2, 3, 4, 5]);
}

// ---------------------------------------------------------------------------
// Lowering
// ---------------------------------------------------------------------------

fn i64_tensor(vals: &[i64]) -> ConstantTensor {
    let mut bytes = Vec::new();
    for v in vals {
        bytes.extend_from_slice(&v.to_le_bytes());
    }
    ConstantTensor {
        dtype: DtypeRepr::I64,
        shape: vec![Some(vals.len())],
        bytes,
    }
}

fn shape_of(dims: &[usize]) -> Vec<Option<usize>> {
    dims.iter().map(|&d| Some(d)).collect()
}

fn input_node(g: &mut Graph, dims: &[usize]) -> usize {
    g.add_node(Op::Input, vec![], DtypeRepr::F32, shape_of(dims))
}

fn i64_constant(g: &mut Graph, vals: &[i64]) -> usize {
    g.add_node(
        Op::Constant {
            dtype: DtypeRepr::I64,
            shape: vec![Some(vals.len())],
        },
        vec![],
        DtypeRepr::I64,
        vec![Some(vals.len())],
    )
}

fn lower(graph: &Graph, constants: ConstantMap) -> anyhow::Result<usize> {
    TritonLowering::new()
        .with_constants(constants)
        .lower(graph, LoweringMode::Inference)
        .map(|dag| dag.len())
}

fn lowering_error(graph: &Graph, constants: ConstantMap) -> String {
    match lower(graph, constants) {
        Ok(_) => panic!("expected lowering to fail"),
        Err(e) => e.to_string(),
    }
}

#[test]
fn test_transpose_lowers() {
    let mut g = Graph::new();
    let x = input_node(&mut g, &[2, 3, 4]);
    g.add_node(
        Op::Transpose {
            perm: vec![2, 0, 1],
        },
        vec![x],
        DtypeRepr::F32,
        shape_of(&[4, 2, 3]),
    );
    assert_eq!(lower(&g, ConstantMap::new()).expect("should lower"), 2);
}

/// ONNX defaults `Transpose`'s `perm` to the reversed axis order, and the importer
/// records an absent attribute as an empty vector. An empty perm must therefore mean
/// reverse, not identity: the two differ for every rank above 1, and reading it as
/// identity would silently produce a contiguous copy.
#[test]
fn test_an_empty_perm_means_the_reversed_axis_order() {
    let mut g = Graph::new();
    let x = input_node(&mut g, &[2, 3, 4]);
    g.add_node(
        Op::Transpose { perm: vec![] },
        vec![x],
        DtypeRepr::F32,
        shape_of(&[4, 3, 2]),
    );
    assert_eq!(lower(&g, ConstantMap::new()).expect("should lower"), 2);

    // And the mapping it chooses is the reversal, not the identity.
    let in_shape = [2usize, 3, 4];
    let reversed: Vec<usize> = (0..3).rev().collect();
    let x_vals = ramp(24);
    let (want, out_shape) = transpose_reference(&x_vals, &in_shape, &reversed);
    assert_eq!(out_shape, vec![4, 3, 2]);
    let op = StridedCopyRuntimeOp::<f32>::transpose(1024, &in_shape, &reversed);
    check_mapping("empty perm", &x_vals, &want, &out_shape, &op);
}

#[test]
fn test_a_perm_that_is_not_a_permutation_is_rejected() {
    for perm in [vec![0, 0], vec![0, 5]] {
        let mut g = Graph::new();
        let x = input_node(&mut g, &[2, 3]);
        g.add_node(
            Op::Transpose { perm: perm.clone() },
            vec![x],
            DtypeRepr::F32,
            shape_of(&[3, 2]),
        );
        let msg = lowering_error(&g, ConstantMap::new());
        assert!(
            msg.contains("is not a permutation"),
            "perm {perm:?} should be rejected: {msg}"
        );
    }
}

/// The runtime op left-pads its shapes to `MAX_RANK` and truncates anything longer,
/// which would drop the fastest-varying axes and copy the wrong elements with no
/// error at all. Lowering has to refuse it.
#[test]
fn test_a_rank_beyond_the_kernels_reach_is_refused() {
    let dims = vec![2usize; MAX_RANK + 1];
    let perm: Vec<usize> = (0..MAX_RANK + 1).rev().collect();
    let mut g = Graph::new();
    let x = input_node(&mut g, &dims);
    g.add_node(
        Op::Transpose { perm },
        vec![x],
        DtypeRepr::F32,
        shape_of(&dims),
    );
    let msg = lowering_error(&g, ConstantMap::new());
    assert!(msg.contains("exceeds the"), "{msg}");
}

/// A perm of the wrong length is reported as a shape problem, not a perm problem: an
/// exporter writes one entry per axis, so the recorded shape is what is usually wrong.
#[test]
fn test_a_perm_of_the_wrong_length_points_at_the_shape() {
    let mut g = Graph::new();
    let x = input_node(&mut g, &[128]);
    g.add_node(
        Op::Transpose {
            perm: vec![0, 2, 1],
        },
        vec![x],
        DtypeRepr::F32,
        shape_of(&[128]),
    );
    let msg = lowering_error(&g, ConstantMap::new());
    assert!(
        msg.contains("recorded shape is the likely culprit"),
        "should point at the shape: {msg}"
    );
}

/// An empty shape means either a rank-0 scalar or a shape nobody worked out, and the
/// IR cannot tell them apart, so it is refused rather than read as rank 0.
#[test]
fn test_an_empty_shape_is_refused_as_ambiguous() {
    let mut g = Graph::new();
    let x = g.add_node(Op::Input, vec![], DtypeRepr::F32, vec![]);
    g.add_node(
        Op::Transpose { perm: vec![1, 0] },
        vec![x],
        DtypeRepr::F32,
        vec![],
    );
    let msg = lowering_error(&g, ConstantMap::new());
    assert!(msg.contains("shape is empty"), "{msg}");
}

/// Integer tensors copy too. A model's shape arithmetic is I64, and broadcasting it is
/// what most of an export's `Expand` nodes do, so a float-only copy would leave them
/// unlowerable.
#[test]
fn test_an_integer_tensor_copies() {
    for dtype in [DtypeRepr::I64, DtypeRepr::I32, DtypeRepr::U8] {
        let mut g = Graph::new();
        let x = g.add_node(Op::Input, vec![], dtype, shape_of(&[1, 3]));
        let target = i64_constant(&mut g, &[2, 3]);
        g.add_node(Op::Expand, vec![x, target], dtype, shape_of(&[2, 3]));
        assert_eq!(
            lower(&g, ConstantMap::new()).unwrap_or_else(|e| panic!("{dtype:?} should lower: {e}")),
            2
        );
    }
}

#[test]
fn test_slice_lowers_from_materialised_operands() {
    let mut g = Graph::new();
    let x = input_node(&mut g, &[6, 4]);
    let starts = i64_constant(&mut g, &[1]);
    let ends = i64_constant(&mut g, &[4]);
    g.add_node(
        Op::Slice,
        vec![x, starts, ends],
        DtypeRepr::F32,
        shape_of(&[3, 4]),
    );
    let mut c = ConstantMap::new();
    c.insert(starts, i64_tensor(&[1]));
    c.insert(ends, i64_tensor(&[4]));
    // The three operands are metadata, so only the input and the slice get DAG nodes.
    assert_eq!(lower(&g, c).expect("should lower"), 2);
}

#[test]
fn test_slice_lowers_with_axes_and_steps() {
    let mut g = Graph::new();
    let x = input_node(&mut g, &[6, 8]);
    let starts = i64_constant(&mut g, &[0]);
    let ends = i64_constant(&mut g, &[8]);
    let axes = i64_constant(&mut g, &[1]);
    let steps = i64_constant(&mut g, &[2]);
    g.add_node(
        Op::Slice,
        vec![x, starts, ends, axes, steps],
        DtypeRepr::F32,
        shape_of(&[6, 4]),
    );
    let mut c = ConstantMap::new();
    c.insert(starts, i64_tensor(&[0]));
    c.insert(ends, i64_tensor(&[8]));
    c.insert(axes, i64_tensor(&[1]));
    c.insert(steps, i64_tensor(&[2]));
    assert_eq!(lower(&g, c).expect("should lower"), 2);
}

#[test]
fn test_slice_without_its_operands_says_so() {
    let mut g = Graph::new();
    let x = input_node(&mut g, &[6, 4]);
    let starts = i64_constant(&mut g, &[1]);
    let ends = i64_constant(&mut g, &[4]);
    g.add_node(
        Op::Slice,
        vec![x, starts, ends],
        DtypeRepr::F32,
        shape_of(&[3, 4]),
    );
    let msg = lowering_error(&g, ConstantMap::new());
    assert!(msg.contains("has no integer value"), "{msg}");
}

/// A zero step is not an empty slice, it is a meaningless one.
#[test]
fn test_a_zero_step_is_rejected() {
    let mut g = Graph::new();
    let x = input_node(&mut g, &[6]);
    let starts = i64_constant(&mut g, &[0]);
    let ends = i64_constant(&mut g, &[6]);
    let axes = i64_constant(&mut g, &[0]);
    let steps = i64_constant(&mut g, &[0]);
    g.add_node(
        Op::Slice,
        vec![x, starts, ends, axes, steps],
        DtypeRepr::F32,
        shape_of(&[0]),
    );
    let mut c = ConstantMap::new();
    c.insert(starts, i64_tensor(&[0]));
    c.insert(ends, i64_tensor(&[6]));
    c.insert(axes, i64_tensor(&[0]));
    c.insert(steps, i64_tensor(&[0]));
    let msg = lowering_error(&g, c);
    assert!(msg.contains("do not describe a slice"), "{msg}");
}

/// If the operands describe a different shape than the node records, the copy and the
/// output buffer would describe different tensors. The kernel would still run and
/// write a plausible tensor of wrong values, so lowering refuses.
///
/// This is not hypothetical: it is what a `Slice` graph built with a `steps` operand
/// but no `axes` does, since the operands are positional and `steps` is the fourth.
#[test]
fn test_a_mapping_that_disagrees_with_the_recorded_shape_is_refused() {
    let mut g = Graph::new();
    let x = input_node(&mut g, &[8]);
    let starts = i64_constant(&mut g, &[0]);
    let ends = i64_constant(&mut g, &[8]);
    // The node claims the shape a step of 2 would give, but no steps operand is
    // present, so the operands describe all 8 elements.
    g.add_node(
        Op::Slice,
        vec![x, starts, ends],
        DtypeRepr::F32,
        shape_of(&[4]),
    );
    let mut c = ConstantMap::new();
    c.insert(starts, i64_tensor(&[0]));
    c.insert(ends, i64_tensor(&[8]));
    let msg = lowering_error(&g, c);
    assert!(
        msg.contains("different") && msg.contains("[8]") && msg.contains("[4]"),
        "should name both shapes: {msg}"
    );
}

#[test]
fn test_expand_lowers_and_needs_no_operand_value() {
    let mut g = Graph::new();
    let x = input_node(&mut g, &[1, 3, 1, 1]);
    let target = i64_constant(&mut g, &[2, 3, 4, 5]);
    g.add_node(
        Op::Expand,
        vec![x, target],
        DtypeRepr::F32,
        shape_of(&[2, 3, 4, 5]),
    );
    // Expand reads its extents from its own inferred shape, which is what the output
    // buffer is allocated from, so no constant is needed for the shape operand.
    assert_eq!(lower(&g, ConstantMap::new()).expect("should lower"), 2);
}

#[test]
fn test_an_expand_that_is_not_a_broadcast_is_rejected() {
    let mut g = Graph::new();
    let x = input_node(&mut g, &[2, 3]);
    let target = i64_constant(&mut g, &[2, 4]);
    g.add_node(
        Op::Expand,
        vec![x, target],
        DtypeRepr::F32,
        shape_of(&[2, 4]),
    );
    let msg = lowering_error(&g, ConstantMap::new());
    assert!(msg.contains("nor is 1"), "{msg}");
}

/// A dynamic extent cannot be copied under a static mapping, and the error says what
/// to do about it rather than reporting an unsupported op.
#[test]
fn test_a_dynamic_extent_is_refused_with_advice() {
    let mut g = Graph::new();
    let x = g.add_node(Op::Input, vec![], DtypeRepr::F32, vec![None, Some(3)]);
    g.add_node(
        Op::Transpose { perm: vec![1, 0] },
        vec![x],
        DtypeRepr::F32,
        vec![Some(3), None],
    );
    let msg = lowering_error(&g, ConstantMap::new());
    assert!(msg.contains("materialize"), "{msg}");
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

    fn run(graph: &Graph, constants: ConstantMap, x: &[f32], want: &[f32], label: &str) {
        let (device, target) = device_and_target();
        let mut model = RuntimeModel::compile(
            &device,
            graph,
            &TritonLowering::new().with_constants(constants),
            LoweringMode::Inference,
            &target,
            1,
            false,
        )
        .expect("compile failed");

        let feeds: Vec<&[f32]> = vec![x];
        let outputs = model.terminal_node_indices_sorted_by_size();
        let got = model.run(&device, &feeds, &outputs).expect("run failed");
        let out = got
            .iter()
            .find(|o| o.len() == want.len())
            .unwrap_or_else(|| panic!("{label}: no output of length {}", want.len()));

        let bad: Vec<usize> = (0..want.len()).filter(|&i| out[i] != want[i]).collect();
        if !bad.is_empty() {
            println!("  got  = {out:?}");
            println!("  want = {want:?}");
            println!("  bad indices = {bad:?}");
        }
        assert!(
            bad.is_empty(),
            "{label}: {} of {} elements wrong",
            bad.len(),
            want.len()
        );
        println!("{label} -> {} elements, exact", want.len());
    }

    /// The mapping is also checked against a reference above with no device at all.
    #[test]
    fn test_transpose_runs_on_device() {
        for (dims, perm) in [
            (vec![2usize, 3], vec![1usize, 0]),
            (vec![2, 3, 4], vec![2, 0, 1]),
            (vec![2, 3, 4], vec![1, 2, 0]),
            (vec![2, 3, 4, 5], vec![0, 2, 3, 1]),
        ] {
            let x = ramp(dims.iter().product());
            let (want, out_shape) = transpose_reference(&x, &dims, &perm);
            let mut g = Graph::new();
            let input = input_node(&mut g, &dims);
            g.add_node(
                Op::Transpose { perm: perm.clone() },
                vec![input],
                DtypeRepr::F32,
                shape_of(&out_shape),
            );
            run(
                &g,
                ConstantMap::new(),
                &x,
                &want,
                &format!("transpose {dims:?} by {perm:?}"),
            );
        }
    }

    #[test]
    fn test_slice_runs_on_device() {
        for (dims, starts, ends, axes, steps) in [
            (vec![6usize], vec![1i64], vec![4i64], None, None),
            // `steps` is operand 4, so supplying it means supplying `axes` too.
            (
                vec![8],
                vec![0],
                vec![8],
                Some(vec![0i64]),
                Some(vec![2i64]),
            ),
            (vec![6], vec![5], vec![-7], Some(vec![0]), Some(vec![-1])),
            (vec![4, 5], vec![1], vec![3], Some(vec![0i64]), None),
            (
                vec![4, 6],
                vec![1, 0],
                vec![4, 6],
                Some(vec![0, 1]),
                Some(vec![2, 3]),
            ),
        ] {
            let x = ramp(dims.iter().product());
            let plan = slice_plan(&dims, &starts, &ends, axes.as_deref(), steps.as_deref())
                .expect("valid slice");
            let want = slice_reference(&x, &dims, &plan.starts, &plan.steps, &plan.out_shape);

            let mut g = Graph::new();
            let input = input_node(&mut g, &dims);
            let mut inputs = vec![input];
            let mut c = ConstantMap::new();
            for vals in [
                Some(starts.clone()),
                Some(ends.clone()),
                axes.clone(),
                steps.clone(),
            ] {
                // Operands are positional, so a missing one ends the list.
                let Some(vals) = vals else { break };
                let id = i64_constant(&mut g, &vals);
                c.insert(id, i64_tensor(&vals));
                inputs.push(id);
            }
            g.add_node(Op::Slice, inputs, DtypeRepr::F32, shape_of(&plan.out_shape));
            run(
                &g,
                c,
                &x,
                &want,
                &format!("slice {dims:?} {starts:?}..{ends:?}"),
            );
        }
    }

    #[test]
    fn test_expand_runs_on_device() {
        for (in_dims, out_dims) in [
            (vec![1usize], vec![4usize]),
            (vec![1, 3], vec![2, 3]),
            (vec![2, 1], vec![2, 3]),
            (vec![3], vec![2, 3]),
            (vec![1, 3, 1, 1], vec![2, 3, 4, 5]),
        ] {
            let x = ramp(in_dims.iter().product());
            let want = expand_reference(&x, &in_dims, &out_dims);
            let mut g = Graph::new();
            let input = input_node(&mut g, &in_dims);
            let target = i64_constant(
                &mut g,
                &out_dims.iter().map(|&d| d as i64).collect::<Vec<_>>(),
            );
            g.add_node(
                Op::Expand,
                vec![input, target],
                DtypeRepr::F32,
                shape_of(&out_dims),
            );
            run(
                &g,
                ConstantMap::new(),
                &x,
                &want,
                &format!("expand {in_dims:?} to {out_dims:?}"),
            );
        }
    }
}
