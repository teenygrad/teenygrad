//! Which `[rows, cols]` view a softmax-family op lowers to, for each way of naming its
//! axis.
//!
//! The kernels normalize the rows of a flattened `[rows, cols]` view and take both
//! extents from the shape their runtime op is handed. ONNX before opset 13 meant that
//! view exactly, starting at any axis ([`SoftmaxAxis::FlattenFrom`]); from opset 13, as in
//! PyTorch, only the one axis is normalized ([`SoftmaxAxis::Single`]), which the view can
//! express only for the innermost axis; any other is transposed innermost and back around
//! the kernel by `strided_copy` (teenygrad#152). These tests pin the extents the lowering passes to the
//! kernel and the transposes around it, and on hardware check the whole lowered DAG
//! against PyTorch (`tests/fixtures/softmax_axis/generate.py`). The kernel's own
//! arithmetic over a given view is pinned by `test_softmax_rows`.

use teeny_core::device::program::ArgVisitor;
use teeny_core::graph::{DtypeRepr, Graph, Op, SoftmaxAxis};
use teeny_core::model::{ExecutableOp, Lowering, LoweringMode, RuntimeOp};
use teeny_kernels::graph::TritonLowering;

const SHAPE: [usize; 4] = [1, 3, 2, 4];

/// The integer kernel arguments, in order. Pointers are skipped.
#[derive(Default)]
struct Ints(Vec<i64>);

impl ArgVisitor for Ints {
    fn visit_ptr(&mut self, _: *mut core::ffi::c_void) {}
    fn visit_bool(&mut self, v: bool) {
        self.0.push(v as i64);
    }
    fn visit_i8(&mut self, v: i8) {
        self.0.push(v as i64);
    }
    fn visit_i16(&mut self, v: i16) {
        self.0.push(v as i64);
    }
    fn visit_i32(&mut self, v: i32) {
        self.0.push(v as i64);
    }
    fn visit_i64(&mut self, v: i64) {
        self.0.push(v);
    }
    fn visit_u8(&mut self, v: u8) {
        self.0.push(v as i64);
    }
    fn visit_u16(&mut self, v: u16) {
        self.0.push(v as i64);
    }
    fn visit_u32(&mut self, v: u32) {
        self.0.push(v as i64);
    }
    fn visit_u64(&mut self, v: u64) {
        self.0.push(v as i64);
    }
    fn visit_f32(&mut self, _: f32) {}
    fn visit_f64(&mut self, _: f64) {}
}

/// A one-node graph applying `op` to an f32 tensor of [`SHAPE`].
fn graph_of(op: Op) -> Graph {
    let shape: Vec<Option<usize>> = SHAPE.iter().map(|d| Some(*d)).collect();
    let mut g = Graph::new();
    let x = g.add_node(Op::Input, vec![], DtypeRepr::F32, shape.clone());
    g.add_node(op, vec![x], DtypeRepr::F32, shape);
    g
}

/// The lowered softmax-family node: its kernel source and its runtime op.
fn lower(op: Op) -> anyhow::Result<(String, std::sync::Arc<dyn RuntimeOp>)> {
    let dag = TritonLowering::new().lower(&graph_of(op), LoweringMode::Inference)?;
    let exec: &dyn ExecutableOp = (0..dag.len())
        .map(|i| dag.node(i).value.as_ref())
        .find(|e| e.forward_kernel_entry_point().contains("softmax"))
        .expect("the lowered graph has a softmax-family kernel");
    let rop = exec.runtime_op().expect("a kernel node has a runtime op");
    Ok((exec.forward_kernel_source().to_string(), rop))
}

/// `(grid x, n_rows, n_cols)` the runtime op launches the kernel with.
fn launch(rop: &dyn RuntimeOp) -> (u32, i64, i64) {
    let mut ints = Ints::default();
    let x = core::ptr::null_mut();
    rop.pack_args(&[(x, &SHAPE[..])], &[], x, &SHAPE, 0, &mut ints);
    let [n_rows, n_cols] = ints.0[..] else {
        panic!(
            "expected the kernel's two integer arguments, got {:?}",
            ints.0
        );
    };
    let grid = rop.grid(&SHAPE);
    assert_eq!(&grid[1..], &[1, 1], "one CTA per row, along x");
    (grid[0], n_rows, n_cols)
}

fn softmax(dim: i64, mode: SoftmaxAxis) -> Op {
    Op::Softmax { dim, mode }
}

fn log_softmax(axis: i64, mode: SoftmaxAxis) -> Op {
    Op::LogSoftmax { axis, mode }
}

#[test]
fn the_innermost_axis_is_one_row_per_outer_element_either_way() {
    for mode in [SoftmaxAxis::Single, SoftmaxAxis::FlattenFrom] {
        for op in [softmax(-1, mode), softmax(3, mode), log_softmax(-1, mode)] {
            let (_, rop) = lower(op).unwrap();
            assert_eq!(launch(rop.as_ref()), (6, 6, 4), "{mode:?}");
        }
    }
}

/// ONNX before opset 13: everything from the axis on is one row, whatever the axis.
#[test]
fn flattening_from_an_outer_axis_makes_the_trailing_block_one_row() {
    for (axis, want) in [
        (1, (1, 1, 24)),
        (2, (3, 3, 8)),
        (0, (1, 1, 24)),
        (-3, (1, 1, 24)),
    ] {
        for op in [
            softmax(axis, SoftmaxAxis::FlattenFrom),
            log_softmax(axis, SoftmaxAxis::FlattenFrom),
        ] {
            let (_, rop) = lower(op).unwrap();
            assert_eq!(launch(rop.as_ref()), want, "axis {axis}");
        }
    }
}

/// `BLOCK_SIZE` covers the whole flattened row, not just the last extent.
#[test]
fn the_block_covers_the_flattened_row() {
    let (src, _) = lower(softmax(1, SoftmaxAxis::FlattenFrom)).unwrap();
    assert!(
        src.contains("f32, 32>("),
        "24 columns need a 32-wide block:\n{src}"
    );
    let (src, _) = lower(softmax(-1, SoftmaxAxis::Single)).unwrap();
    assert!(
        src.contains("f32, 4>("),
        "4 columns need a 4-wide block:\n{src}"
    );
    let (src, _) = lower(log_softmax(2, SoftmaxAxis::FlattenFrom)).unwrap();
    assert!(
        src.contains("LlvmTriton, 8>("),
        "8 columns need an 8-wide block:\n{src}"
    );
}

/// The kernel names of `op`'s lowered DAG over [`SHAPE`], in order, without the input.
fn kernels_of(op: Op) -> Vec<String> {
    let dag = TritonLowering::new()
        .lower(&graph_of(op), LoweringMode::Inference)
        .expect("lowers");
    (1..dag.len())
        .map(|i| dag.node(i).value.name().to_string())
        .collect()
}

/// From opset 13 an outer axis is normalized alone: the lowering moves it innermost,
/// normalizes, and moves it back.
#[test]
fn a_single_outer_axis_is_transposed_innermost_and_back() {
    for op in [
        softmax(1, SoftmaxAxis::Single),
        softmax(0, SoftmaxAxis::Single),
        log_softmax(-2, SoftmaxAxis::Single),
    ] {
        let kernels = kernels_of(op.clone());
        assert_eq!(kernels.len(), 3, "{op:?}: {kernels:?}");
        assert!(kernels[0].contains("strided_copy"), "{op:?}: {kernels:?}");
        assert!(kernels[1].contains("softmax"), "{op:?}: {kernels:?}");
        assert!(kernels[2].contains("strided_copy"), "{op:?}: {kernels:?}");
    }
}

#[test]
fn the_innermost_axis_and_flattening_need_no_transpose() {
    for op in [
        softmax(-1, SoftmaxAxis::Single),
        softmax(1, SoftmaxAxis::FlattenFrom),
        log_softmax(3, SoftmaxAxis::Single),
    ] {
        assert_eq!(kernels_of(op.clone()).len(), 1, "{op:?}");
    }
}

/// The graph node maps to the transpose back, whose output has the op's own shape, and
/// the kernel normalizes the transposed copy's innermost axis.
#[test]
fn the_kernel_sees_the_axis_innermost_and_the_node_keeps_its_shape() {
    let g = graph_of(softmax(1, SoftmaxAxis::Single));
    let (dag, map) = TritonLowering::new()
        .lower_with_mapping(&g, LoweringMode::Inference)
        .expect("lowers");
    let back = map[1];
    let shape = |dims: &[usize]| dims.iter().map(|d| Some(*d)).collect::<Vec<_>>();
    assert!(dag.node(back).value.name().contains("strided_copy"));
    assert_eq!(dag.node(back).value.output_shape(), &shape(&SHAPE));
    let kernel = dag.node(back).parents[0];
    assert!(dag.node(kernel).value.name().contains("softmax"));
    // [1, 3, 2, 4] with axis 1 moved last.
    assert_eq!(dag.node(kernel).value.output_shape(), &shape(&[1, 2, 4, 3]));
    let rop = dag.node(kernel).value.runtime_op().unwrap();
    let mut ints = Ints::default();
    let x = core::ptr::null_mut();
    let moved = [1, 2, 4, 3];
    rop.pack_args(&[(x, &moved[..])], &[], x, &moved, 0, &mut ints);
    assert_eq!(ints.0, vec![8, 3], "8 rows of the 3 elements along axis 1");
    assert_eq!(rop.grid(&moved)[0], 8);
}

/// Only extent-1 axes after the named one: its elements are already a row.
#[test]
fn trailing_axes_of_extent_one_need_no_transpose() {
    let mut g = Graph::new();
    let shape = vec![Some(2), Some(3), Some(1)];
    let x = g.add_node(Op::Input, vec![], DtypeRepr::F32, shape.clone());
    g.add_node(
        softmax(1, SoftmaxAxis::Single),
        vec![x],
        DtypeRepr::F32,
        shape,
    );
    let dag = TritonLowering::new()
        .lower(&g, LoweringMode::Inference)
        .expect("lowers");
    assert_eq!(dag.len(), 2);
}

/// The transposes have no backward kernel, so training would skip the gradient.
#[cfg(feature = "training")]
#[test]
fn training_through_a_transposed_axis_is_refused() {
    let err = TritonLowering::new()
        .lower(
            &graph_of(softmax(1, SoftmaxAxis::Single)),
            LoweringMode::Training,
        )
        .map(|_| ())
        .expect_err("the transposes have no backward");
    assert!(err.to_string().contains("no backward"), "{err}");
    TritonLowering::new()
        .lower(
            &graph_of(softmax(-1, SoftmaxAxis::Single)),
            LoweringMode::Training,
        )
        .map(|_| ())
        .expect("the innermost axis trains as before");
}

#[test]
fn an_axis_outside_the_rank_is_refused() {
    for mode in [SoftmaxAxis::Single, SoftmaxAxis::FlattenFrom] {
        for axis in [4, -5] {
            let err = lower(softmax(axis, mode))
                .map(|_| ())
                .expect_err("rank 4 has no such axis");
            assert!(
                err.to_string().contains("outside"),
                "{mode:?} axis {axis}: {err}"
            );
        }
    }
}

// ── Hardware: the lowered DAG against PyTorch ───────────────────────────────

#[cfg(feature = "hardware")]
mod hardware {
    use super::*;
    use teeny_runtime::Target;
    use teeny_test::load_fixture;
    use teeny_test::model::RuntimeModel;

    /// (fixture name, shape, axis, flatten from the axis), as in `generate.py`.
    const CASES: &[(&str, &[usize], i64, bool)] = &[
        ("s6x4_last", &[6, 4], -1, false),
        ("s4x8_last", &[4, 8], -1, false),
        ("s4x6_last", &[4, 6], -1, false),
        ("s1x3x2x4_last", &[1, 3, 2, 4], -1, false),
        ("s6x4_axis0", &[6, 4], 0, false),
        ("s1x3x2x4_axis1", &[1, 3, 2, 4], 1, false),
        ("s2x3x4x5_axis0", &[2, 3, 4, 5], 0, false),
        ("s2x3x4x5_axis1", &[2, 3, 4, 5], 1, false),
        ("s2x3x4x5_axis2", &[2, 3, 4, 5], -2, false),
        ("s1x2x5x3x4_axis2", &[1, 2, 5, 3, 4], 2, false),
        ("s2x3x1_axis1", &[2, 3, 1], 1, false),
        ("s1x9x24x32_axis1", &[1, 9, 24, 32], 1, false),
        ("s1x3x2x4_flatten1", &[1, 3, 2, 4], 1, true),
        ("s2x3x4_flatten1", &[2, 3, 4], 1, true),
    ];
    const TOL: f32 = 1e-5;

    fn device_and_target<'a>() -> (teeny_runtime::Device<'a>, Target) {
        #[cfg(feature = "cuda")]
        {
            let device = teeny_runtime::open().expect("a CUDA device");
            let target = teeny_runtime::default_target(&device).expect("target");
            (device, target)
        }
        #[cfg(not(feature = "cuda"))]
        {
            use teeny_riscv::compiler::target::Capability;
            (
                teeny_runtime::Device::new(Default::default()),
                Target::new(Capability::GenericRvv1_0),
            )
        }
    }

    /// Compiles `op` over `shape` and runs the whole lowered graph on `x`.
    fn run(
        device: &teeny_runtime::Device<'_>,
        target: &Target,
        op: Op,
        shape: &[usize],
        x: &[f32],
    ) -> Vec<f32> {
        let dims: Vec<_> = shape.iter().map(|d| Some(*d)).collect();
        let mut g = Graph::new();
        let input = g.add_node(Op::Input, vec![], DtypeRepr::F32, dims.clone());
        g.add_node(op, vec![input], DtypeRepr::F32, dims);
        let mut model = RuntimeModel::compile(
            device,
            &g,
            &TritonLowering::new(),
            LoweringMode::Inference,
            target,
            1,
            false,
        )
        .expect("compile");
        let outputs = model.terminal_node_indices_sorted_by_size();
        model
            .run(device, &[x], &outputs)
            .expect("run")
            .pop()
            .expect("an output")
    }

    fn check(log: bool) {
        dotenv::dotenv().ok();
        let (device, target) = device_and_target();
        let dir = env!("CARGO_MANIFEST_DIR");
        let prefix = if log { "log_softmax" } else { "softmax" };
        let mut failures = Vec::new();
        for &(name, shape, axis, flatten) in CASES {
            let x = load_fixture(dir, &format!("softmax_axis/x_{name}.bin"));
            let want = load_fixture(dir, &format!("softmax_axis/{prefix}_{name}.bin"));
            let mode = if flatten {
                SoftmaxAxis::FlattenFrom
            } else {
                SoftmaxAxis::Single
            };
            let op = if log {
                log_softmax(axis, mode)
            } else {
                softmax(axis, mode)
            };
            let got = run(&device, &target, op, shape, &x);
            if got.len() != want.len() {
                failures.push(format!(
                    "{prefix} {name}: {} values, want {}",
                    got.len(),
                    want.len()
                ));
                continue;
            }
            if let Some(i) = (0..want.len()).find(|&i| !((got[i] - want[i]).abs() <= TOL)) {
                failures.push(format!(
                    "{prefix} {name}: [{i}] got {}, want {}",
                    got[i], want[i]
                ));
            }
            // An unwritten row stays zero, which a softmax output never is.
            if !log && got.contains(&0.0) {
                failures.push(format!("{prefix} {name}: has exact zeros"));
            }
        }
        assert!(failures.is_empty(), "{}", failures.join("\n"));
    }

    #[test]
    fn softmax_over_any_axis_matches_torch() {
        check(false);
    }

    #[test]
    fn log_softmax_over_any_axis_matches_torch() {
        check(true);
    }
}
