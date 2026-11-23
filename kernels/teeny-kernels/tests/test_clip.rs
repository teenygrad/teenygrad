// Clip: clamp a tensor into a range.
//
// The lowering used to hardcode both bounds to infinity, so every Clip in every model
// was a pass-through that cost a kernel launch and clamped nothing. The execution
// tests here are the ones that matter: a lowering test can only check that a bound
// reached the runtime op, while running the kernel is what shows values were actually
// clamped.
//
//   cargo test -p teeny-kernels --test test_clip
//   cargo test -p teeny-kernels --features cuda,hardware --test test_clip
//   cargo test -p teeny-kernels --no-default-features --features qemu,hardware \
//       --test test_clip

use teeny_core::graph::{ConstantMap, ConstantTensor, DtypeRepr, Graph, Op};
use teeny_core::model::{Lowering, LoweringMode};
use teeny_kernels::graph::TritonLowering;

fn f32_scalar(v: f32) -> ConstantTensor {
    ConstantTensor {
        dtype: DtypeRepr::F32,
        shape: vec![],
        bytes: v.to_le_bytes().to_vec(),
    }
}

/// An empty f32 tensor, which is how some exporters write "no bound on this side"
/// without omitting the operand.
fn f32_empty() -> ConstantTensor {
    ConstantTensor {
        dtype: DtypeRepr::F32,
        shape: vec![Some(0)],
        bytes: Vec::new(),
    }
}

const N: usize = 9;

/// Values spanning well below, inside and well above any bound used here, so a clamp
/// that does not happen cannot pass by coincidence.
///
/// Only the execution tests feed it, so it is unused without the `hardware` feature.
#[cfg_attr(not(feature = "hardware"), allow(dead_code))]
fn input() -> Vec<f32> {
    vec![-9.0, -6.0, -1.5, -0.25, 0.0, 0.25, 1.5, 6.0, 9.0]
}

/// How the bounds are supplied.
#[derive(Clone, Copy)]
#[cfg_attr(not(feature = "hardware"), allow(dead_code))]
enum Bounds {
    /// Opset 11+: operands 1 and 2, `None` meaning the operand is omitted.
    Operands(Option<f32>, Option<f32>),
    /// Opset 11+, but with an empty tensor rather than an omitted operand.
    EmptyOperands,
    /// Opset 6: `min` and `max` attributes.
    Attributes(Option<f64>, Option<f64>),
    /// No operands at all beyond the input.
    None,
}

/// Builds `Clip(x, min?, max?)` and the constant values for its bounds.
fn clip_graph(bounds: Bounds) -> (Graph, ConstantMap) {
    let mut g = Graph::new();
    let shape: Vec<Option<usize>> = vec![Some(N)];
    let x = g.add_node(Op::Input, vec![], DtypeRepr::F32, shape.clone());
    let mut constants = ConstantMap::new();
    let mut inputs = vec![x];

    let scalar_operand = |g: &mut Graph, constants: &mut ConstantMap, v: Option<f32>| match v {
        Some(v) => {
            let id = g.add_node(
                Op::Constant {
                    dtype: DtypeRepr::F32,
                    shape: vec![],
                },
                vec![],
                DtypeRepr::F32,
                vec![],
            );
            constants.insert(id, f32_scalar(v));
            id
        }
        // An omitted operand holds its position, as the importer records it.
        None => g.add_node(Op::Omitted, vec![], DtypeRepr::F32, Vec::new()),
    };

    let op = match bounds {
        Bounds::Operands(lo, hi) => {
            let a = scalar_operand(&mut g, &mut constants, lo);
            let b = scalar_operand(&mut g, &mut constants, hi);
            inputs.push(a);
            inputs.push(b);
            Op::Clip {
                min: None,
                max: None,
            }
        }
        Bounds::EmptyOperands => {
            for _ in 0..2 {
                let id = g.add_node(
                    Op::Constant {
                        dtype: DtypeRepr::F32,
                        shape: vec![Some(0)],
                    },
                    vec![],
                    DtypeRepr::F32,
                    vec![Some(0)],
                );
                constants.insert(id, f32_empty());
                inputs.push(id);
            }
            Op::Clip {
                min: None,
                max: None,
            }
        }
        Bounds::Attributes(lo, hi) => Op::Clip { min: lo, max: hi },
        Bounds::None => Op::Clip {
            min: None,
            max: None,
        },
    };

    g.add_node(op, inputs, DtypeRepr::F32, shape);
    (g, constants)
}

fn lower(bounds: Bounds) -> anyhow::Result<usize> {
    let (graph, constants) = clip_graph(bounds);
    TritonLowering::new()
        .with_constants(constants)
        .lower(&graph, LoweringMode::Inference)
        .map(|dag| dag.len())
}

/// Both bounds supplied as operands, which is what an opset-11-or-later export writes.
#[test]
fn test_clip_lowers_with_operand_bounds() {
    // The bounds are read on the host and packed as kernel arguments, so they get no
    // DAG node: the input and the clip are all that remain.
    assert_eq!(
        lower(Bounds::Operands(Some(-1.0), Some(1.0))).expect("should lower"),
        2
    );
}

#[test]
fn test_clip_lowers_with_an_omitted_bound() {
    assert_eq!(
        lower(Bounds::Operands(None, Some(6.0))).expect("max only"),
        2
    );
    assert_eq!(
        lower(Bounds::Operands(Some(0.0), None)).expect("min only"),
        2
    );
    assert_eq!(lower(Bounds::Operands(None, None)).expect("neither"), 2);
}

#[test]
fn test_clip_lowers_with_attribute_bounds() {
    assert_eq!(
        lower(Bounds::Attributes(Some(0.0), Some(6.0))).expect("opset 6 form"),
        2
    );
    assert_eq!(lower(Bounds::None).expect("no bounds at all"), 2);
}

/// A bound the lowering cannot see is an error, not an infinity. Silently substituting
/// no bound is exactly the behaviour this change removes.
#[test]
fn test_a_bound_with_no_value_is_refused() {
    let mut g = Graph::new();
    let x = g.add_node(Op::Input, vec![], DtypeRepr::F32, vec![Some(N)]);
    // A bound computed at runtime, which the kernel cannot take: it packs its bounds
    // as scalar arguments.
    let computed = g.add_node(Op::Input, vec![], DtypeRepr::F32, vec![]);
    g.add_node(
        Op::Clip {
            min: None,
            max: None,
        },
        vec![x, computed],
        DtypeRepr::F32,
        vec![Some(N)],
    );
    let msg = match TritonLowering::new().lower(&g, LoweringMode::Inference) {
        Ok(_) => panic!("a bound with no value should not lower"),
        Err(e) => e.to_string(),
    };
    assert!(msg.contains("has no value"), "{msg}");
}

#[test]
fn test_an_inverted_range_is_refused() {
    let msg = match lower(Bounds::Operands(Some(1.0), Some(-1.0))) {
        Ok(_) => panic!("min above max should not lower"),
        Err(e) => e.to_string(),
    };
    assert!(msg.contains("is above max"), "{msg}");
}

/// The kernel takes f32 bounds whatever the tensor's dtype. A finite bound that
/// overflows f32 would become infinity, which is not a near-miss but the absence of a
/// bound, so it is refused.
#[test]
fn test_a_bound_outside_f32_range_is_refused() {
    let mut g = Graph::new();
    let x = g.add_node(Op::Input, vec![], DtypeRepr::F64, vec![Some(N)]);
    let hi = g.add_node(
        Op::Constant {
            dtype: DtypeRepr::F64,
            shape: vec![],
        },
        vec![],
        DtypeRepr::F64,
        vec![],
    );
    let omitted = g.add_node(Op::Omitted, vec![], DtypeRepr::F64, Vec::new());
    g.add_node(
        Op::Clip {
            min: None,
            max: None,
        },
        vec![x, omitted, hi],
        DtypeRepr::F64,
        vec![Some(N)],
    );
    let mut constants = ConstantMap::new();
    constants.insert(
        hi,
        ConstantTensor {
            dtype: DtypeRepr::F64,
            shape: vec![],
            bytes: 1e300f64.to_le_bytes().to_vec(),
        },
    );
    let msg = match TritonLowering::new()
        .with_constants(constants)
        .lower(&g, LoweringMode::Inference)
    {
        Ok(_) => panic!("an out-of-f32-range bound should not lower"),
        Err(e) => e.to_string(),
    };
    assert!(msg.contains("outside f32 range"), "{msg}");
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

    fn run(bounds: Bounds, lo: f32, hi: f32, label: &str) {
        let x = input();
        let want: Vec<f32> = x.iter().map(|v| v.clamp(lo, hi)).collect();

        let (graph, constants) = clip_graph(bounds);
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
            .find(|o| o.len() == N)
            .unwrap_or_else(|| panic!("{label}: no output of length {N}"));

        let bad: Vec<usize> = (0..N).filter(|&i| out[i] != want[i]).collect();
        if !bad.is_empty() {
            println!("  in   = {x:?}");
            println!("  got  = {out:?}");
            println!("  want = {want:?}");
        }
        assert!(
            bad.is_empty(),
            "{label}: {} of {N} wrong at {bad:?}",
            bad.len()
        );
        println!("{label}: clamped to [{lo}, {hi}], exact");
    }

    /// The test this change exists for. The input spans far outside both bounds, so a
    /// Clip that does not clamp returns its input and fails here.
    #[test]
    fn test_clip_actually_clamps_on_device() {
        run(
            Bounds::Operands(Some(-1.0), Some(1.0)),
            -1.0,
            1.0,
            "both bounds",
        );
    }

    /// A one-sided clamp has to leave the other side alone. Getting this wrong by
    /// reading the upper bound as the lower one, which is what dropping an omitted
    /// operand did, clamps everything to a single value.
    #[test]
    fn test_a_one_sided_clamp_leaves_the_other_side_alone() {
        run(
            Bounds::Operands(Some(0.0), None),
            0.0,
            f32::INFINITY,
            "min only (a relu)",
        );
        run(
            Bounds::Operands(None, Some(1.0)),
            f32::NEG_INFINITY,
            1.0,
            "max only",
        );
    }

    /// The shape every `Clip` in the FoundationStereo export takes: a tiny positive
    /// lower bound with the upper one omitted, guarding a later division or square
    /// root against zero. Unclamped, the zero in the input reaches that guard.
    #[test]
    fn test_a_tiny_lower_bound_guard_clamps() {
        const FLOOR: f32 = 1e-12;
        let x = input();
        assert!(
            x.contains(&0.0) && x.iter().any(|&v| v < 0.0),
            "the input must include values the floor has to raise"
        );
        run(
            Bounds::Operands(Some(FLOOR), None),
            FLOOR,
            f32::INFINITY,
            "1e-12 floor, no ceiling",
        );
    }

    #[test]
    fn test_attribute_bounds_clamp_too() {
        run(
            Bounds::Attributes(Some(-6.0), Some(6.0)),
            -6.0,
            6.0,
            "opset 6 attributes",
        );
    }

    /// With no bounds at all a Clip is genuinely a pass-through, which is the one case
    /// the old behaviour got right.
    #[test]
    fn test_no_bounds_passes_values_through() {
        run(Bounds::None, f32::NEG_INFINITY, f32::INFINITY, "no bounds");
        run(
            Bounds::EmptyOperands,
            f32::NEG_INFINITY,
            f32::INFINITY,
            "empty bound tensors",
        );
    }
}
