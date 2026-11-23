// Resize: nearest, linear and cubic over the trailing axes.
//
// The kernel interpolates nothing on its own: it multiplies and adds the taps the host
// planned. So these tests check the two halves separately. resize_axis_plan has its
// own tests in teeny-core for the coordinate arithmetic, and what is checked here is
// that the kernel reads the taps it was given, in the right places, for each mode and
// for two and three resized axes.
//
//   cargo test -p teeny-kernels --test test_resize
//   cargo test -p teeny-kernels --no-default-features --features qemu,hardware \
//       --test test_resize

use teeny_core::graph::{DtypeRepr, Graph, Op};
use teeny_core::model::{Lowering, LoweringMode};
use teeny_kernels::graph::TritonLowering;

fn resize_graph(
    in_shape: &[usize],
    out_shape: &[usize],
    mode: &str,
    coords: &str,
    nearest: &str,
) -> Graph {
    let mut g = Graph::new();
    let x = g.add_node(
        Op::Input,
        vec![],
        DtypeRepr::F32,
        in_shape.iter().map(|&d| Some(d)).collect(),
    );
    g.add_node(
        Op::Resize {
            mode: mode.to_string(),
            coordinate_transformation_mode: coords.to_string(),
            nearest_mode: nearest.to_string(),
            antialias: false,
        },
        vec![x],
        DtypeRepr::F32,
        out_shape.iter().map(|&d| Some(d)).collect(),
    );
    g
}

fn lower(
    in_shape: &[usize],
    out_shape: &[usize],
    mode: &str,
    coords: &str,
) -> anyhow::Result<usize> {
    TritonLowering::new()
        .lower(
            &resize_graph(in_shape, out_shape, mode, coords, "floor"),
            LoweringMode::Inference,
        )
        .map(|dag| dag.len())
}

#[test]
fn test_the_export_s_shapes_lower() {
    // Two spatial axes, linear with aligned corners: the export's commonest resize.
    assert_eq!(
        lower(
            &[1, 128, 30, 40],
            &[1, 128, 60, 80],
            "linear",
            "align_corners"
        )
        .expect("2-D linear"),
        2
    );
    // Three spatial axes.
    assert_eq!(
        lower(
            &[1, 28, 26, 30, 40],
            &[1, 28, 104, 120, 160],
            "linear",
            "half_pixel"
        )
        .expect("3-D linear"),
        2
    );
    // Cubic, and nearest.
    assert_eq!(
        lower(&[2, 3, 480, 640], &[2, 3, 560, 672], "cubic", "half_pixel").expect("cubic"),
        2
    );
    assert_eq!(
        lower(
            &[1, 9, 120, 160],
            &[1, 9, 480, 640],
            "nearest",
            "asymmetric"
        )
        .expect("nearest"),
        2
    );
}

/// More resized axes than the kernel indexes, named as such rather than silently
/// resizing the trailing three.
#[test]
fn test_too_many_resized_axes_are_refused() {
    let msg = match lower(&[2, 3, 4, 5], &[4, 6, 8, 10], "linear", "half_pixel") {
        Ok(_) => panic!("four resized axes should not lower"),
        Err(e) => e.to_string(),
    };
    assert!(msg.contains("changes 4 axes"), "{msg}");
}

#[test]
fn test_an_unimplemented_mode_is_refused_by_name() {
    let msg = match lower(
        &[1, 1, 4, 4],
        &[1, 1, 8, 8],
        "cubic_antialias",
        "half_pixel",
    ) {
        Ok(_) => panic!("an unknown mode should not lower"),
        Err(e) => e.to_string(),
    };
    assert!(msg.contains("\"cubic_antialias\""), "{msg}");
}

// ---------------------------------------------------------------------------
// Execution
// ---------------------------------------------------------------------------

#[cfg(all(feature = "hardware", any(feature = "cuda", feature = "riscv")))]
mod execution {
    use super::*;
    use teeny_core::graph::{ResizeCoords, ResizeMode, ResizeNearest, resize_axis_plan};
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

    /// The resize on the host, from the same per-axis plans.
    ///
    /// This shares the plans with the kernel rather than recomputing the
    /// interpolation, which is the point: the plans are tested on their own in
    /// teeny-core, and what is in question here is whether the kernel applies them
    /// where it should. A wrong stride, a coordinate peeled from the wrong axis or a
    /// tap read from another axis' table all show up as a mismatch.
    fn reference(
        x: &[f32],
        in_shape: &[usize],
        out_shape: &[usize],
        mode: ResizeMode,
        coords: ResizeCoords,
        first: usize,
    ) -> Vec<f32> {
        let rank = in_shape.len();
        let plans: Vec<_> = (first..rank)
            .map(|ax| {
                resize_axis_plan(mode, coords, in_shape[ax], out_shape[ax], None)
                    .expect("plannable")
            })
            .collect();
        let mut in_strides = vec![1usize; rank];
        for ax in (0..rank.saturating_sub(1)).rev() {
            in_strides[ax] = in_strides[ax + 1] * in_shape[ax + 1];
        }

        let n_out: usize = out_shape.iter().product();
        let mut want = vec![0.0f32; n_out];
        for flat in 0..n_out {
            // The output's coordinates, and the prefix the resize does not touch.
            let mut rest = flat;
            let mut coord = vec![0usize; rank];
            for ax in (0..rank).rev() {
                coord[ax] = rest % out_shape[ax];
                rest /= out_shape[ax];
            }
            let prefix: usize = (0..first).map(|ax| coord[ax] * in_strides[ax]).sum();

            // Every combination of taps across the resized axes.
            let taps: Vec<usize> = plans.iter().map(|p| p.taps).collect();
            let total: usize = taps.iter().product();
            let mut acc = 0.0f64;
            for t in 0..total {
                let mut rest = t;
                let mut weight = 1.0f64;
                let mut offset = prefix;
                for (k, plan) in plans.iter().enumerate().rev() {
                    let tap = rest % plan.taps;
                    rest /= plan.taps;
                    let ax = first + k;
                    let slot = coord[ax] * plan.taps + tap;
                    weight *= plan.weights[slot];
                    offset += plan.indices[slot] as usize * in_strides[ax];
                }
                acc += weight * x[offset] as f64;
            }
            want[flat] = acc as f32;
        }
        want
    }

    fn run(in_shape: &[usize], out_shape: &[usize], mode: &str, coords: &str, nearest: &str) {
        let interpolation = match (mode, nearest) {
            ("nearest", "floor") => ResizeMode::Nearest(ResizeNearest::Floor),
            ("nearest", _) => ResizeMode::Nearest(ResizeNearest::RoundPreferFloor),
            ("linear", _) => ResizeMode::Linear,
            _ => ResizeMode::Cubic,
        };
        let transform = match coords {
            "align_corners" => ResizeCoords::AlignCorners,
            "asymmetric" => ResizeCoords::Asymmetric,
            _ => ResizeCoords::HalfPixel,
        };
        let rank = in_shape.len();
        let first = (0..rank)
            .find(|&ax| in_shape[ax] != out_shape[ax])
            .unwrap_or(rank)
            .min(rank - 1);

        let n_in: usize = in_shape.iter().product();
        // Values that vary along every axis, so a coordinate peeled from the wrong
        // one changes the answer.
        let x: Vec<f32> = (0..n_in).map(|i| (i % 17) as f32 * 0.25 - 1.0).collect();
        let want = reference(&x, in_shape, out_shape, interpolation, transform, first);

        let graph = resize_graph(in_shape, out_shape, mode, coords, nearest);
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

        let feeds: Vec<&[f32]> = vec![&x];
        let outputs = model.terminal_node_indices_sorted_by_size();
        let got = model.run(&device, &feeds, &outputs).expect("run failed");
        let out = got
            .iter()
            .find(|o| o.len() == want.len())
            .unwrap_or_else(|| panic!("no output of length {}", want.len()));
        let mut worst = 0.0f32;
        let mut at = 0usize;
        for i in 0..want.len() {
            let d = (out[i] - want[i]).abs() / want[i].abs().max(1.0);
            if d > worst {
                worst = d;
                at = i;
            }
        }
        assert!(
            worst < 1e-5,
            "{in_shape:?} -> {out_shape:?} {mode}/{coords}: worst {worst} at {at} \
             (got {}, want {})",
            out[at],
            want[at]
        );
    }

    #[test]
    fn test_linear_over_two_axes() {
        run(
            &[1, 2, 3, 4],
            &[1, 2, 6, 8],
            "linear",
            "align_corners",
            "floor",
        );
    }

    /// Downsampling as well as up, since the plan's taps move in the other direction.
    #[test]
    fn test_linear_downsampling() {
        run(
            &[1, 1, 8, 6],
            &[1, 1, 3, 4],
            "linear",
            "half_pixel",
            "floor",
        );
    }

    /// Three resized axes, which is where a stride taken from the wrong axis shows up.
    #[test]
    fn test_linear_over_three_axes() {
        run(
            &[1, 2, 2, 3, 2],
            &[1, 2, 4, 6, 4],
            "linear",
            "half_pixel",
            "floor",
        );
    }

    /// Nearest with floor, which is what the export asks for: a whole-factor repeat.
    #[test]
    fn test_nearest_repeats_each_element() {
        run(
            &[1, 3, 2, 3],
            &[1, 3, 8, 12],
            "nearest",
            "asymmetric",
            "floor",
        );
    }

    /// Cubic's four taps per axis, sixteen per output element, including the negative
    /// weights that make it a sharpening filter rather than an average.
    #[test]
    fn test_cubic_over_two_axes() {
        run(&[1, 1, 5, 6], &[1, 1, 7, 9], "cubic", "half_pixel", "floor");
    }

    /// An uneven ratio, where every output coordinate has a different fractional
    /// offset and a mistake in one does not repeat into the others.
    #[test]
    fn test_an_uneven_ratio() {
        run(
            &[1, 1, 37, 37],
            &[1, 1, 40, 48],
            "linear",
            "half_pixel",
            "floor",
        );
    }
}
