// GridSample, bilinear, with zero padding.
//
// Unlike a resize, the positions are not known when the graph is lowered: the grid is
// computed, so the coordinate arithmetic happens per lane. What the host precomputes
// is the scale and offset that turn a normalised coordinate into an index, which is
// where align_corners lives.
//
//   cargo test -p teeny-kernels --test test_grid_sample
//   cargo test -p teeny-kernels --no-default-features --features qemu,hardware \
//       --test test_grid_sample

use teeny_core::graph::{DtypeRepr, Graph, Op};
use teeny_core::model::{Lowering, LoweringMode};
use teeny_kernels::graph::TritonLowering;

fn sample_graph(
    x: &[usize],
    grid: &[usize],
    mode: &str,
    padding: &str,
    align_corners: bool,
) -> Graph {
    let dims = |s: &[usize]| -> Vec<Option<usize>> { s.iter().map(|&d| Some(d)).collect() };
    let out = vec![x[0], x[1], grid[1], grid[2]];
    let mut g = Graph::new();
    let xn = g.add_node(Op::Input, vec![], DtypeRepr::F32, dims(x));
    let gn = g.add_node(Op::Input, vec![], DtypeRepr::F32, dims(grid));
    g.add_node(
        Op::GridSample {
            mode: mode.to_string(),
            padding_mode: padding.to_string(),
            align_corners,
        },
        vec![xn, gn],
        DtypeRepr::F32,
        dims(&out),
    );
    g
}

fn lower(x: &[usize], grid: &[usize]) -> anyhow::Result<usize> {
    TritonLowering::new()
        .lower(
            &sample_graph(x, grid, "bilinear", "zeros", true),
            LoweringMode::Inference,
        )
        .map(|dag| dag.len())
}

#[test]
fn test_the_export_s_shapes_lower() {
    // The export's own form: a single row sampled at nine positions, 19200 of them.
    assert_eq!(
        lower(&[19200, 28, 1, 104], &[19200, 1, 9, 2]).expect("export"),
        3
    );
}

#[test]
fn test_another_padding_mode_is_refused_by_name() {
    let msg = match TritonLowering::new().lower(
        &sample_graph(&[1, 1, 4, 4], &[1, 2, 2, 2], "bilinear", "border", true),
        LoweringMode::Inference,
    ) {
        Ok(_) => panic!("border padding should not lower"),
        Err(e) => e.to_string(),
    };
    assert!(msg.contains("\"border\""), "{msg}");
}

#[test]
fn test_a_grid_that_is_not_two_wide_is_refused() {
    let mut g = Graph::new();
    let x = g.add_node(
        Op::Input,
        vec![],
        DtypeRepr::F32,
        vec![Some(1), Some(1), Some(4), Some(4)],
    );
    let grid = g.add_node(
        Op::Input,
        vec![],
        DtypeRepr::F32,
        vec![Some(1), Some(2), Some(2), Some(3)],
    );
    g.add_node(
        Op::GridSample {
            mode: "bilinear".to_string(),
            padding_mode: "zeros".to_string(),
            align_corners: true,
        },
        vec![x, grid],
        DtypeRepr::F32,
        vec![Some(1), Some(1), Some(2), Some(2)],
    );
    let msg = match TritonLowering::new().lower(&g, LoweringMode::Inference) {
        Ok(_) => panic!("a three-wide grid should not lower"),
        Err(e) => e.to_string(),
    };
    assert!(msg.contains("has to be"), "{msg}");
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

    /// Bilinear sampling on the host, written out from ONNX's definition rather than
    /// from the kernel's arithmetic: the normalisation, the four taps and the zero
    /// padding are all spelled out here independently.
    fn reference(
        x: &[f32],
        x_shape: &[usize],
        grid: &[f32],
        grid_shape: &[usize],
        align_corners: bool,
    ) -> Vec<f32> {
        let (n, c, h_in, w_in) = (x_shape[0], x_shape[1], x_shape[2], x_shape[3]);
        let (h_out, w_out) = (grid_shape[1], grid_shape[2]);
        let denorm = |g: f32, extent: usize| -> f32 {
            if align_corners {
                (g + 1.0) * 0.5 * (extent as f32 - 1.0)
            } else {
                ((g + 1.0) * extent as f32 - 1.0) * 0.5
            }
        };
        let at = |ni: usize, ci: usize, yi: i64, xi: i64| -> f32 {
            if yi < 0 || yi >= h_in as i64 || xi < 0 || xi >= w_in as i64 {
                return 0.0;
            }
            x[((ni * c + ci) * h_in + yi as usize) * w_in + xi as usize]
        };

        let mut out = vec![0.0f32; n * c * h_out * w_out];
        for ni in 0..n {
            for ci in 0..c {
                for ho in 0..h_out {
                    for wo in 0..w_out {
                        let g = ((ni * h_out + ho) * w_out + wo) * 2;
                        let fx = denorm(grid[g], w_in);
                        let fy = denorm(grid[g + 1], h_in);
                        let x0 = fx.floor();
                        let y0 = fy.floor();
                        let tx = fx - x0;
                        let ty = fy - y0;
                        let (x0, y0) = (x0 as i64, y0 as i64);
                        let v = at(ni, ci, y0, x0) * (1.0 - ty) * (1.0 - tx)
                            + at(ni, ci, y0, x0 + 1) * (1.0 - ty) * tx
                            + at(ni, ci, y0 + 1, x0) * ty * (1.0 - tx)
                            + at(ni, ci, y0 + 1, x0 + 1) * ty * tx;
                        out[((ni * c + ci) * h_out + ho) * w_out + wo] = v;
                    }
                }
            }
        }
        out
    }

    fn run(x_shape: &[usize], grid_shape: &[usize], grid: &[f32], align_corners: bool) {
        let n_in: usize = x_shape.iter().product();
        let x: Vec<f32> = (0..n_in).map(|i| (i % 23) as f32 * 0.5 - 2.0).collect();
        let want = reference(&x, x_shape, grid, grid_shape, align_corners);

        let graph = sample_graph(x_shape, grid_shape, "bilinear", "zeros", align_corners);
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

        let feeds: Vec<&[f32]> = vec![&x, grid];
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
            "{x_shape:?} by {grid_shape:?}: worst {worst} at {at} (got {}, want {})",
            out[at],
            want[at]
        );
    }

    /// Positions between elements, at the edges, and outside: the last of those is
    /// where zero padding has to apply rather than the nearest element.
    #[test]
    fn test_sampling_inside_and_outside() {
        let grid = vec![
            -1.0, -1.0, // the first element exactly
            0.0, 0.0, // the middle
            0.3, -0.7, // between elements on both axes
            1.0, 1.0, // the last element exactly
            -1.6, 0.0, // outside: zero padding
            0.5, 2.2, // outside on the other axis
        ];
        run(&[1, 2, 3, 4], &[1, 2, 3, 2], &grid, true);
    }

    /// Without aligned corners the edges mean something else, and the first and last
    /// positions fall half an element outside the input.
    #[test]
    fn test_sampling_without_aligned_corners() {
        let grid = vec![-1.0, -1.0, 0.0, 0.0, 0.25, 0.75, 1.0, 1.0];
        run(&[1, 1, 4, 4], &[1, 2, 2, 2], &grid, false);
    }

    /// A single row, which is the export's shape: the y coordinate has one element to
    /// land on, so a mistake in the row arithmetic shows up as zeros.
    #[test]
    fn test_sampling_a_single_row() {
        // Five positions per batch, and the two batches look at different places.
        let grid = vec![
            -1.0, 0.0, -0.5, 0.0, 0.0, 0.0, 0.5, 0.0, 1.0, 0.0, //
            0.9, 0.0, 0.1, 0.0, -0.3, 0.0, -0.9, 0.0, 0.45, 0.0,
        ];
        run(&[2, 3, 1, 8], &[2, 1, 5, 2], &grid, true);
    }

    /// Several batches and channels: the grid is shared across channels but not
    /// across batches, which is the indexing most easily got wrong.
    #[test]
    fn test_the_grid_is_per_batch_and_shared_across_channels() {
        let grid = vec![
            // batch 0 looks at the top-left corner
            -1.0, -1.0, -0.5, -0.5, //
            // batch 1 looks at the bottom-right
            1.0, 1.0, 0.5, 0.5,
        ];
        run(&[2, 3, 4, 4], &[2, 1, 2, 2], &grid, true);
    }
}
