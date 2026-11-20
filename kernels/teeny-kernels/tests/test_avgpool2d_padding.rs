// AvgPool2d with symmetric padding, and both denominators.
//
// The kernel had no padding at all, so the 32 of 34 AveragePool nodes in the
// FoundationStereo export that pad by 1 could not be expressed. Shape inference
// declined for the same reason, which is what left those subtrees unshaped.
//
// Values are checked against a host reference that implements ONNX AveragePool
// directly, over every element, on device. `count_include_pad` is the part a shape
// test cannot reach: it changes no extent and every border value.
//
//   cargo test -p teeny-kernels --test test_avgpool2d_padding
//   cargo test -p teeny-kernels --no-default-features --features qemu,hardware \
//       --test test_avgpool2d_padding

use teeny_core::graph::{DtypeRepr, Graph, Op};
use teeny_core::model::{Lowering, LoweringMode};
use teeny_kernels::graph::TritonLowering;

#[derive(Clone, Copy)]
struct Pool {
    kernel_h: usize,
    kernel_w: usize,
    stride_h: usize,
    stride_w: usize,
    pad_h: usize,
    pad_w: usize,
    count_include_pad: bool,
}

fn out_extent(extent: usize, kernel: usize, stride: usize, pad: usize) -> usize {
    (extent + 2 * pad - kernel) / stride + 1
}

/// ONNX `AveragePool` on the host, for NCHW input.
///
/// Only the execution tests compare against it, so it is unused without the
/// `hardware` feature.
///
/// The denominator is the whole point: with `count_include_pad` the window is always
/// `kernel_h * kernel_w`, and without it only the positions inside the input count.
#[cfg_attr(not(feature = "hardware"), allow(dead_code))]
fn reference(x: &[f32], n: usize, c: usize, h: usize, w: usize, p: Pool) -> (Vec<f32>, Vec<usize>) {
    let oh = out_extent(h, p.kernel_h, p.stride_h, p.pad_h);
    let ow = out_extent(w, p.kernel_w, p.stride_w, p.pad_w);
    let mut out = vec![0.0f32; n * c * oh * ow];

    for bi in 0..n {
        for ci in 0..c {
            for y in 0..oh {
                for xi in 0..ow {
                    let mut sum = 0.0f32;
                    let mut valid = 0usize;
                    for ky in 0..p.kernel_h {
                        for kx in 0..p.kernel_w {
                            let iy = (y * p.stride_h + ky) as isize - p.pad_h as isize;
                            let ix = (xi * p.stride_w + kx) as isize - p.pad_w as isize;
                            if iy < 0 || ix < 0 || iy >= h as isize || ix >= w as isize {
                                continue;
                            }
                            sum += x[((bi * c + ci) * h + iy as usize) * w + ix as usize];
                            valid += 1;
                        }
                    }
                    let denom = if p.count_include_pad {
                        p.kernel_h * p.kernel_w
                    } else {
                        valid.max(1)
                    };
                    out[((bi * c + ci) * oh + y) * ow + xi] = sum / denom as f32;
                }
            }
        }
    }
    (out, vec![n, c, oh, ow])
}

fn pool_graph(dims: &[usize], p: Pool) -> (Graph, Vec<usize>) {
    let (n, c, h, w) = (dims[0], dims[1], dims[2], dims[3]);
    let oh = out_extent(h, p.kernel_h, p.stride_h, p.pad_h);
    let ow = out_extent(w, p.kernel_w, p.stride_w, p.pad_w);
    let mut g = Graph::new();
    let x = g.add_node(
        Op::Input,
        vec![],
        DtypeRepr::F32,
        dims.iter().map(|&d| Some(d)).collect(),
    );
    g.add_node(
        Op::AvgPool2d {
            kernel_h: p.kernel_h,
            kernel_w: p.kernel_w,
            stride_h: p.stride_h,
            stride_w: p.stride_w,
            pad_h: p.pad_h,
            pad_w: p.pad_w,
            count_include_pad: p.count_include_pad,
        },
        vec![x],
        DtypeRepr::F32,
        vec![Some(n), Some(c), Some(oh), Some(ow)],
    );
    (g, vec![n, c, oh, ow])
}

/// The configuration every padded AveragePool in the FoundationStereo export uses.
const FS: Pool = Pool {
    kernel_h: 3,
    kernel_w: 3,
    stride_h: 2,
    stride_w: 2,
    pad_h: 1,
    pad_w: 1,
    count_include_pad: true,
};

#[test]
fn a_padded_pool_lowers() {
    let (graph, _) = pool_graph(&[1, 2, 8, 8], FS);
    let dag = TritonLowering::new()
        .lower(&graph, LoweringMode::Inference)
        .expect("a padded AvgPool2d should lower");
    assert_eq!(dag.len(), 2);
}

#[test]
fn both_denominators_lower() {
    for count_include_pad in [true, false] {
        let p = Pool {
            count_include_pad,
            ..FS
        };
        let (graph, _) = pool_graph(&[1, 2, 8, 8], p);
        TritonLowering::new()
            .lower(&graph, LoweringMode::Inference)
            .unwrap_or_else(|e| panic!("count_include_pad={count_include_pad}: {e}"));
    }
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

    fn run(dims: &[usize], p: Pool, label: &str) {
        let n: usize = dims.iter().product();
        // A ramp with a fraction, so a wrong window or a wrong denominator both show
        // up rather than landing on the same value by chance.
        let x: Vec<f32> = (0..n).map(|i| i as f32 * 0.25 - 3.0).collect();
        let (want, out_shape) = reference(&x, dims[0], dims[1], dims[2], dims[3], p);

        let (graph, graph_shape) = pool_graph(dims, p);
        assert_eq!(graph_shape, out_shape, "{label}: shape disagreement");

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
            .unwrap_or_else(|| panic!("{label}: no output of length {}", want.len()));

        let mut worst = 0.0f32;
        let mut at = 0usize;
        for i in 0..want.len() {
            let d = (out[i] - want[i]).abs();
            if d > worst {
                worst = d;
                at = i;
            }
        }
        if worst > 1e-5 {
            println!("  got[{at}]  = {}", out[at]);
            println!("  want[{at}] = {}", want[at]);
            println!("  got  = {:?}", &out[..out.len().min(16)]);
            println!("  want = {:?}", &want[..want.len().min(16)]);
        }
        assert!(
            worst <= 1e-5,
            "{label}: max |device - host| = {worst:e} at {at}"
        );
        println!("{label} -> {:?}, max err {worst:e}", out_shape);
    }

    /// The export's own configuration, which the kernel could not express at all.
    #[test]
    fn the_foundationstereo_pool_matches_the_reference() {
        run(&[1, 2, 8, 8], FS, "3x3 s2 p1 include_pad");
        run(&[1, 3, 12, 16], FS, "3x3 s2 p1 include_pad, larger");
    }

    /// The denominator is what `count_include_pad` selects, and it changes every
    /// border output while leaving the interior and the extent alone. A kernel that
    /// always divides by the full window passes the extent check and fails here.
    #[test]
    fn excluding_the_padding_changes_the_border_values() {
        let p = Pool {
            count_include_pad: false,
            ..FS
        };
        run(&[1, 2, 8, 8], p, "3x3 s2 p1 exclude_pad");

        // And the two really do differ, so the test above is not vacuous.
        let x: Vec<f32> = (0..64).map(|i| i as f32 * 0.25 - 3.0).collect();
        let (with, _) = reference(&x, 1, 1, 8, 8, FS);
        let (without, _) = reference(&x, 1, 1, 8, 8, p);
        assert_ne!(
            with, without,
            "the two denominators must differ somewhere, or this proves nothing"
        );
    }

    /// Padding has to leave an unpadded pool exactly as it was.
    #[test]
    fn an_unpadded_pool_is_unchanged() {
        let p = Pool {
            pad_h: 0,
            pad_w: 0,
            ..FS
        };
        run(&[1, 2, 8, 8], p, "3x3 s2 no padding");
        // The other FoundationStereo configuration: a 1x2 window, unpadded.
        let wide = Pool {
            kernel_h: 1,
            kernel_w: 2,
            stride_h: 1,
            stride_w: 2,
            pad_h: 0,
            pad_w: 0,
            count_include_pad: true,
        };
        run(&[1, 2, 8, 8], wide, "1x2 s1x2 no padding");
    }

    /// Padding on one axis only, and a window larger than the stride so the pools
    /// overlap.
    #[test]
    fn asymmetric_between_axes_and_overlapping_windows() {
        run(
            &[1, 2, 8, 8],
            Pool {
                pad_h: 1,
                pad_w: 0,
                ..FS
            },
            "pad rows only",
        );
        run(
            &[1, 2, 8, 8],
            Pool {
                pad_h: 0,
                pad_w: 1,
                ..FS
            },
            "pad columns only",
        );
        run(
            &[1, 2, 8, 8],
            Pool {
                stride_h: 1,
                stride_w: 1,
                count_include_pad: false,
                ..FS
            },
            "3x3 s1 p1 overlapping, exclude_pad",
        );
    }
}
