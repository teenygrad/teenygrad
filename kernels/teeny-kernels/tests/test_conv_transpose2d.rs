// ConvTranspose: a convolution run backwards, as an upsampling layer.
//
// Run from the input it wants atomics, because several input elements land on the same
// output element. Run from the output it does not: an output element collects from
// every kernel tap whose source input element exists. These tests check that the "if
// it exists" condition is right, which is where the stride and the padding meet.
//
//   cargo test -p teeny-kernels --test test_conv_transpose2d
//   cargo test -p teeny-kernels --no-default-features --features qemu,hardware \
//       --test test_conv_transpose2d

use teeny_core::graph::{DtypeRepr, Graph, Op};
use teeny_core::model::{Lowering, LoweringMode};
use teeny_kernels::graph::TritonLowering;

struct Spec {
    n: usize,
    c_in: usize,
    c_out: usize,
    h_in: usize,
    w_in: usize,
    kh: usize,
    kw: usize,
    stride: usize,
    pad: usize,
    bias: bool,
}

impl Spec {
    fn out(&self) -> (usize, usize) {
        (
            (self.h_in - 1) * self.stride - 2 * self.pad + self.kh,
            (self.w_in - 1) * self.stride - 2 * self.pad + self.kw,
        )
    }
}

fn convt_graph(s: &Spec) -> Graph {
    let (h_out, w_out) = s.out();
    let mut g = Graph::new();
    let x = g.add_node(
        Op::Input,
        vec![],
        DtypeRepr::F32,
        vec![Some(s.n), Some(s.c_in), Some(s.h_in), Some(s.w_in)],
    );
    let w = g.add_node(
        Op::Input,
        vec![],
        DtypeRepr::F32,
        vec![Some(s.c_in), Some(s.c_out), Some(s.kh), Some(s.kw)],
    );
    let mut inputs = vec![x, w];
    if s.bias {
        inputs.push(g.add_node(Op::Input, vec![], DtypeRepr::F32, vec![Some(s.c_out)]));
    }
    g.add_node(
        Op::ConvTranspose {
            in_channels: s.c_in,
            out_channels: s.c_out,
            kernel_h: s.kh,
            kernel_w: s.kw,
            stride_h: s.stride,
            stride_w: s.stride,
            padding_h: s.pad,
            padding_w: s.pad,
            output_padding_h: 0,
            output_padding_w: 0,
            groups: 1,
            has_bias: s.bias,
        },
        inputs,
        DtypeRepr::F32,
        vec![Some(s.n), Some(s.c_out), Some(h_out), Some(w_out)],
    );
    g
}

#[test]
fn the_export_s_shapes_lower() {
    // The export's commonest: a 4x4 kernel at stride 2 with a pad of 1, doubling.
    let s = Spec {
        n: 26,
        c_in: 112,
        c_out: 56,
        h_in: 30,
        w_in: 40,
        kh: 4,
        kw: 4,
        stride: 2,
        pad: 1,
        bias: false,
    };
    assert_eq!(s.out(), (60, 80));
    assert_eq!(
        TritonLowering::new()
            .lower(&convt_graph(&s), LoweringMode::Inference)
            .expect("export shape")
            .len(),
        3
    );
}

#[test]
fn groups_are_refused_by_name() {
    let s = Spec {
        n: 1,
        c_in: 2,
        c_out: 2,
        h_in: 2,
        w_in: 2,
        kh: 2,
        kw: 2,
        stride: 1,
        pad: 0,
        bias: false,
    };
    let mut g = convt_graph(&s);
    let last = g.nodes.len() - 1;
    if let Op::ConvTranspose { groups, .. } = &mut g.nodes[last].op {
        *groups = 2;
    }
    let msg = match TritonLowering::new().lower(&g, LoweringMode::Inference) {
        Ok(_) => panic!("grouped should not lower"),
        Err(e) => e.to_string(),
    };
    assert!(msg.contains("2 groups"), "{msg}");
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

    /// The transposed convolution on the host, written the way it is defined: from the
    /// input, scattering each element over its window. That is the opposite direction
    /// to the kernel's, so an off-by-one in the stride or the padding shows up as a
    /// mismatch rather than agreeing with itself.
    fn reference(x: &[f32], w: &[f32], b: Option<&[f32]>, s: &Spec) -> Vec<f32> {
        let (h_out, w_out) = s.out();
        let mut out = vec![0.0f32; s.n * s.c_out * h_out * w_out];
        if let Some(b) = b {
            for ni in 0..s.n {
                for co in 0..s.c_out {
                    for i in 0..h_out * w_out {
                        out[(ni * s.c_out + co) * h_out * w_out + i] = b[co];
                    }
                }
            }
        }
        for ni in 0..s.n {
            for ci in 0..s.c_in {
                for ih in 0..s.h_in {
                    for iw in 0..s.w_in {
                        let xv = x[((ni * s.c_in + ci) * s.h_in + ih) * s.w_in + iw];
                        for co in 0..s.c_out {
                            for kh in 0..s.kh {
                                for kw in 0..s.kw {
                                    let oh = ih * s.stride + kh;
                                    let ow = iw * s.stride + kw;
                                    if oh < s.pad || ow < s.pad {
                                        continue;
                                    }
                                    let (oh, ow) = (oh - s.pad, ow - s.pad);
                                    if oh >= h_out || ow >= w_out {
                                        continue;
                                    }
                                    let wv = w[((ci * s.c_out + co) * s.kh + kh) * s.kw + kw];
                                    out[((ni * s.c_out + co) * h_out + oh) * w_out + ow] += xv * wv;
                                }
                            }
                        }
                    }
                }
            }
        }
        out
    }

    fn run(s: &Spec) {
        let n_x = s.n * s.c_in * s.h_in * s.w_in;
        let n_w = s.c_in * s.c_out * s.kh * s.kw;
        let x: Vec<f32> = (0..n_x).map(|i| (i % 13) as f32 * 0.25 - 1.0).collect();
        let w: Vec<f32> = (0..n_w).map(|i| (i % 7) as f32 * 0.5 - 1.5).collect();
        let bias: Vec<f32> = (0..s.c_out).map(|i| i as f32 * 0.125 + 0.5).collect();
        let want = reference(&x, &w, s.bias.then_some(bias.as_slice()), s);

        let graph = convt_graph(s);
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

        let mut feeds: Vec<&[f32]> = vec![&x, &w];
        if s.bias {
            feeds.push(&bias);
        }
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
            "worst {worst} at {at} (got {}, want {})",
            out[at],
            want[at]
        );
    }

    /// The export's shape in miniature: a 4x4 kernel at stride 2 with a pad of 1,
    /// which doubles the extents exactly.
    #[test]
    fn doubling_with_a_four_tap_kernel() {
        run(&Spec {
            n: 1,
            c_in: 2,
            c_out: 3,
            h_in: 3,
            w_in: 4,
            kh: 4,
            kw: 4,
            stride: 2,
            pad: 1,
            bias: false,
        });
    }

    /// Stride 1 with no padding, where every tap of every element lands inside the
    /// output and the condition never rejects anything.
    #[test]
    fn stride_one_without_padding() {
        run(&Spec {
            n: 2,
            c_in: 2,
            c_out: 2,
            h_in: 3,
            w_in: 3,
            kh: 3,
            kw: 2,
            stride: 1,
            pad: 0,
            bias: false,
        });
    }

    /// Stride 4 with a 4x4 kernel and no padding: the export has one of these, and
    /// every output position has exactly one contributing tap, so a wrong divisibility
    /// test leaves most of the output empty.
    #[test]
    fn stride_four_with_a_bias() {
        run(&Spec {
            n: 1,
            c_in: 3,
            c_out: 2,
            h_in: 2,
            w_in: 3,
            kh: 4,
            kw: 4,
            stride: 4,
            pad: 0,
            bias: true,
        });
    }

    /// A stride larger than the kernel, which leaves output positions no tap reaches
    /// at all: those have to come out as zero, or as the bias.
    #[test]
    fn a_stride_wider_than_the_kernel_leaves_gaps() {
        run(&Spec {
            n: 1,
            c_in: 1,
            c_out: 1,
            h_in: 3,
            w_in: 3,
            kh: 2,
            kw: 2,
            stride: 3,
            pad: 0,
            bias: true,
        });
    }

    /// Padding that crops more than one element off each side, where the first output
    /// element comes from a tap in the middle of the kernel.
    #[test]
    fn padding_that_crops_two() {
        run(&Spec {
            n: 1,
            c_in: 2,
            c_out: 2,
            h_in: 4,
            w_in: 4,
            kh: 5,
            kw: 5,
            stride: 2,
            pad: 2,
            bias: false,
        });
    }
}
