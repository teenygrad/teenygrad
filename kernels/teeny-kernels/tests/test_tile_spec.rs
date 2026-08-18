/*
 * Copyright (c) 2026 teenygrad (https://teenygrad.org).
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

//! Unit tests for `#[tile(...)]`-derived `KernelTileSpec` metadata
//! (teenygrad-3w0.1: flat 1D; teenygrad-3w0.2: multi-axis/reduction;
//! teenygrad-3w0.7: `#[tile_loop(...)]` loop-carry metadata).
//!
//! No CUDA / kernel compilation involved — this only checks the macro-generated
//! `tile_spec()` const fn produces the declared metadata.

use teeny_kernels::math::gemm::MatmulForward;
use teeny_kernels::nn::activation::elu::{EluBackward, EluForward};
use teeny_kernels::nn::attention::flash_attn2::FlashAttention2Forward;
use teeny_kernels::nn::conv::conv2d::Conv2dForward;
use teeny_triton::{TileSpecLayout, mem_traffic, tile::KernelTileSpec};

fn axis_names(spec: &KernelTileSpec) -> (Vec<&'static str>, Vec<&'static str>) {
    let inputs = spec.inputs.iter().map(|t| t.axes[0].block_const).collect();
    let outputs = spec.outputs.iter().map(|t| t.axes[0].block_const).collect();
    (inputs, outputs)
}

#[test]
fn elu_forward_tile_spec_matches_declared_attrs() {
    let spec = EluForward::<f32>::tile_spec();
    assert_eq!(spec.inputs.len(), 1, "one #[tile(...)] In pointer (x_ptr)");
    assert_eq!(
        spec.outputs.len(),
        1,
        "one #[tile(...)] Out pointer (y_ptr)"
    );

    let x = &spec.inputs[0];
    assert_eq!(x.param, "x_ptr");
    assert_eq!(x.rank, 1);
    assert_eq!(x.axes.len(), 1);
    assert_eq!(x.axes[0].dim, 0);
    assert_eq!(x.axes[0].block_const, "BLOCK_SIZE");
    assert_eq!(x.axes[0].extent_param, "n_elements");
    assert_eq!(x.reduction_axis, None);

    let y = &spec.outputs[0];
    assert_eq!(y.param, "y_ptr");
    assert_eq!(y.axes[0].block_const, "BLOCK_SIZE");
    assert_eq!(y.axes[0].extent_param, "n_elements");

    // Also reachable through the trait object, not just the inherent const fn.
    let via_trait = <EluForward<f32> as TileSpecLayout>::tile_spec();
    assert_eq!(axis_names(&via_trait), axis_names(&spec));
}

#[test]
fn elu_backward_tile_spec_has_two_inputs_one_output() {
    let spec = EluBackward::<f32>::tile_spec();
    // dy_ptr, x_ptr are both #[tile(...)]-tagged In pointers; dx_ptr is Out.
    assert_eq!(spec.inputs.len(), 2);
    assert_eq!(spec.outputs.len(), 1);
    assert_eq!(spec.inputs[0].param, "dy_ptr");
    assert_eq!(spec.inputs[1].param, "x_ptr");
    assert_eq!(spec.outputs[0].param, "dx_ptr");
    for t in spec.inputs.iter().chain(spec.outputs.iter()) {
        assert_eq!(t.axes[0].block_const, "BLOCK_SIZE");
        assert_eq!(t.axes[0].extent_param, "n_elements");
        assert_eq!(t.reduction_axis, None);
    }
}

#[test]
fn tile_spec_is_a_const_fn() {
    // Compile-time-evaluability check: if this doesn't const-eval, the build fails.
    const SPEC: KernelTileSpec = EluForward::<f32>::tile_spec();
    assert_eq!(SPEC.inputs.len(), 1);
}

#[test]
fn matmul_forward_tile_spec_has_two_axes_and_a_reduction_axis() {
    let spec = MatmulForward::<f32>::tile_spec();
    // a_ptr, b_ptr are In; c_ptr is InOut, so it appears in both lists.
    assert_eq!(
        spec.inputs.len(),
        3,
        "a_ptr, b_ptr, c_ptr (InOut counts as input too)"
    );
    assert_eq!(spec.outputs.len(), 1, "c_ptr (InOut counts as output too)");

    let a = spec.inputs.iter().find(|t| t.param == "a_ptr").unwrap();
    assert_eq!(a.rank, 2);
    assert_eq!(a.axes.len(), 2);
    assert_eq!(
        (a.axes[0].block_const, a.axes[0].extent_param),
        ("BLOCK_M", "M")
    );
    assert_eq!(
        (a.axes[1].block_const, a.axes[1].extent_param),
        ("BLOCK_K", "K")
    );
    assert_eq!(
        a.reduction_axis,
        Some(1),
        "A reduces along its K axis (dim 1)"
    );

    let b = spec.inputs.iter().find(|t| t.param == "b_ptr").unwrap();
    assert_eq!(
        (b.axes[0].block_const, b.axes[0].extent_param),
        ("BLOCK_K", "K")
    );
    assert_eq!(
        (b.axes[1].block_const, b.axes[1].extent_param),
        ("BLOCK_N", "N")
    );
    assert_eq!(
        b.reduction_axis,
        Some(0),
        "B reduces along its K axis (dim 0)"
    );

    let c_in = spec.inputs.iter().find(|t| t.param == "c_ptr").unwrap();
    let c_out = spec.outputs.iter().find(|t| t.param == "c_ptr").unwrap();
    assert_eq!(
        c_in, c_out,
        "the InOut c_ptr spec is identical in both lists"
    );
    assert_eq!(
        c_in.reduction_axis, None,
        "C's own axes aren't reduced over"
    );
}

#[test]
fn mem_traffic_from_a_real_kernel_tile_spec() {
    // elu_forward at BLOCK_SIZE=256, n_elements=1000, f32: ceil(1000/256)=4
    // tiles of 256 elements each, for both x_ptr (input) and y_ptr (output).
    let spec = EluForward::<f32>::tile_spec();
    let resolve = |name: &str| match name {
        "BLOCK_SIZE" => 256,
        "n_elements" => 1000,
        other => panic!("unexpected param {other}"),
    };
    assert_eq!(mem_traffic(&spec, 4, resolve), 2 * 4 * 256 * 4);
}

#[test]
fn conv2d_forward_tile_spec_covers_output_and_windowed_input() {
    // y_ptr's OW axis is a plain arange(BLOCK_OW)+ow_start slice. x_ptr's W
    // axis is a strided/padded sliding window (teenygrad-3w0.5): its tile
    // spec declares the conservative-upper-bound window fields instead of
    // being left untagged. w_ptr stays untagged (its per-tap scalar load
    // doesn't have a tile shape at all).
    let spec = Conv2dForward::<f32>::tile_spec();
    assert_eq!(spec.inputs.len(), 1, "x_ptr only — w_ptr stays untagged");
    assert_eq!(spec.outputs.len(), 1);

    let y = &spec.outputs[0];
    assert_eq!(y.param, "y_ptr");
    assert_eq!(y.axes[0].block_const, "BLOCK_OW");
    assert_eq!(y.axes[0].extent_param, "OW");
    assert!(y.axes[0].window.is_none());
    assert_eq!(y.untiled_dims, &["_B", "C_OUT", "OH"]);

    let x = &spec.inputs[0];
    assert_eq!(x.param, "x_ptr");
    assert_eq!(x.axes[0].block_const, "BLOCK_OW");
    assert_eq!(x.axes[0].extent_param, "W");
    let window = x.axes[0].window.expect("x_ptr's W axis is windowed");
    assert_eq!(window.stride_const, "STRIDE_W");
    assert_eq!(window.pad_const, "PAD_W");
    assert_eq!(window.kernel_size_const, "KW");
    assert_eq!(x.untiled_dims, &["_B", "C_IN", "H"]);
}

#[test]
fn flash_attention2_forward_tile_spec_declares_loop_carry_only() {
    // No #[tile(...)]-tagged pointers (q_ptr/k_ptr/v_ptr/o_ptr/l_ptr all
    // untagged — HEAD_DIM-wide loads are compile-time-fixed, not a
    // grid-varying tile), just the online-softmax loop-carry metadata.
    let spec = FlashAttention2Forward::<f32>::tile_spec();
    assert_eq!(spec.inputs.len(), 0);
    assert_eq!(spec.outputs.len(), 0);

    let loop_spec = spec
        .loop_spec
        .expect("#[tile_loop(...)] declared on this kernel");
    assert_eq!(loop_spec.trip_count_param, "n_ctx_k");
    assert_eq!(loop_spec.carries.len(), 3);
    let names: Vec<&str> = loop_spec.carries.iter().map(|c| c.name).collect();
    assert_eq!(names, vec!["acc", "m_i", "l_i"]);
    for carry in loop_spec.carries {
        assert_eq!(carry.shape_consts, &["HEAD_DIM"]);
    }
}

#[test]
fn mem_traffic_accounts_for_conv2d_windowed_input() {
    // BLOCK_OW=32, STRIDE_W=1, KW=3, W=OW=64: y's footprint per tile is the
    // plain block size (32); x's is the conservative upper bound
    // (32-1)*1+3 = 34, wider than the output tile since the receptive field
    // overlaps between adjacent output positions. B=1, C_IN=C_OUT=1, OH=1
    // keep the untiled-dims multiplier trivial so this test still isolates
    // the windowed-axis math specifically (a separate test below covers the
    // untiled-dims multiplier itself).
    let spec = Conv2dForward::<f32>::tile_spec();
    let resolve = |name: &str| match name {
        "BLOCK_OW" => 32,
        "OW" | "W" => 64,
        "STRIDE_W" => 1,
        "KW" => 3,
        "_B" | "C_IN" | "C_OUT" | "OH" | "H" => 1,
        other => panic!("unexpected param {other}"),
    };
    let n_tiles = 2u64; // ceil(64/32)
    let y_traffic = n_tiles * 32 * 4;
    let x_traffic = n_tiles * 34 * 4;
    assert_eq!(mem_traffic(&spec, 4, resolve), x_traffic + y_traffic);
}

#[test]
fn mem_traffic_multiplies_in_conv2d_untiled_dims() {
    // Same shape as above but with real B/C_IN/C_OUT/OH values -- confirms
    // the teenygrad-3w0.8 fix: previously these were silently omitted
    // entirely (see teenygrad-3w0.4's calibration test, which measured this
    // gap on real hardware before untiled_dims existed).
    let spec = Conv2dForward::<f32>::tile_spec();
    let resolve = |name: &str| match name {
        "BLOCK_OW" => 32,
        "OW" | "W" => 64,
        "STRIDE_W" => 1,
        "KW" => 3,
        "_B" => 4,
        "C_IN" => 16,
        "C_OUT" => 32,
        "OH" => 64,
        "H" => 64,
        other => panic!("unexpected param {other}"),
    };
    let n_ow_tiles = 2u64; // ceil(64/32)
    let y_traffic = n_ow_tiles * 32 * 4 * (4 * 32 * 64); // B * C_OUT * OH
    let x_traffic = n_ow_tiles * 34 * 4 * (4 * 16 * 64); // B * C_IN * H
    assert_eq!(mem_traffic(&spec, 4, resolve), x_traffic + y_traffic);
}
