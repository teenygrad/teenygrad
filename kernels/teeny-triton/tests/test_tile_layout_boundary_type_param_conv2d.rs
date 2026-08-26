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

//! Real end-to-end test of the type-parameterized `BoundaryFold` design
//! (as an alternative to the runtime/const-generic enum approach blocked
//! by teenyc-3af.3) against a conv2d-shaped kernel: a genuine `KH x KW`
//! sliding-window accumulation loop with per-axis stride/padding, where
//! the out-of-bounds default value for each windowed load comes from a
//! `B: BoundaryFold<T, D>` type parameter (`teeny_triton::triton::{Zero,
//! Clamp}`) instead of a `match` on an enum field. Compiles the *same*
//! generic kernel function twice, once per marker type, and confirms both
//! monomorphize to distinct, straight-line MLIR with no branch/select/
//! discriminant of any kind -- unlike the enum-based probes
//! (`test_tile_layout_boundary_enum*.rs`), this never touches
//! `Rvalue::Discriminant` at all, since there's no enum value anywhere.
//!
//! Single input/output channel, no groups -- enough to exercise real 2-D
//! windowed addressing (stride, padding, per-iteration boundary check)
//! without the unrelated complexity (multi-channel weight layout, groups)
//! a full production conv2d carries.

use std::path::PathBuf;

use insta::assert_debug_snapshot;
use teeny_compiler::compiler::backend::llvm::compiler::LlvmCompiler;
use teeny_core::{compiler::Compiler, device::program::Kernel, dtype::Num};
use teeny_cuda::compiler::target::{Capability, Target};
use teeny_macros::tiled_kernel;
use teeny_triton::triton::{
    BoundaryFold, Clamp, Zero,
    types::{AddOffsets, Comparison},
    *,
};

#[tiled_kernel]
#[allow(unused)]
#[allow(clippy::too_many_arguments)]
fn conv2d_boundary_type_param_probe<
    T: Triton,
    D: Num,
    B: BoundaryFold<D>,
    const KH: i32,
    const KW: i32,
    const STRIDE_H: i32,
    const STRIDE_W: i32,
    const PAD_H: i32,
    const PAD_W: i32,
    const BLOCK_OW: i32,
>(
    x_ptr: In<T::Pointer<D>>,
    w_ptr: In<T::Pointer<D>>,
    y_ptr: Out<T::Pointer<D>>,
    H: i32,
    W: i32,
    OH: i32,
    OW: i32,
) where
    T::I32Tensor: types::Tensor<i32, 1>,
    T::I32Tensor: Comparison<i32, BoolTensor = T::BoolTensor>,
    T::BoolTensor: core::ops::BitAnd<Output = T::BoolTensor>,
    T::Pointer<D>: AddOffsets<i32, 1, T::I32Tensor, Output = T::Tensor<T::Pointer<D>>>,
{
    let pid = T::program_id(Axis::X);
    let num_ow_tiles = T::cdiv(OW, BLOCK_OW);
    let ow_tile = pid % num_ow_tiles;
    let oh = pid / num_ow_tiles;

    let ow_start = ow_tile * BLOCK_OW;
    let ow_range = T::arange(0, BLOCK_OW) + ow_start;
    let ow_mask = ow_range.lt(OW);

    let mut acc = T::zeros::<D>(&[BLOCK_OW]);

    let loop_bound = KH * KW;
    for idx in 0..loop_bound {
        let kw = idx % KW;
        let kh = idx / KW;

        let ih = oh * STRIDE_H + kh - PAD_H;
        let iw_range = ow_range * STRIDE_W + kw - PAD_W;

        #[allow(clippy::erasing_op)]
        let ih_t = ow_range * 0 + ih;
        let h_in_bounds = ih_t.ge(0) & ih_t.lt(H);
        let w_in_bounds = iw_range.ge(0) & iw_range.lt(W);
        let load_mask = ow_mask & h_in_bounds & w_in_bounds;

        let x_offsets = iw_range + ih * W;
        // The point of this probe: B::default_value picks the boundary
        // behavior via monomorphization, not a runtime/const-generic
        // enum match.
        let x_tile = T::load(
            x_ptr.add_offsets(x_offsets),
            Some(load_mask),
            Some(B::default_value::<T>(&[BLOCK_OW])),
            &[],
            None,
            None,
            None,
            false,
        );

        let w_off = T::arange(0, 1) + idx;
        let w_1 = T::load(
            w_ptr.add_offsets(w_off),
            None,
            None,
            &[],
            None,
            None,
            None,
            false,
        );
        let w_tile = T::broadcast_to(w_1, &[BLOCK_OW]);

        acc = acc + x_tile * w_tile;
    }

    let out_offsets = ow_range + oh * OW;
    T::store(
        y_ptr.add_offsets(out_offsets),
        acc,
        Some(ow_mask),
        &[],
        None,
        None,
    );
}

#[test]
fn test_conv2d_boundary_type_param_zero_and_clamp_both_compile() -> anyhow::Result<()> {
    let teenyc_path = teeny_compiler::compiler::find_teenyc()?;
    let cache_dir =
        std::env::var("TEENYC_CACHE_DIR").unwrap_or_else(|_| "/tmp/teenyc_cache".to_string());
    let compiler = LlvmCompiler::new(teenyc_path, cache_dir)?;
    let target = Target::new(Capability::Sm90);

    let zero_kernel = Conv2dBoundaryTypeParamProbe::<f32, Zero>::new(3, 3, 1, 1, 1, 1, 16);
    let zero_ptx: PathBuf = compiler.compile(&zero_kernel, &target, true)?.into();
    let zero_mlir = std::fs::read_to_string(zero_ptx.with_extension("mlir"))?;

    let clamp_kernel = Conv2dBoundaryTypeParamProbe::<f32, Clamp>::new(3, 3, 1, 1, 1, 1, 16);
    let clamp_ptx: PathBuf = compiler.compile(&clamp_kernel, &target, true)?.into();
    let clamp_mlir = std::fs::read_to_string(clamp_ptx.with_extension("mlir"))?;

    // Distinct BoundaryFold impls must produce distinct compiled bodies
    // (Clamp's extra arith.addf), not the same code twice.
    assert_ne!(
        zero_mlir, clamp_mlir,
        "Zero and Clamp instantiations should compile to different MLIR"
    );

    // Neither should contain any branch/select/discriminant-related
    // construct -- the whole point of the type-parameter approach is that
    // boundary-mode selection is fully resolved at monomorphization, with
    // nothing left for the MLIR to represent a choice about.
    for (label, mlir) in [("Zero", &zero_mlir), ("Clamp", &clamp_mlir)] {
        for needle in ["cf.br", "cf.cond_br", "scf.if", "arith.select"] {
            assert!(
                !mlir.contains(needle),
                "{label} instantiation's MLIR unexpectedly contains `{needle}`; \
                 boundary-mode selection should be branch-free: {mlir}"
            );
        }
    }

    assert_debug_snapshot!(
        "conv2d_boundary_type_param_zero_source",
        zero_kernel.source()
    );
    assert_debug_snapshot!("conv2d_boundary_type_param_zero_mlir", zero_mlir.trim());
    assert_debug_snapshot!(
        "conv2d_boundary_type_param_clamp_source",
        clamp_kernel.source()
    );
    assert_debug_snapshot!("conv2d_boundary_type_param_clamp_mlir", clamp_mlir.trim());

    Ok(())
}
