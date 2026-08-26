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

//! Third variant of the boundary-mode probe. The realistic `Tile<T,D>`
//! design picks `BoundaryMode` per kernel *monomorphization* (a const
//! generic, like `BLOCK_SIZE` already is), not from a runtime kernel
//! parameter -- `test_tile_layout_boundary_enum_stress.rs` deliberately
//! used a runtime parameter to stress-test the worst case, but that isn't
//! the shape any real kernel would actually use.
//!
//! This variant checks whether *that* realistic shape also hits
//! teenyc-3af.3's `todo!("Discriminant for non-Option")`, or whether a
//! const-generic-driven discriminant folds away before reaching codegen
//! the same way `test_tile_layout_boundary_enum.rs`'s hardcoded literal
//! did -- which determines whether teenyc-3af.3 is an actual blocker for
//! any real kernel, or only for the (out-of-scope) runtime-varying case.

use std::path::PathBuf;

use insta::assert_debug_snapshot;
use teeny_compiler::compiler::backend::llvm::compiler::LlvmCompiler;
use teeny_core::{compiler::Compiler, device::program::Kernel, dtype::Num};
use teeny_cuda::compiler::target::{Capability, Target};
use teeny_macros::kernel;
use teeny_triton::triton::{
    types::{AddOffsets, Comparison},
    *,
};

#[kernel]
#[allow(unused)]
fn tile_layout_boundary_enum_const_generic_probe<
    T: Triton,
    D: Num,
    const BLOCK_SIZE: i32,
    const USE_CLAMP: i32,
>(
    x_ptr: In<T::Pointer<D>>,
    y_ptr: Out<T::Pointer<D>>,
    n_elements: i32,
) where
    T::I32Tensor: types::Tensor<i32, 1>,
    T::I32Tensor: Comparison<i32, BoolTensor = T::BoolTensor>,
    T::Pointer<D>: AddOffsets<i32, 1, T::I32Tensor, Output = T::Tensor<T::Pointer<D>>>,
{
    /// Same two-variant probe enum as the other boundary-enum tests.
    enum BoundaryMode {
        Zero,
        Clamp,
    }

    /// Local items don't inherit the enclosing function's generics (see
    /// `test_kitchen_sink.rs`'s nested `combine_num`), so this declares
    /// its own `TT`/`DD`.
    struct BoundaryTile<TT: Triton, DD: Num> {
        base: TT::Pointer<DD>,
        boundary: BoundaryMode,
    }

    let pid = T::program_id(Axis::X);
    let block_start = pid * BLOCK_SIZE;
    let offsets = T::arange(0, BLOCK_SIZE) + block_start;
    let in_bounds = offsets.lt(n_elements);

    // Const-generic-driven discriminant -- the realistic per-monomorphization
    // shape (like BLOCK_SIZE), not a runtime kernel argument
    // (test_tile_layout_boundary_enum_stress.rs) or an inline literal
    // (test_tile_layout_boundary_enum.rs).
    let boundary = match USE_CLAMP {
        0 => BoundaryMode::Zero,
        _ => BoundaryMode::Clamp,
    };

    let tile = BoundaryTile::<T, D> {
        base: x_ptr,
        boundary,
    };

    let default = match tile.boundary {
        BoundaryMode::Zero => T::zeros::<D>(&[BLOCK_SIZE]),
        BoundaryMode::Clamp => T::zeros::<D>(&[BLOCK_SIZE]),
    };

    let x = T::load(
        tile.base.add_offsets(offsets),
        Some(in_bounds),
        Some(default),
        &[],
        None,
        None,
        None,
        false,
    );

    T::store(
        y_ptr.add_offsets(offsets),
        x,
        Some(in_bounds),
        &[],
        None,
        None,
    );
}

#[test]
fn test_tile_layout_boundary_enum_const_generic() -> anyhow::Result<()> {
    let kernel = TileLayoutBoundaryEnumConstGenericProbe::<f32>::new(1024, 1);
    let teenyc_path = teeny_compiler::compiler::find_teenyc()?;
    let cache_dir =
        std::env::var("TEENYC_CACHE_DIR").unwrap_or_else(|_| "/tmp/teenyc_cache".to_string());
    let compiler = LlvmCompiler::new(teenyc_path, cache_dir)?
        .with_extra_rustc_flags(["-Zmir-enable-passes=-ScalarReplacementOfAggregates"]);
    let target = Target::new(Capability::Sm90);
    let ptx_path: PathBuf = compiler.compile(&kernel, &target, true)?.into();
    let mlir = std::fs::read_to_string(ptx_path.with_extension("mlir"))?;

    assert_debug_snapshot!(
        "tile_layout_boundary_enum_const_generic_source",
        kernel.source()
    );
    assert_debug_snapshot!("tile_layout_boundary_enum_const_generic_mlir", mlir.trim());

    Ok(())
}
