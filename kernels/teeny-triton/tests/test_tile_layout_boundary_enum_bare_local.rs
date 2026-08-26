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

//! Fourth variant of the boundary-mode probe, isolating whether
//! teenyc-3af.3 is really about *struct fields*, or whether this backend
//! has no custom-enum-discriminant support at all, independent of
//! aggregates. No `Tile`/struct here -- `BoundaryMode` is a bare local,
//! matched directly, with SROA left at its default (no
//! `-Zmir-enable-passes=-ScalarReplacementOfAggregates`). If this also
//! ICEs with the same `Discriminant for non-Option`, the gap is in
//! `Rvalue::Discriminant`/enum support generally, not specifically in how
//! it composes with `FieldSlot`.

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
fn tile_layout_boundary_enum_bare_local_probe<
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
    /// Same two-variant probe enum as the other boundary-enum tests --
    /// no struct wrapping it this time.
    enum BoundaryMode {
        Zero,
        Clamp,
    }

    let pid = T::program_id(Axis::X);
    let block_start = pid * BLOCK_SIZE;
    let offsets = T::arange(0, BLOCK_SIZE) + block_start;
    let in_bounds = offsets.lt(n_elements);

    // Bare local, no Tile/struct at all -- just the const-generic-driven
    // discriminant itself.
    let boundary = match USE_CLAMP {
        0 => BoundaryMode::Zero,
        _ => BoundaryMode::Clamp,
    };

    let default = match boundary {
        BoundaryMode::Zero => T::zeros::<D>(&[BLOCK_SIZE]),
        BoundaryMode::Clamp => T::zeros::<D>(&[BLOCK_SIZE]),
    };

    let x = T::load(
        x_ptr.add_offsets(offsets),
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
fn test_tile_layout_boundary_enum_bare_local() -> anyhow::Result<()> {
    let kernel = TileLayoutBoundaryEnumBareLocalProbe::<f32>::new(1024, 1);
    let teenyc_path = teeny_compiler::compiler::find_teenyc()?;
    let cache_dir =
        std::env::var("TEENYC_CACHE_DIR").unwrap_or_else(|_| "/tmp/teenyc_cache".to_string());
    let compiler = LlvmCompiler::new(teenyc_path, cache_dir)?;
    let target = Target::new(Capability::Sm90);
    let ptx_path: PathBuf = compiler.compile(&kernel, &target, true)?.into();
    let mlir = std::fs::read_to_string(ptx_path.with_extension("mlir"))?;

    assert_debug_snapshot!(
        "tile_layout_boundary_enum_bare_local_source",
        kernel.source()
    );
    assert_debug_snapshot!("tile_layout_boundary_enum_bare_local_mlir", mlir.trim());

    Ok(())
}
