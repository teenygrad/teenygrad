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

//! Probes the "per-axis boundary mode" case of a CuTe-style `Tile<T,D>`
//! layout: a kernel-local struct holding a field of a *custom
//! multi-variant enum* (`BoundaryMode`), matched inside the kernel body to
//! pick the load's default value.
//!
//! **This test alone is inconclusive** -- `boundary` here is a
//! struct-literal constant (`BoundaryMode::Zero`), so the emitted MLIR
//! shows one folded load with no branch, which is equally consistent with
//! "the enum survived as a field and got constant-folded downstream" or
//! "it never survived as a field at all." See
//! `test_tile_layout_boundary_enum_stress.rs` for the real answer: with a
//! genuinely runtime-decided discriminant, `teenyc` ICEs with `not yet
//! implemented: Discriminant for non-Option` -- a general enum's
//! discriminant read is simply unimplemented in codegen, only `Option<T>`
//! is special-cased.
//!
//! This is the genuinely unproven case: teenyc-3af.1 only proves the
//! built-in `Option<T>` survives as a struct field (see
//! `../teeny/compiler/rustc_codegen_llvm/tests/test_struct_option.rs`) --
//! no ordinary multi-variant `enum` field has any existing precedent
//! anywhere in this repo or in `../teeny`'s own tests, and
//! `ConstParamTy`/`adt_const_params` don't exist in this `no_core`
//! environment at all, so a `BoundaryMode` can only be probed as a
//! *runtime* enum field (not a const generic) here.

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
fn tile_layout_boundary_enum_probe<T: Triton, D: Num, const BLOCK_SIZE: i32>(
    x_ptr: In<T::Pointer<D>>,
    y_ptr: Out<T::Pointer<D>>,
    n_elements: i32,
) where
    T::I32Tensor: types::Tensor<i32, 1>,
    T::I32Tensor: Comparison<i32, BoolTensor = T::BoolTensor>,
    T::Pointer<D>: AddOffsets<i32, 1, T::I32Tensor, Output = T::Tensor<T::Pointer<D>>>,
{
    /// How an out-of-range coordinate on this axis is resolved. `Zero`
    /// masks the load (today's behavior); `Clamp` would fold the
    /// coordinate instead -- this probe only needs two variants to
    /// exercise a real multi-arm `match`, not full coverage.
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

    let tile = BoundaryTile::<T, D> {
        base: x_ptr,
        boundary: BoundaryMode::Zero,
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
fn test_tile_layout_boundary_enum() -> anyhow::Result<()> {
    let kernel = TileLayoutBoundaryEnumProbe::<f32>::new(1024);
    let teenyc_path = teeny_compiler::compiler::find_teenyc()?;
    let cache_dir =
        std::env::var("TEENYC_CACHE_DIR").unwrap_or_else(|_| "/tmp/teenyc_cache".to_string());
    let compiler = LlvmCompiler::new(teenyc_path, cache_dir)?;
    let target = Target::new(Capability::Sm90);
    let ptx_path: PathBuf = compiler.compile(&kernel, &target, true)?.into();
    let mlir = std::fs::read_to_string(ptx_path.with_extension("mlir"))?;

    assert_debug_snapshot!("tile_layout_boundary_enum_source", kernel.source());
    assert_debug_snapshot!("tile_layout_boundary_enum_mlir", mlir.trim());

    Ok(())
}
