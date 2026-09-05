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

//! Probes the "Affine" case of a CuTe-style `Tile<T,D>` layout: a
//! kernel-local struct with a pointer field plus *two* plain (non-Option)
//! scalar fields (`shape`, `stride`), used to compute an offset via
//! `coord * stride` before loading.
//!
//! teenyc-3af.1 (`nest Option fields in kernel-local structs`) proves a
//! 2-field `{ tensor: Value, mask: Option<Value> }` shape survives SROA
//! (see `../teeny/compiler/rustc_codegen_llvm/tests/test_struct_option.rs`,
//! and the real `teeny_triton::triton::tile::Tile<T,D>` it was written to
//! support). This probe checks whether that generalizes to a 3-field,
//! all-plain (no `Option` at all) struct with a pointer field -- the
//! shape a real affine `Tile` descriptor (`base`/`shape`/`stride`) needs.

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
fn tile_layout_affine_probe<T: Triton, D: Num, const BLOCK_SIZE: i32>(
    x_ptr: In<T::Pointer<D>>,
    y_ptr: Out<T::Pointer<D>>,
    n_elements: i32,
) where
    T::I32Tensor: types::Tensor<i32, 1>,
    T::I32Tensor: Comparison<i32, BoolTensor = T::BoolTensor>,
    T::Pointer<D>: AddOffsets<i32, 1, T::I32Tensor, Output = T::Tensor<T::Pointer<D>>>,
{
    /// Minimal affine tile descriptor: base pointer + this axis's shape
    /// and stride, both plain `i32` fields (no `Option` anywhere) --
    /// probes whether a >2-field, all-plain struct survives the same way
    /// teenyc-3af.1's 2-field fixture does. Local items don't inherit the
    /// enclosing function's generics (see `test_kitchen_sink.rs`'s nested
    /// `combine_num`), so this declares its own `TT`/`DD`.
    struct AffineTile<TT: Triton, DD: Num> {
        base: TT::Pointer<DD>,
        shape: i32,
        stride: i32,
    }

    let pid = T::program_id(Axis::X);
    let block_start = pid * BLOCK_SIZE;

    let tile = AffineTile::<T, D> {
        base: x_ptr,
        shape: n_elements,
        stride: 1,
    };

    let offsets = T::arange(0, BLOCK_SIZE) + block_start * tile.stride;
    let in_bounds = offsets.lt(tile.shape);

    let x = T::load(
        tile.base.add_offsets(offsets),
        Some(in_bounds),
        None,
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
fn test_tile_layout_affine() -> anyhow::Result<()> {
    let kernel = TileLayoutAffineProbe::<f32>::new(1024);
    let teenyc_path = teeny_compiler::compiler::find_teenyc()?;
    let cache_dir =
        std::env::var("TEENYC_CACHE_DIR").unwrap_or_else(|_| "/tmp/teenyc_cache".to_string());
    let compiler = LlvmCompiler::new(teenyc_path, cache_dir)?;
    let target = Target::new(Capability::Sm90);
    let ptx_path: PathBuf = compiler.compile(&kernel, &target, true)?.into();
    let mlir = std::fs::read_to_string(ptx_path.with_extension("mlir"))?;

    assert_debug_snapshot!("tile_layout_affine_source", kernel.source());
    assert_debug_snapshot!("tile_layout_affine_mlir", mlir.trim());

    Ok(())
}
