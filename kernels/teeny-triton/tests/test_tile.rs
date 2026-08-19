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

//! Smoke test for `Tile<T, D>` (teenygrad-1nr.1 / teenyc-3af.1): construct a
//! local `Tile { tensor, mask: Some(..) }`, copy its `Option` field to a
//! local, and — the harder case teenyc's own fixture doesn't cover — pass
//! the whole `Tile` by value into and back out of a separate `#[inline(always)]`
//! helper function nested in the kernel body, mirroring how a composed
//! tile-op function (teenygrad-1nr.1) would take/return `Tile<T, D>`.

use std::path::PathBuf;

use insta::assert_debug_snapshot;
use teeny_compiler::compiler::backend::llvm::compiler::LlvmCompiler;
use teeny_core::{compiler::Compiler, device::program::Kernel, dtype::Float};
use teeny_cuda::compiler::target::{Capability, Target};
use teeny_macros::kernel;
use teeny_triton::triton::{
    Axis, In, Out, Tile, Triton, types,
    types::{AddOffsets, Comparison},
};

#[kernel]
fn tile_smoke<T: Triton, D: Float, const BLOCK_SIZE: i32>(
    x_ptr: In<T::Pointer<D>>,
    y_ptr: Out<T::Pointer<D>>,
    n_elements: i32,
) where
    T::I32Tensor: types::Tensor<i32, 1>,
    T::I32Tensor: Comparison<i32, BoolTensor = T::BoolTensor>,
    T::Pointer<D>: AddOffsets<i32, 1, T::I32Tensor, Output = T::Tensor<T::Pointer<D>>>,
{
    #[inline(always)]
    fn double_tile<TT: Triton, DD: Float>(t: Tile<TT, DD>) -> Tile<TT, DD> {
        Tile {
            tensor: t.tensor + t.tensor,
            mask: t.mask,
        }
    }

    let pid = T::program_id(Axis::X);
    let block_start = pid * BLOCK_SIZE;
    let offsets = T::arange(0, BLOCK_SIZE) + block_start;
    let in_bounds = offsets.lt(n_elements);

    let loaded = T::load(
        x_ptr.add_offsets(offsets),
        Some(in_bounds),
        None,
        &[],
        None,
        None,
        None,
        false,
    );
    let in_tile = Tile::<T, D> {
        tensor: loaded,
        mask: Some(in_bounds),
    };

    // Pass the whole Tile by value across a real function-call boundary
    // (relies on this getting inlined before/without breaking the aggregate).
    let out_tile = double_tile::<T, D>(in_tile);

    // Field-project the Option field to a local (Tile has no Copy/Clone, so
    // this must be the tile's last use, not read a second time from it).
    let m = out_tile.mask;

    T::store(
        y_ptr.add_offsets(offsets),
        out_tile.tensor,
        m,
        &[],
        None,
        None,
    );
}

#[test]
fn test_tile_smoke() -> anyhow::Result<()> {
    let kernel = TileSmoke::<f32>::new(1024);
    let teenyc_path = teeny_compiler::compiler::find_teenyc()?;
    let cache_dir =
        std::env::var("TEENYC_CACHE_DIR").unwrap_or_else(|_| "/tmp/teenyc_cache".to_string());
    let compiler = LlvmCompiler::new(teenyc_path, cache_dir)?;
    let target = Target::new(Capability::Sm90);
    let ptx_path: PathBuf = compiler.compile(&kernel, &target, true)?.into();
    let mlir = std::fs::read_to_string(ptx_path.with_extension("mlir"))?;

    assert_debug_snapshot!("tile_smoke_source", kernel.source());
    assert_debug_snapshot!("tile_smoke_mlir", mlir.trim());

    Ok(())
}
