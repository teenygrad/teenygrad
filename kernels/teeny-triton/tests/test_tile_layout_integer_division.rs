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

//! Probes the "integer division" `AxisIndex::Div` case of a CuTe-style
//! `Tile<T,D>` layout: an offset computed as `coord / divisor` (the
//! `upsample_nearest2d`-style index mapping) instead of `coord * stride`,
//! with the divisor read off a kernel-local struct field.
//!
//! Plain `i32` division is already proven end-to-end in real kernels
//! (`conv2d_forward`'s `pid / num_ow_tiles`, `C_IN / G`, etc.), so unlike
//! the other two probes this isn't expected to find a new compiler gap --
//! it's the control case, confirming the composition (division read off a
//! struct field, inside an offset expression, through a real load) works
//! the same way once wrapped in a `Tile`-like descriptor.

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
fn tile_layout_integer_division_probe<T: Triton, D: Num, const BLOCK_SIZE: i32>(
    x_ptr: In<T::Pointer<D>>,
    y_ptr: Out<T::Pointer<D>>,
    n_elements: i32,
    divisor: i32,
) where
    T::I32Tensor: types::Tensor<i32, 1>,
    T::I32Tensor: Comparison<i32, BoolTensor = T::BoolTensor>,
    T::Pointer<D>: AddOffsets<i32, 1, T::I32Tensor, Output = T::Tensor<T::Pointer<D>>>,
{
    /// Local items don't inherit the enclosing function's generics (see
    /// `test_kitchen_sink.rs`'s nested `combine_num`), so this declares
    /// its own `TT`/`DD`.
    struct DivTile<TT: Triton, DD: Num> {
        base: TT::Pointer<DD>,
        divisor: i32,
    }

    let pid = T::program_id(Axis::X);
    let block_start = pid * BLOCK_SIZE;

    let tile = DivTile::<T, D> {
        base: x_ptr,
        divisor,
    };

    // AxisIndex::Div: offset = coord / divisor -- as opposed to
    // AxisIndex::Stride's coord * stride (test_tile_layout_affine.rs).
    let scaled_start = block_start / tile.divisor;
    let offsets = T::arange(0, BLOCK_SIZE) + scaled_start;
    let in_bounds = offsets.lt(n_elements);

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
fn test_tile_layout_integer_division() -> anyhow::Result<()> {
    let kernel = TileLayoutIntegerDivisionProbe::<f32>::new(1024);
    let teenyc_path = teeny_compiler::compiler::find_teenyc()?;
    let cache_dir =
        std::env::var("TEENYC_CACHE_DIR").unwrap_or_else(|_| "/tmp/teenyc_cache".to_string());
    let compiler = LlvmCompiler::new(teenyc_path, cache_dir)?;
    let target = Target::new(Capability::Sm90);
    let ptx_path: PathBuf = compiler.compile(&kernel, &target, true)?.into();
    let mlir = std::fs::read_to_string(ptx_path.with_extension("mlir"))?;

    assert_debug_snapshot!("tile_layout_integer_division_source", kernel.source());
    assert_debug_snapshot!("tile_layout_integer_division_mlir", mlir.trim());

    Ok(())
}
