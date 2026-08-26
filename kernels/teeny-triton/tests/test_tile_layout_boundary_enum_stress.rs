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

//! Stress variant of `test_tile_layout_boundary_enum.rs`. That test's
//! `BoundaryMode` was a struct-literal constant (`BoundaryMode::Zero`), so
//! the emitted MLIR showed a single folded load with no branch at all --
//! consistent with either "the enum survived as a struct field and the
//! `match` got constant-folded downstream" or "it never survived as a
//! field in the first place." Inconclusive either way.
//!
//! This variant closes both gaps at once:
//! - The discriminant is derived from a genuine runtime kernel parameter
//!   (`use_clamp`, via a `match` on its value -- not a hardcoded literal),
//!   so it cannot be constant-folded away before reaching codegen.
//! - The compile is run with `-Zmir-enable-passes=-ScalarReplacementOfAggregates`
//!   (see `LlvmCompiler::with_extra_rustc_flags`), the same flag
//!   `../teeny/compiler/rustc_codegen_llvm/tests/test_struct_option.rs`
//!   needs to force MIR's SROA pass off and exercise teenyc-3af.1's *new*
//!   "aggregate survives whole" codegen path -- without it, SROA splits
//!   the struct into separate locals before that path is ever reached,
//!   and a *pre-existing* code path (not what we're testing) handles the
//!   split case instead.
//!
//! **Result: this is a real, precisely-located compiler gap, not a probe
//! artifact.** `teenyc` ICEs with `not yet implemented: Discriminant for
//! non-Option: _10` (`compiler/rustc_codegen_llvm/src/mlir/codegen/triton/mod.rs:2780`,
//! `codegen_assign`'s `Rvalue::Discriminant` arm). That arm special-cases
//! `Option<T>` only (reads `None`/`Some` from `option_table`, lines
//! 2757-2777) and falls through to a bare `todo!()` for any other enum's
//! discriminant -- teenyc-3af.1 generalized struct *fields* to carry an
//! ordinary enum, but never generalized discriminant *reads* beyond the
//! `Option`-specific case, so `match tile.boundary { .. }` has nowhere to
//! go once `boundary`'s value isn't foldable. This test is expected to
//! keep failing until that arm gains a general enum-discriminant path.

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
fn tile_layout_boundary_enum_stress_probe<T: Triton, D: Num, const BLOCK_SIZE: i32>(
    x_ptr: In<T::Pointer<D>>,
    y_ptr: Out<T::Pointer<D>>,
    n_elements: i32,
    use_clamp: i32,
) where
    T::I32Tensor: types::Tensor<i32, 1>,
    T::I32Tensor: Comparison<i32, BoolTensor = T::BoolTensor>,
    T::Pointer<D>: AddOffsets<i32, 1, T::I32Tensor, Output = T::Tensor<T::Pointer<D>>>,
{
    /// Same two-variant probe enum as `test_tile_layout_boundary_enum.rs`.
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

    // Genuinely runtime-decided discriminant, via a plain integer-literal
    // `match` (a language builtin -- no `PartialEq`/`!=` operator overload
    // needed) rather than a struct-literal constant -- can't be folded
    // away before `boundary` becomes a struct field.
    let boundary = match use_clamp {
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
fn test_tile_layout_boundary_enum_stress() -> anyhow::Result<()> {
    let kernel = TileLayoutBoundaryEnumStressProbe::<f32>::new(1024);
    let teenyc_path = teeny_compiler::compiler::find_teenyc()?;
    let cache_dir =
        std::env::var("TEENYC_CACHE_DIR").unwrap_or_else(|_| "/tmp/teenyc_cache".to_string());
    let compiler = LlvmCompiler::new(teenyc_path, cache_dir)?
        .with_extra_rustc_flags(["-Zmir-enable-passes=-ScalarReplacementOfAggregates"]);
    let target = Target::new(Capability::Sm90);
    let ptx_path: PathBuf = compiler.compile(&kernel, &target, true)?.into();
    let mlir = std::fs::read_to_string(ptx_path.with_extension("mlir"))?;

    assert_debug_snapshot!("tile_layout_boundary_enum_stress_source", kernel.source());
    assert_debug_snapshot!("tile_layout_boundary_enum_stress_mlir", mlir.trim());

    Ok(())
}
