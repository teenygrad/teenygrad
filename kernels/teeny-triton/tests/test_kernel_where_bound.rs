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

//! `#[kernel]` used to find the hardware type parameter only through an inline `Triton` bound,
//! but an associated-type equality like `I32Tensor = Tensor<i32>` can only be written in a where
//! clause, so a kernel needing one had to bound `T: Triton` twice and trip
//! `clippy::multiple_bound_locations` (teenygrad#118, item 2). Here the bound is only in the
//! where clause, and the equality is used: `T::arange`'s `I32Tensor` is stored as a
//! `T::Tensor<i32>`.

#![deny(clippy::multiple_bound_locations)]

use std::path::PathBuf;

use teeny_compiler::compiler::backend::llvm::compiler::LlvmCompiler;
use teeny_core::{
    compiler::{Compiler, CompilerOptions},
    dtype::Float,
};
use teeny_cuda::compiler::target::{Capability, Target};
use teeny_macros::kernel;
use teeny_triton::triton::{OutPtr, Triton};

#[kernel]
fn where_clause_bound<T, D: Float, const BLOCK_SIZE: i32>(out_ptr: OutPtr<T::Pointer<D>>)
where
    T: Triton<I32Tensor = <T as Triton>::Tensor<i32>>,
{
    let offsets: T::Tensor<i32> = T::arange(0, BLOCK_SIZE);
    let ptrs = T::zeros::<T::Pointer<i32>>(&[BLOCK_SIZE]);
    T::store::<i32, 1>(ptrs, offsets, None, &[0], None, None);
    let _ = out_ptr;
}

#[test]
fn test_kernel_where_clause_bound() -> anyhow::Result<()> {
    dotenv::dotenv().ok();
    let kernel = WhereClauseBound::<f32>::new(128);
    let teenyc_path = teeny_compiler::compiler::find_teenyc()?;
    let cache_dir =
        std::env::var("TEENYC_CACHE_DIR").unwrap_or_else(|_| "/tmp/teenyc_cache".to_string());
    let compiler = LlvmCompiler::new(teenyc_path, cache_dir)?;
    let target = Target::new(Capability::Sm90);
    let options = CompilerOptions::default().with_force(true);
    let ptx_path: PathBuf = compiler.compile(&kernel, &target, &options)?.into();
    let ptx = std::fs::read_to_string(&ptx_path)?;
    assert!(ptx.contains("st.global"), "no global store in PTX:\n{ptx}");
    Ok(())
}
