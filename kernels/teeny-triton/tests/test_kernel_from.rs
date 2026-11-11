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

//! `#[kernel]` bodies are compiled by `teenyc` as a `no_core` crate whose `From` is the one in
//! `triton/llvm/core.rs`, so every `From` conversion a kernel uses has to exist there too, not
//! just in `std` (teenygrad#118, item 1). `w` is a kernel argument, so no conversion can fold.

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
fn from_conversions<T: Triton, D: Float, const BLOCK_SIZE: i32>(
    out_ptr: OutPtr<T::Pointer<D>>,
    w: i32,
    h: u16,
    x: f32,
) {
    let ptrs = T::zeros::<T::Pointer<D>>(&[BLOCK_SIZE]);
    // The issue's spelling.
    let wmax = T::full(&[BLOCK_SIZE], D::from_f64(f64::from(w - 1)));
    T::store::<D, 1>(ptrs, wmax, None, &[0], None, None);
    // Other lossless conversions std provides.
    T::store::<D, 1>(
        ptrs,
        T::full(&[BLOCK_SIZE], D::from_f64(f64::from(h))),
        None,
        &[0],
        None,
        None,
    );
    T::store::<D, 1>(
        ptrs,
        T::full(&[BLOCK_SIZE], D::from_f64(f64::from(x))),
        None,
        &[0],
        None,
        None,
    );
    let wide = i64::from(w) + i64::from(h);
    T::store::<D, 1>(
        ptrs,
        T::full(&[BLOCK_SIZE], D::from_f64(wide as f64)),
        None,
        &[0],
        None,
        None,
    );
    let _ = out_ptr;
}

#[test]
fn test_kernel_from_conversions() -> anyhow::Result<()> {
    dotenv::dotenv().ok();
    let kernel = FromConversions::<f32>::new(128);
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
