/*
 * Copyright (c) 2026 Teenygrad.
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

use std::{error::Error, path::Path};

use dotenv::dotenv;
use insta::assert_snapshot;
use teeny_compiler::compiler::{Compiler, backend::llvm::compiler::LlvmCompiler, target::Target};
use teeny_cuda::target::{Capability, CudaTarget};
use tracing_subscriber::{EnvFilter, fmt};

#[test]
fn test_compile() -> Result<(), Box<dyn Error>> {
    dotenv().ok();

    // Initialize logging for the test - only show warnings and errors by default
    // Set RUST_LOG=debug in environment to see debug output
    let _ = fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")),
        )
        .try_init();

    let rustc_path = std::env::var("RUSTC_PATH")?;
    println!("RUSTC_PATH: {}", rustc_path);
    let compiler = LlvmCompiler::new(&rustc_path);
    let tensor_add = &teeny_kernels::math::add::TensorAdd::<f32, 1024>::new();
    let target = Target::Cuda(CudaTarget::new(Capability::Sm120));
    compiler.compile(tensor_add, &target, Path::new("/tmp/tensor_add.ptx"))?;

    // Use insta snapshot testing to compare the generated PTX to a reference file
    let generated_ptx = std::fs::read_to_string("/tmp/tensor_add.ptx")?;
    assert_snapshot!("tensor_add_sm120", generated_ptx.trim());

    Ok(())
}
