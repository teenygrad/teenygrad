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

use std::error::Error;

use dotenv::dotenv;
use insta::assert_snapshot;
use teeny_compiler::compiler::{driver::cuda::compile_kernel, target::cuda::Target};
use teeny_cuda::target::Capability;
use tracing_subscriber::{EnvFilter, fmt};

#[test]
fn test_compile() -> Result<(), Box<dyn Error>> {
    dotenv()?;

    // Initialize logging for the test - only show warnings and errors by default
    // Set RUST_LOG=debug in environment to see debug output
    let _ = fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")),
        )
        .try_init();

    let tensor_add = &teeny_kernels::math::add::VectorAdd::<f32, 1024>::new();
    let target = Target::new(Capability::Sm90);
    let output_file = compile_kernel(tensor_add, &target, true)?;

    let generated_ptx = std::fs::read_to_string(output_file)?;
    assert_snapshot!("vector_add_sm90", generated_ptx.trim());

    Ok(())
}
