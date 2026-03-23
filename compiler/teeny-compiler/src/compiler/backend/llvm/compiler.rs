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

use std::env;
use std::fs::File;
use std::io::Write;

use rustc_driver::{Callbacks, run_compiler};
use rustc_interface::interface;
use rustc_session::config;
use rustc_target::spec::Target as RustcTarget;
use tracing::{debug, info};

use crate::{compiler::target::Target, error::Result};

/// Custom callbacks that register the MLIR codegen backend programmatically
struct MlirBackendCallbacks;

impl Callbacks for MlirBackendCallbacks {
    fn config(&mut self, config: &mut interface::Config) {
        debug!("MlirBackendCallbacks::config called - registering backend");
        // Register the MLIR codegen backend programmatically
        // This closure will be called when rustc needs to create the codegen backend
        config.make_codegen_backend = Some(Box::new(
            |_opts: &config::Options, _target: &RustcTarget| {
                debug!("make_codegen_backend closure called - creating MlirCodegenBackend");
                // Create and return the MLIR codegen backend
                rustc_codegen_llvm::mlir::MlirCodegenBackend::new()
            },
        ));
    }
}

#[derive(Debug, Clone, Default)]
pub struct LlvmCompiler {}

impl LlvmCompiler {
    pub fn new() -> Self {
        Self {}
    }

    pub fn compile(&self, kernel: &teeny_triton::TritonKernel, _target: &Target) -> Result<()> {
        // Create a proper working directory for rustc
        let temp_dir = env::temp_dir();
        let working_dir = temp_dir.join("teenygrad_rustc");
        debug!("Creating working directory: {}", working_dir.display());
        std::fs::create_dir_all(&working_dir)?;

        // Change to the working directory to avoid path issues
        let original_dir = env::current_dir()?;
        debug!(
            "Changing directory from {} to {}",
            original_dir.display(),
            working_dir.display()
        );
        env::set_current_dir(&working_dir)?;

        let filename = working_dir.join("kernel.txt");
        debug!("Creating kernel file: {}", filename.display());
        let mut file = File::create(&filename)?;

        let user_func = r#"
            use triton::llvm::triton::num::*;
            use triton::llvm::triton::pointer::Pointer;

            type LlvmTriton = triton::llvm::triton::LlvmTriton;

            #[no_mangle]
            pub extern "C" fn entry_point(x_ptr: *mut f32, y_ptr: *mut f32, output_ptr: *mut f32, n_elements: i32) 
            {    
                let x_ptr = Pointer(x_ptr as *mut _ );
                let y_ptr = Pointer(y_ptr as *mut _ );
                let output_ptr = Pointer(output_ptr as *mut _ );

                tensor_add::<LlvmTriton, f32, 128>(x_ptr, y_ptr, output_ptr, n_elements);
            }
        "#;

        info!("Writing kernel code to file");
        file.write_all(teeny_triton::triton_lang::TRITON.as_bytes())?;
        file.write_all(user_func.as_bytes())?;
        file.write_all(kernel.block_str.as_bytes())?;

        // Use custom callbacks that register the MLIR backend
        let mut callbacks = MlirBackendCallbacks;
        let exe_name = "/home/arshadm/.cargo/bin/rustc".to_string(); // AXM FIXME: remove this once API changes
        let output = format!("-o{}", working_dir.join("kernel.ll").display());
        let build_type = "-Copt-level=3".to_string(); // Use opt-level=3 for release build
        let target = "--target=nvptx64-nvidia-cuda".to_string();
        let crate_type = "--crate-type=lib".to_string();
        // let emit = "--emit=llvm-ir".to_string();
        let overflow_checks = "-C".to_string();
        let overflow_checks_off = "overflow-checks=off".to_string();
        let frontend = "--frontend=triton".to_string();

        info!("Working directory: {}", working_dir.display());
        info!("Target: {}", target);
        info!("Output: {}", output);
        debug!(
            "Rustc command: {} {} {} {} {}",
            exe_name,
            filename.display(),
            output,
            target,
            crate_type
        );

        unsafe {
            env::set_var("CFG_VERSION", "tg-1.90.0");
        }

        // Build the arguments for the compiler
        // Note: We no longer need -Zcodegen-backend flag since we're registering
        // the backend programmatically via the callbacks
        let args = vec![
            exe_name,
            filename.display().to_string(),
            build_type,
            output,
            target,
            crate_type,
            // emit,
            overflow_checks,
            overflow_checks_off,
            frontend,
        ];

        run_compiler(&args, &mut callbacks);

        // Restore original directory
        debug!("Restoring original directory: {}", original_dir.display());
        env::set_current_dir(original_dir)?;

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use teeny_cuda::target::{Capability, CudaTarget};
    use tracing_subscriber::{EnvFilter, fmt};

    use super::*;

    #[test]
    fn test_compile() {
        // Initialize logging for the test - only show warnings and errors by default
        // Set RUST_LOG=debug in environment to see debug output
        let _ = fmt()
            .with_env_filter(
                EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")),
            )
            .try_init();

        let compiler = LlvmCompiler::new();
        let tensor_add = &teeny_kernels::math::add::tensor_add_kernel;
        let target = Target::Cuda(CudaTarget::new(Capability::Sm89));
        let result = compiler.compile(tensor_add, &target);
        assert!(result.is_ok());
    }
}
