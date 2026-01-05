/*
 * Copyright (c) 2025 Teenygrad. All rights reserved.
 *
 * This program is free software: you can redistribute it and/or modify it
 * under the terms of the GNU General Public License as published by the
 * Free Software Foundation, either version 3 of the License, or (at your
 * option) any later version.
 *
 * This program is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program. If not, see <https://www.gnu.org/licenses/>.
 */

use std::env;
use std::fs::File;
use std::io::Write;

use rustc_driver::{TimePassesCallbacks, run_compiler};
use tracing::{debug, info};

use crate::{compiler::target::Target, error::Result};

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
            pub extern "C" fn entry_point(
                n_elements: i32) 
            {
                use triton::llvm::triton::num::*;
                use triton::llvm::triton::pointer::Pointer;
                
                let x_ptr = Pointer(0 as *mut _ );
                let y_ptr = Pointer(0 as *mut _ );
                let output_ptr = Pointer(0 as *mut _ );
                let n_elements = I32(n_elements);

                tensor_add::<triton::llvm::triton::LlvmTriton, F32, 128>(x_ptr, y_ptr, output_ptr, n_elements);
            }
        "#;

        info!("Writing kernel code to file");
        file.write_all(teeny_triton::triton_lang::TRITON.as_bytes())?;
        file.write_all(user_func.as_bytes())?;
        file.write_all(kernel.block_str.as_bytes())?;

        let mut callbacks = TimePassesCallbacks::default();
        let exe_name = "/home/arshadm/.cargo/bin/rustc".to_string(); // AXM FIXME: remove this once API changes
        let output = format!("-o{}", working_dir.join("kernel.ll").display());
        let build_type = "-Copt-level=3".to_string(); // Use opt-level=3 for release build
        let target = "-tnvptx64-nvidia-cuda".to_string();
        let crate_type = "--crate-type=lib".to_string();
        let emit = "--emit=llvm-ir".to_string();
        let overflow_checks = "-C".to_string();
        let overflow_checks_off = "overflow-checks=off".to_string();

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
        run_compiler(
            &[
                exe_name,
                filename.display().to_string(),
                build_type,
                output,
                target,
                crate_type,
                emit,
                overflow_checks,
                overflow_checks_off,
            ],
            &mut callbacks,
        );

        // Restore original directory
        debug!("Restoring original directory: {}", original_dir.display());
        env::set_current_dir(original_dir)?;

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use teeny_cuda::target::{Capability, CudaTarget};

    use super::*;

    #[test]
    fn test_compile() {
        let compiler = LlvmCompiler::new();
        let tensor_add = &teeny_kernels::math::add::tensor_add_kernel;
        let target = Target::Cuda(CudaTarget::new(Capability::Sm89));
        let result = compiler.compile(tensor_add, &target);
        assert!(result.is_ok());
    }
}
