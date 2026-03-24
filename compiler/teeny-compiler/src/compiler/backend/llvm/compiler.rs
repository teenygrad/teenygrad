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

use std::fs::File;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::{env};

use teeny_triton::TritonKernel;
use tracing::{debug, info};

use crate::compiler::Compiler;
use crate::{compiler::target::Target, error::Result};

#[derive(Debug, Clone)]
pub struct LlvmCompiler {
    rustc_path: PathBuf,
}

impl LlvmCompiler {
    pub fn new(rustc_path: impl Into<PathBuf>) -> Self {
        Self {
            rustc_path: rustc_path.into(),
        }
    }
}

impl Compiler for LlvmCompiler {
    fn compile(&self, kernel: &TritonKernel, _target: &Target, output: &Path) -> Result<()> {
        // Create a proper working directory for rustc
        let temp_dir = env::temp_dir();
        let working_dir = temp_dir.join("teenygrad_rustc");
        debug!("Creating working directory: {}", working_dir.display());
        std::fs::create_dir_all(&working_dir)?;

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

        let output_str = output.display().to_string();

        info!("Working directory: {}", working_dir.display());
        info!("Target: nvptx64-nvidia-cuda");
        info!("Output: {}", output_str);
        debug!(
            "Rustc command: {} {} -o{} --target=nvptx64-nvidia-cuda --crate-type=lib",
            self.rustc_path.display(),
            filename.display(),
            output_str,
        );

        let status = Command::new(&self.rustc_path)
            .env("CFG_VERSION", "tg-1.90.0")
            .arg(&filename)
            .arg("-Copt-level=3")
            .arg(format!("-o{}", output_str))
            .arg("--target=nvptx64-nvidia-cuda")
            .arg("--crate-type=lib")
            .arg("-C")
            .arg("overflow-checks=off")
            .arg("--frontend=triton")
            .current_dir(&working_dir)
            .status()?;

        if !status.success() {
            anyhow::bail!("rustc exited with status {}", status);
        }

        Ok(())
    }
}
