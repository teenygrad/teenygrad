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

use std::fs::{File, create_dir_all};
use std::io::Write;
use std::path::PathBuf;
use std::process::Command;

use teeny_core::compiler::{Compiler, Target};
use teeny_core::context::program::Kernel;
use tracing::info;

use crate::errors::Result;

#[derive(Debug, Clone)]
pub struct LlvmCompiler {
    rustc_path: PathBuf,
    cache_dir: PathBuf,
}

impl LlvmCompiler {
    pub fn new(rustc_path: impl Into<PathBuf>, cache_dir: impl Into<PathBuf>) -> Result<Self> {
        let rustc_path = rustc_path.into();
        let cache_dir = cache_dir.into();

        if !rustc_path.exists() {
            anyhow::bail!("rustc path does not exist: {}", rustc_path.display());
        }

        if !cache_dir.exists() {
            create_dir_all(&cache_dir)?;
        }

        Ok(Self {
            rustc_path,
            cache_dir,
        })
    }
}

impl Compiler for LlvmCompiler {
    fn compile(&self, kernel: &impl Kernel, _target: &impl Target, force: bool) -> Result<String> {
        let id_hex: String = kernel.id().iter().map(|b| format!("{:02x}", b)).collect();
        let kernel_file_name = format!("{}_{}", kernel.name(), id_hex);
        let kernel_file = self.cache_dir.join(&kernel_file_name).with_extension("rs");
        let output_file = self.cache_dir.join(kernel_file_name).with_extension("o");

        if !output_file.exists() || force {
            let mut file = File::create(&kernel_file)?;

            info!("Writing kernel code to file");
            file.write_all(teeny_triton::triton_lang::TRITON.as_bytes())?;
            file.write_all(kernel.source().as_bytes())?;

            let status = Command::new(&self.rustc_path)
                .arg(&kernel_file)
                .arg("-Copt-level=3")
                .arg("-Zcodegen-backend=mlir")
                .arg("--emit=obj")
                .arg(format!("-o{}", output_file.display()))
                .arg("--target=nvptx64-nvidia-cuda")
                .arg("--crate-type=lib")
                .arg("-C")
                .arg("overflow-checks=off")
                .arg("--frontend=triton")
                .current_dir(&self.cache_dir)
                .status()?;

            if !status.success() {
                anyhow::bail!("rustc exited with status {}", status);
            }
        }

        Ok(output_file.to_string_lossy().to_string())
    }
}
