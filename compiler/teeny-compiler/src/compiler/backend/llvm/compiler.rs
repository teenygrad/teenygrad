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

use sha2::{Digest, Sha256};
use teeny_core::compiler::{Compiler, Target};
use teeny_core::device::program::Kernel;
use tracing::info;

use crate::errors::Result;

#[derive(Debug, Clone)]
pub struct LlvmCompiler {
    teenyc_path: PathBuf,
    cache_dir: PathBuf,
    target_cpu: Option<String>,
    ptx_version: Option<u32>,
}

impl LlvmCompiler {
    pub fn new(teenyc_path: impl Into<PathBuf>, cache_dir: impl Into<PathBuf>) -> Result<Self> {
        let teenyc_path = teenyc_path.into();
        let cache_dir = cache_dir.into();

        if !cache_dir.exists() {
            create_dir_all(&cache_dir)?;
        }

        Ok(Self {
            teenyc_path,
            cache_dir,
            target_cpu: None,
            ptx_version: None,
        })
    }

    pub fn with_target_cpu(mut self, cpu: impl Into<String>) -> Self {
        self.target_cpu = Some(cpu.into());
        self
    }

    /// Override the PTX ISA version `teenyc` stamps into the generated PTX
    /// (encoded as `major*10 + minor`, e.g. `82` for `8.2`), via
    /// `TEENYC_PTX_VERSION`. Without this, `teenyc` picks a conservative
    /// default from the target capability — set this when the deployment
    /// target's exact CUDA version is known and needs a precise match.
    pub fn with_ptx_version(mut self, ptx_version: u32) -> Self {
        self.ptx_version = Some(ptx_version);
        self
    }
}

impl Compiler for LlvmCompiler {
    fn compile(&self, kernel: &impl Kernel, _target: &impl Target, force: bool) -> Result<String> {
        // Hash the kernel id together with target cpu and ptx version so that
        // different targets/overrides each get their own cache entry.
        let effective_id = {
            let mut h = Sha256::new();
            h.update(kernel.id().as_bytes());
            if let Some(cpu) = &self.target_cpu {
                h.update(cpu.as_bytes());
            }
            if let Some(ptx_version) = self.ptx_version {
                h.update(ptx_version.to_le_bytes());
            }
            h.finalize().iter().map(|b| format!("{b:02x}")).collect::<String>()
        };
        let kernel_file_name = format!("{}_{}", kernel.name(), effective_id);
        let kernel_file = self.cache_dir.join(&kernel_file_name).with_extension("rs");
        let output_file = self.cache_dir.join(kernel_file_name).with_extension("o");

        if !output_file.exists() || force {
            anyhow::ensure!(
                self.teenyc_path.exists(),
                "kernel not cached and rustc not found at {:?}; \
                 set TEENYC_PATH to a valid rustc binary",
                self.teenyc_path
            );

            let mut file = File::create(&kernel_file)?;

            info!("Writing kernel code to file");
            file.write_all(teeny_triton::triton_lang::TRITON.as_bytes())?;
            file.write_all(kernel.source().as_bytes())?;

            let mut cmd = Command::new(&self.teenyc_path);
            cmd.arg(&kernel_file)
                .arg("-Copt-level=3")
                .arg("-Zcodegen-backend=mlir")
                .arg("--emit=obj")
                .arg(format!("-o{}", output_file.display()))
                .arg("--target=nvptx64-nvidia-cuda")
                .arg("--crate-type=lib")
                .arg("-C")
                .arg("overflow-checks=off")
                .arg("--frontend=triton")
                .current_dir(&self.cache_dir);
            if let Some(cpu) = &self.target_cpu {
                cmd.arg(format!("-Ctarget-cpu={cpu}"));
            }
            if let Some(ptx_version) = self.ptx_version {
                cmd.env("TEENYC_PTX_VERSION", ptx_version.to_string());
            }
            let output = cmd.output()?;

            if !output.status.success() {
                let stderr = String::from_utf8_lossy(&output.stderr);
                anyhow::bail!("rustc exited with status {}\n{}", output.status, stderr);
            }
        }

        Ok(output_file.to_string_lossy().to_string())
    }
}
