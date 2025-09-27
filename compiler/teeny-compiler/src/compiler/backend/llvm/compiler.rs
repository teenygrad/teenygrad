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

use std::fs::File;
use std::io::Write;

use rustc_driver::{TimePassesCallbacks, run_compiler};

use crate::{compiler::target::Target, error::Result};

#[derive(Debug, Clone, Default)]
pub struct LlvmCompiler {}

impl LlvmCompiler {
    pub fn new() -> Self {
        Self {}
    }

    pub fn compile(
        &self,
        kernel: &teeny_triton::triton::TritonKernel,
        _target: &Target,
    ) -> Result<()> {
        let filename = "/tmp/kernel.txt".to_string();
        let mut file = File::create(&filename)?;

        file.write_all(teeny_triton::lang::CORE.as_bytes())?;
        file.write_all(teeny_triton::lang::TRITON.as_bytes())?;
        file.write_all(kernel.block_str.as_bytes())?;

        let mut callbacks = TimePassesCallbacks::default();
        let exe_name = "/home/arshadm/.cargo/bin/rustc".to_string(); // AXM FIXME: remove this once API changes
        let output = "-o /tmp/kernel.ll".to_string();
        let target = "-tnvptx64-nvidia-cuda".to_string();
        let crate_type = "--crate-type=lib".to_string();
        println!("target: {}", target);
        run_compiler(
            &[exe_name, filename, output, target, crate_type],
            &mut callbacks,
        );
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
