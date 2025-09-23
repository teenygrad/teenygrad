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

use crate::compiler::target::Target;
use crate::error::Result;

mod backend;
mod target;

pub struct Compiler {
    pub src: syn::File,
    pub target: Target,
}

impl Compiler {
    pub fn new(src: &str, target: Target) -> Result<Self> {
        let src = syn::parse_str(src)?;

        Ok(Self { src, target })
    }

    pub fn compile(&self) -> Result<()> {
        todo!()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[cfg(feature = "cuda")]
    fn test_compile_vector_add() {
        use teeny_cuda::target::{Capability, CudaTarget};

        let target = Target::Cuda(CudaTarget::new(Capability::Sm89));
        let kernel = &teeny_kernels::math::add::tensor_add_kernel;
        let compiler = Compiler::new(kernel.block_str, target).unwrap();

        println!("kernel: {:?}", kernel);
        let result = compiler.compile();

        assert!(result.is_ok());
        println!("result: {:?}", result);
    }
}
