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
