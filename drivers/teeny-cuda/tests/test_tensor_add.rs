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

use teeny_compiler::compiler::driver::cuda::compile_kernel;
use teeny_compiler::compiler::target::cuda::Target;
use teeny_core::context::Context;
use teeny_cuda::context::Cuda;
use teeny_cuda::errors::Result;
use teeny_cuda::target::Capability;

#[test]
fn test_tensor_add() -> Result<()> {
    let cuda_available = Cuda::is_available()?;
    assert!(cuda_available, "CUDA is not available");

    let cuda = Cuda::try_new()?;
    let devices = cuda.list_devices()?;
    assert!(!devices.is_empty(), "No CUDA devices found");

    let _device = cuda.device(&devices[0].id)?;

    let tensor_add = &teeny_kernels::math::add::TensorAdd::<f32, 1024>::new();
    let target = Target::new(Capability::Sm120);
    let output_file = compile_kernel(tensor_add, &target, true)?;
    let generated_ptx = std::fs::read_to_string(output_file)?;

    Ok(())
}
